// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "search.h"
#include "constrained_logits_processor.h"
#include "models/model.h"
#include "mtp_generator.h"
#include "speculative_sampling.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>

namespace Generators {

MtpGenerator::MtpGenerator(const Model& main_model, const Model& mtp_model, const GeneratorParams& params)
    : main_model_{main_model}, mtp_model_{mtp_model} {
  ValidateMtpPair(main_model_, mtp_model_, params);
  if (main_model_.config_->model.decoder.hidden_size != mtp_model_.config_->model.decoder.hidden_size) {
    throw std::runtime_error("MtpGenerator requires matching main-model and MTP-head hidden sizes");
  }
  const std::string& main_hidden_states = main_model_.config_->model.mtp.main_hidden_states;
  if (main_hidden_states.empty() || !main_model_.session_info_.HasOutput(main_hidden_states)) {
    throw std::runtime_error(
        "MtpGenerator: model.mtp.main_hidden_states must name a main-model output");
  }
  const auto main_hidden_type = main_model_.session_info_.GetOutputDataType(
      main_hidden_states);
  const auto mtp_hidden_type = mtp_model_.session_info_.GetInputDataType(
      mtp_model_.config_->model.decoder.inputs.hidden_states);
  if (main_hidden_type != mtp_hidden_type) {
    throw std::runtime_error("MtpGenerator requires matching hidden-state tensor types");
  }
  // Number of speculative draft tokens per step (N), shared with draft-model speculative decoding
  // via speculative.max_draft_tokens. N=1 is the original single-token fast path; N>1 chains the
  // single MTP module N times (feeding its own post-norm hidden back), as vLLM's
  // AutoRegressiveSpeculator does. Chaining needs the head to emit that hidden, so a head exported
  // without `mtp_emit_hidden=true` can only run N=1.
  num_speculative_tokens_ = std::max(1, params.speculative.max_draft_tokens);
  const std::string& head_hidden_states = main_model_.config_->model.mtp.outputs.hidden_states;
  if (num_speculative_tokens_ > 1 &&
      (head_hidden_states.empty() || !mtp_model_.session_info_.HasOutput(head_hidden_states))) {
    throw std::runtime_error(
        "MtpGenerator: max_draft_tokens=" + std::to_string(num_speculative_tokens_) +
        " requires the configured MTP feedback output '" + head_hidden_states +
        "' (model.mtp.outputs.hidden_states)");
  }
  // Chained greedy drafts stay on device between head forwards on CUDA, avoiding a
  // device->host sync per draft. Other devices fall back to the host path.
  device_draft_chain_ = mtp_model_.p_device_->GetType() == DeviceType::CUDA;
  // Chunked prefill. The windowed recurrent state is a fixed [B, W, ...] buffer, so it no longer
  // scales with the prompt, but a single-shot forward over a long prompt still blows up the ORT
  // activation arena (measured: 54 GB chunked vs 94 GB unchunked on a 2.8k-token prompt). Feed
  // the prompt in chunks so peak memory stays bounded; only the last chunk's outputs are consumed.
  if (params.search.chunk_size.has_value()) {
    prefill_chunk_ = static_cast<int>(*params.search.chunk_size);
    if (prefill_chunk_ < 0) prefill_chunk_ = 0;
    prefill_chunk_explicit_ = true;
  }
  // Keep an MTP-owned parameter object so extending graph capture to the verify shapes does not
  // mutate caller-owned params that may be reused to create an ordinary Generator.
  main_params_ = std::make_shared<GeneratorParams>(main_model_);
  main_params_->search = params.search;
  main_params_->max_batch_size = params.max_batch_size;
  main_params_->use_graph_capture = params.use_graph_capture;
  main_params_->max_graph_capture_length = num_speculative_tokens_ + 1;
  main_params_->use_multi_profile = params.use_multi_profile;
  main_params_->p_device = params.p_device;
  main_params_->guidance_type = params.guidance_type;
  main_params_->guidance_data = params.guidance_data;
  main_params_->guidance_ff_tokens_enabled = params.guidance_ff_tokens_enabled;

  main_ = CreateGenerator(main_model_, *main_params_);
  // A windowed recurrent state holds W per-token states, and the verify forward is N+1 wide, so
  // the window caps N at W-1. Exceeding it fails mid-generation inside CropToPosition, so clamp
  // here instead. Graph capture was sized from the requested N, which is a harmless upper bound.
  if (const int64_t window = main_->RecurrentStateWindow(); window > 1) {
    num_speculative_tokens_ = std::min(num_speculative_tokens_, static_cast<int>(window) - 1);
  }
  // Default the chunking on for windowed-state models only (they are the ones running long
  // prompts through the MTP loop). A 1024-token chunk keeps activation memory bounded without
  // splitting common 1K prompts into several underfilled forwards.
  if (!prefill_chunk_explicit_ && main_->CanCropRecurrentState()) prefill_chunk_ = 1024;
  mtp_params_ = std::make_shared<GeneratorParams>(mtp_model_);
  mtp_params_->search = params.search;
  // CUDA-graph capture on the MTP head: the head is a single standard-attention layer (KV
  // share-buffer, NO GatedDeltaNet recurrent state), so it is graph-capture-safe just like GQA.
  // Its hidden_states input already stages the (per-step rebound) source into a stable static
  // device buffer under capture (see HiddenStatesInputs::Update), so replay reads a stable
  // address. Across the draft / batched-refeed / DraftTwo paths the head runs sequence lengths
  // 1..N+1, so size the captured range to N+1 (matching the main model). GeneratorParams(mtp_model_)
  // already set use_graph_capture from the head's session_options (enable_cuda_graph); honor it
  // here instead of the previous unconditional disable.
  mtp_params_->max_graph_capture_length =
      mtp_params_->use_graph_capture ? (num_speculative_tokens_ + 1) : 1;
  mtp_ = CreateGenerator(mtp_model_, *mtp_params_);

  hidden_size_ = main_model_.config_->model.decoder.hidden_size;
  vocab_size_ = main_model_.config_->model.vocab_size;
  max_length_ = params.search.max_length;
  eos_token_ids_ = main_model_.config_->model.eos_token_id;

  // Speculative sampling: when the caller requests randomized sampling (do_sample) with a positive
  // temperature, drafts are sampled from their truncated distribution and accepted via the
  // Leviathan/Chen rejection test, so MTP draws from the same distribution as plain top-k/top-p
  // decoding (see GenerateStepMultiSample). Greedy (do_sample=false or temperature==0) keeps the
  // original argmax accept path.
  sampling_ = params.search.do_sample && params.search.temperature > 0.0f;
  top_k_ = params.search.top_k;
  top_p_ = params.search.top_p;
  temperature_ = params.search.temperature;
  main_logits_penalties_ = std::make_unique<LogitsPenaltyProcessor>(
      vocab_size_, params.search.repetition_penalty, params.search.min_length,
      params.search.no_repeat_ngram_size, main_model_.config_->model.eos_token_id);
  if (sampling_) {
    if (params.search.random_seed == -1) {
      std::random_device rd;
      rng_.seed(rd());
    } else {
      rng_.seed(static_cast<uint32_t>(params.search.random_seed));
    }
  }

  // Reusable [1, 1, hidden] device buffer for the on-device hidden-state handoff.
  hidden_slice_ = std::make_shared<Tensor>(
      main_model_.p_device_inputs_,
      main_model_.session_info_.GetOutputDataType(main_model_.config_->model.mtp.main_hidden_states));
  const std::array<int64_t, 3> slice_shape{1, 1, hidden_size_};
  hidden_slice_->CreateTensor(slice_shape);

  // Reusable [1, 2, hidden] device buffer for the batched 2-token draft (post-accept KV-advance
  // fused with the next step's draft into one MTP forward).
  hidden_slice2_ = std::make_shared<Tensor>(
      main_model_.p_device_inputs_,
      main_model_.session_info_.GetOutputDataType(main_model_.config_->model.mtp.main_hidden_states));
  const std::array<int64_t, 3> slice2_shape{1, 2, hidden_size_};
  hidden_slice2_->CreateTensor(slice2_shape);

  // Multi-token (N>1) scratch: the head's own hidden output (chain feedback) and a re-feed buffer
  // used to re-materialize accepted drafts in the head KV with the main model's hidden states.
  // Speculative sampling (any N, including N=1) uses the same chained draft/verify machinery.
  if (num_speculative_tokens_ > 1 || sampling_) {
    head_out_hidden_ = std::make_shared<Tensor>(
        mtp_model_.p_device_inputs_,
        mtp_model_.session_info_.GetInputDataType(mtp_model_.config_->model.decoder.inputs.hidden_states));
    head_out_hidden_->CreateTensor(slice_shape);
    refeed_hidden_ = std::make_shared<Tensor>(
        mtp_model_.p_device_inputs_,
        mtp_model_.session_info_.GetInputDataType(mtp_model_.config_->model.decoder.inputs.hidden_states));
    refeed_hidden_->CreateTensor(slice_shape);
    drafts_.resize(num_speculative_tokens_);
    drafts_device_ = mtp_model_.p_device_->Allocate<int32_t>(num_speculative_tokens_);
    verify_tokens_.resize(num_speculative_tokens_ + 1);
    verify_argmax_.resize(num_speculative_tokens_ + 1);
    merged_tokens_.resize(static_cast<size_t>(num_speculative_tokens_) + 1);
    // Per-size [1,j,H] hidden buffers (j=1..N+1) so the post-verify head refeed of the j accepted
    // drafts runs as ONE batched head forward instead of one forward per token. The greedy path
    // additionally fuses the next step's first draft into that forward, so it needs j = a+1 rows
    // (up to N+1); the sampling path uses j = a (up to N).
    refeed_multi_.resize(static_cast<size_t>(num_speculative_tokens_) + 2);
    for (int j = 1; j <= num_speculative_tokens_ + 1; ++j) {
      refeed_multi_[j] = std::make_shared<Tensor>(
          mtp_model_.p_device_inputs_,
          mtp_model_.session_info_.GetInputDataType(mtp_model_.config_->model.decoder.inputs.hidden_states));
      const std::array<int64_t, 3> sh{1, j, hidden_size_};
      refeed_multi_[j]->CreateTensor(sh);
    }
  }
  if (sampling_) {
    draft_idx_.resize(num_speculative_tokens_);
    draft_prob_.resize(num_speculative_tokens_);
    target_idx_.resize(num_speculative_tokens_ + 1);
    target_prob_.resize(num_speculative_tokens_ + 1);
  }
}

void MtpGenerator::ExtractHiddenPosition(OrtValue* hidden, int position) {
  // hidden is [1, S, H] on the model device; copy row `position` into hidden_slice_ ([1,1,H]).
  CopyHiddenRow(hidden, position, *hidden_slice_);
}

void MtpGenerator::CopyHiddenRow(OrtValue* hidden, int position, Tensor& dst) {
  CopyTensorRow(*hidden, position, dst, *main_model_.p_device_);
}

int32_t MtpGenerator::DraftHeadStep(int32_t token, bool need_draft) {
  // The head's `hidden_states` input must already be set by the caller (SetHiddenStates). Append
  // `token` (the head KV grows by one), capture the head's own post-final-norm output for the next
  // chained step, and return the greedy draft for the token after `token`.
  std::array<int32_t, 1> tok{token};
  mtp_->AppendTokens(cpu_span<const int32_t>(tok));
  stats_.draft_forward_passes++;
  ++head_len_;

  int32_t draft = 0;
  if (need_draft) {
    // ArgMax synchronizes the head's logits producer before the feedback D2D copy below.
    auto logits_span = mtp_->GetLogits();  // fp32, last token, [1, V]
    if (!mtp_model_.p_device_->ArgMax(logits_span.Span().data(), Ort::TypeToTensorType<float>, 1, vocab_size_, &draft)) {
      auto logits = logits_span.CopyDeviceToCpu();  // host fallback
      draft = ArgmaxRow(logits.data(), vocab_size_);
    }
  }

  // Capture the head's recurrent feedback hidden (hidden_states_out, the single processed row).
  CaptureHeadFeedbackHidden();
  return draft;
}

void MtpGenerator::CaptureDraftToDevice(DeviceSpan<int32_t> draft) {
  auto logits_span = mtp_->GetLogits();  // fp32, last token, [1, V]
  if (mtp_model_.p_device_->ArgMaxDevice(logits_span.Span().data(), Ort::TypeToTensorType<float>, 1,
                                         vocab_size_, draft)) {
    return;
  }

  int32_t host_draft = 0;
  if (!mtp_model_.p_device_->ArgMax(logits_span.Span().data(), Ort::TypeToTensorType<float>, 1,
                                    vocab_size_, &host_draft)) {
    auto logits = logits_span.CopyDeviceToCpu();
    host_draft = ArgmaxRow(logits.data(), vocab_size_);
  }
  // A subspan shares its backing allocation, and CopyCpuToDevice transfers that whole allocation.
  // Preserve any earlier chained drafts before updating this slot on an unsupported device.
  auto draft_cpu = draft.CopyDeviceToCpu();
  draft_cpu[0] = host_draft;
  draft.CopyCpuToDevice();
}

void MtpGenerator::DraftHeadStepToDevice(int32_t token, DeviceSpan<int32_t> draft) {
  std::array<int32_t, 1> tok{token};
  mtp_->AppendTokens(cpu_span<const int32_t>(tok));
  stats_.draft_forward_passes++;
  ++head_len_;
  CaptureDraftToDevice(draft);
  CaptureHeadFeedbackHidden();
}

void MtpGenerator::DraftHeadStepToDevice(DeviceSpan<int32_t> token, DeviceSpan<int32_t> draft) {
  mtp_->AppendTokens(token);
  stats_.draft_forward_passes++;
  ++head_len_;
  CaptureDraftToDevice(draft);
  CaptureHeadFeedbackHidden();
}

void MtpGenerator::CaptureHeadFeedbackHidden() {
  // Capture the head's own post-final-norm output (hidden_states_out, the single processed row)
  // into head_out_hidden_ for the next chained draft step.
  const std::string& head_hidden_states = main_model_.config_->model.mtp.outputs.hidden_states;
  OrtValue* head_hidden = mtp_->state_->GetOutput(head_hidden_states.c_str());
  if (head_hidden == nullptr) {
    throw std::runtime_error(
        "MtpGenerator: multi-token speculation requires the MTP head exported with "
        "mtp_emit_hidden=true (missing configured output '" +
        head_hidden_states +
        "').");
  }
  const size_t row_bytes = head_out_hidden_->GetByteSpan().size();
  // The head may have processed several tokens in one forward (the fused refeed+draft step), in
  // which case hidden_states_out is [1,S,H]. The chain always feeds forward from the LAST row.
  auto hh_info = head_hidden->GetTensorTypeAndShapeInfo();
  const auto hh_shape = hh_info->GetShape();
  const size_t hh_rows =
      hh_shape.size() >= 2 ? static_cast<size_t>(hh_shape[hh_shape.size() - 2]) : 1;
  const size_t last_row_offset = (hh_rows > 0 ? hh_rows - 1 : 0) * row_bytes;
  auto dst = head_out_hidden_->GetByteSpan();
  if (head_hidden->GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_CPU) {
    // ExtraOutputs are not IO-bound, so ORT may allocate this output on the CPU. Stage it through
    // the destination's pinned host buffer rather than passing a host pointer to cudaMemcpy D2D.
    auto dst_cpu = dst.CpuSpan();
    std::memcpy(dst_cpu.data(),
                static_cast<const uint8_t*>(head_hidden->GetTensorRawData()) + last_row_offset,
                row_bytes);
    dst.CopyCpuToDevice();
  } else {
    auto src = ByteWrapTensor(*mtp_model_.p_device_, *head_hidden);
    dst.CopyFrom(src.subspan(last_row_offset, row_bytes));
  }
}

int32_t MtpGenerator::DraftHeadStepMulti(const int32_t* tokens, int count) {
  // The head's `hidden_states` input must already be set by the caller to a [1,count,H] buffer
  // whose rows are MAIN-model hidden states for consecutive positions. One forward appends all
  // `count` tokens to the head KV; only the last row's logits are needed (the draft for the token
  // after tokens[count-1]) -- the earlier rows exist purely to re-materialize the accepted drafts.
  mtp_->AppendTokens(cpu_span<const int32_t>(tokens, static_cast<size_t>(count)));
  stats_.draft_forward_passes++;
  head_len_ += static_cast<size_t>(count);

  // ArgMax synchronizes the head's logits producer before the feedback D2D copy below.
  auto logits_span = mtp_->GetLogits();  // fp32, last token, [1, V]
  int32_t draft = 0;
  if (!mtp_model_.p_device_->ArgMax(logits_span.Span().data(), Ort::TypeToTensorType<float>, 1,
                                    vocab_size_, &draft)) {
    auto logits = logits_span.CopyDeviceToCpu();  // host fallback
    draft = ArgmaxRow(logits.data(), vocab_size_);
  }

  CaptureHeadFeedbackHidden();
  return draft;
}

void MtpGenerator::DraftHeadStepMultiToDevice(const int32_t* tokens, int count, DeviceSpan<int32_t> draft) {
  mtp_->AppendTokens(cpu_span<const int32_t>(tokens, static_cast<size_t>(count)));
  stats_.draft_forward_passes++;
  head_len_ += static_cast<size_t>(count);
  CaptureDraftToDevice(draft);
  CaptureHeadFeedbackHidden();
}

int32_t MtpGenerator::DraftHeadStepSample(int32_t token, int k) {
  // Same KV-advance + hidden-feedback capture as DraftHeadStep, but the draft is SAMPLED from its
  // truncated distribution q (top_k/top_p/temperature). The sparse q (kept ids + renormalized
  // probs) is stored in draft_idx_[k]/draft_prob_[k] so the verify step can compute the
  // min(1, p(d)/q(d)) acceptance test with a cheap sparse probability lookup.
  std::array<int32_t, 1> tok{token};
  mtp_->AppendTokens(cpu_span<const int32_t>(tok));
  stats_.draft_forward_passes++;
  ++head_len_;

  auto logits_span = mtp_->GetLogits();  // fp32, last token, [1, V]
  const int fp32_type = static_cast<int>(Ort::TypeToTensorType<float>);
  if (TopKScoresRows(logits_span.Span().data(), fp32_type, 1, *mtp_model_.p_device_)) {
    SparseFromTopKRow(0, draft_idx_[k], draft_prob_[k]);
  } else {
    auto cpu = logits_span.CopyDeviceToCpu();
    ComputeSampledCategorical(std::span<const float>(cpu.data(), static_cast<size_t>(vocab_size_)),
                              top_k_, top_p_, temperature_, sampled_scratch_);
    draft_idx_[k] = sampled_scratch_.indices;
    draft_prob_[k] = sampled_scratch_.probs;
  }
  int32_t draft = SampleSparseToken(draft_idx_[k], draft_prob_[k], rng_);

  CaptureHeadFeedbackHidden();
  return draft;
}

const float* MtpGenerator::MainLogitsRowsCpu(int first_row, int num_rows) {
  OrtValue* raw = main_->state_->GetOutput(main_model_.config_->model.decoder.outputs.logits.c_str());
  // Cast to fp32 on device (cheap kernel), then copy only the requested rows to the host.
  Cast(*raw, logits_fp32_, *main_model_.p_device_, Ort::TypeToTensorType<float>);
  auto span = ByteWrapTensor(*main_model_.p_device_, *logits_fp32_);
  const size_t row_bytes = static_cast<size_t>(vocab_size_) * sizeof(float);
  auto rows = span.subspan(static_cast<size_t>(first_row) * row_bytes,
                           static_cast<size_t>(num_rows) * row_bytes);
  auto cpu = rows.CopyDeviceToCpu();
  const float* data = reinterpret_cast<const float*>(cpu.data());
  main_logits_cpu_.assign(data, data + static_cast<size_t>(num_rows) * vocab_size_);
  return main_logits_cpu_.data();
}

bool MtpGenerator::TopKScoresRows(const void* logits, int onnx_type, int num_rows, DeviceInterface& dev) {
  // Mirror ComputeSampledCategorical's `apply_topk = top_k > 1` gate: top_k of 0 or 1 means "no
  // top-k truncation" (pure nucleus / full softmax), which the device top-k cannot express -- it
  // would silently collapse the distribution to k entries. Fall back to the host path instead.
  if (top_k_ <= 1) return false;
  topk_k_ = std::min(top_k_, vocab_size_);
  const size_t need = static_cast<size_t>(num_rows) * topk_k_;
  topk_tok_scratch_.resize(need);
  topk_score_scratch_.resize(need);
  return dev.TopKScores(logits, static_cast<ONNXTensorElementDataType>(onnx_type), num_rows, vocab_size_,
                        topk_k_, topk_tok_scratch_.data(), topk_score_scratch_.data());
}

std::span<const float> MtpGenerator::ProcessMainLogitsRow(std::span<const float> logits, int row) {
  if (!main_logits_penalties_->IsActive())
    return logits;

  OrtValue* raw = main_->state_->GetOutput(main_model_.config_->model.decoder.outputs.logits.c_str());
  const auto shape = raw->GetTensorTypeAndShapeInfo()->GetShape();
  if (shape.size() < 2 || row < 0 || row >= shape[shape.size() - 2])
    throw std::runtime_error("MtpGenerator: target logits row is outside the model output");

  auto sequence = main_->GetSequence(0).CopyDeviceToCpu();
  const size_t output_rows = static_cast<size_t>(shape[shape.size() - 2]);
  if (sequence.size() < output_rows)
    throw std::runtime_error("MtpGenerator: target logits contain more rows than the token sequence");
  const size_t prefix_length = sequence.size() - output_rows + static_cast<size_t>(row) + 1;
  return main_logits_penalties_->Apply(logits, static_cast<int>(prefix_length), sequence.first(prefix_length));
}

void MtpGenerator::SparseFromTopKRow(int row, std::vector<int32_t>& idx, std::vector<float>& prob) {
  // The device returns the k top scores sorted descending; apply temperature softmax over them and
  // a top-p nucleus cutoff -- identical to ComputeSampledCategorical's top-k branch, but over only
  // the k values that left the GPU (no full-vocab host work).
  const int k = topk_k_;
  const int32_t* toks = topk_tok_scratch_.data() + static_cast<size_t>(row) * k;
  const float* scs = topk_score_scratch_.data() + static_cast<size_t>(row) * k;
  const float inv_temp = 1.0f / temperature_;
  topk_prob_scratch_.resize(static_cast<size_t>(k));
  const float maxs = scs[0];
  float sum = 0.0f;
  for (int j = 0; j < k; ++j) {
    const float e = std::exp((scs[j] - maxs) * inv_temp);
    topk_prob_scratch_[j] = e;
    sum += e;
  }
  const float invs = sum > 0.0f ? 1.0f / sum : 0.0f;
  for (int j = 0; j < k; ++j) topk_prob_scratch_[j] *= invs;
  int keep = k;
  if (top_p_ > 0.0f && top_p_ < 1.0f) {
    float cum = 0.0f;
    for (int j = 0; j < k; ++j) {
      cum += topk_prob_scratch_[j];
      if (cum >= top_p_) {
        keep = j + 1;
        break;
      }
    }
  }
  float ks = 0.0f;
  for (int j = 0; j < keep; ++j) ks += topk_prob_scratch_[j];
  const float ki = ks > 0.0f ? 1.0f / ks : 0.0f;
  idx.assign(toks, toks + keep);
  prob.resize(static_cast<size_t>(keep));
  for (int j = 0; j < keep; ++j) prob[j] = topk_prob_scratch_[j] * ki;
}

void MtpGenerator::ArgmaxMainRows(int first_row, int num_rows, int32_t* out) {
  OrtValue* raw = main_->state_->GetOutput(main_model_.config_->model.decoder.outputs.logits.c_str());
  auto info = raw->GetTensorTypeAndShapeInfo();
  const ONNXTensorElementDataType type = info->GetElementType();

  if (main_logits_penalties_->IsActive()) {
    const float* rows = MainLogitsRowsCpu(first_row, num_rows);
    for (int r = 0; r < num_rows; ++r) {
      auto processed = ProcessMainLogitsRow(
          std::span<const float>(rows + static_cast<size_t>(r) * vocab_size_, static_cast<size_t>(vocab_size_)),
          first_row + r);
      out[r] = ArgmaxRow(processed.data(), vocab_size_);
    }
    return;
  }

  // The CUDA distributed-select Top-K implementation is batch-1 only. The N=1 MTP verify uses
  // two rows and is covered by its existing tuned path, but N>1 verifies have 3+ rows. Submit
  // those rows independently so each invocation uses the proven batch-1 path while keeping the
  // full logits device-resident. This is a compatibility bridge until Top-K has a native batched
  // argmax path for the large Qwen vocabulary.
  if (num_rows > 2) {
    const uint8_t* base = static_cast<const uint8_t*>(raw->GetTensorRawData());
    const size_t row_bytes = static_cast<size_t>(vocab_size_) * Ort::SizeOf(type);
    bool all_device = true;
    for (int row = 0; row < num_rows; ++row) {
      const void* row_ptr = base + static_cast<size_t>(first_row + row) * row_bytes;
      if (!main_model_.p_device_->ArgMax(row_ptr, type, 1, vocab_size_, out + row)) {
        all_device = false;
        break;
      }
    }
    if (all_device) return;
  }

  // Fast path: argmax the rows on-device with the high-performance Top-K kernel (k=1). Only the
  // small token ids are copied to the host -- the full [1,S,V] logits never leave the GPU.
  const uint8_t* base = static_cast<const uint8_t*>(raw->GetTensorRawData());
  const void* row_ptr = base + static_cast<size_t>(first_row) * vocab_size_ * Ort::SizeOf(type);
  if (main_model_.p_device_->ArgMax(row_ptr, type, num_rows, vocab_size_, out))
    return;

  // Host fallback (e.g. CPU device): cast the logits to fp32, copy to the host, argmax each row.
  Cast(*raw, logits_fp32_, *main_model_.p_device_, Ort::TypeToTensorType<float>);
  auto span = ByteWrapTensor(*main_model_.p_device_, *logits_fp32_);
  auto cpu = span.CopyDeviceToCpu();
  const float* data = reinterpret_cast<const float*>(cpu.data());
  for (int r = 0; r < num_rows; ++r)
    out[r] = ArgmaxRow(data + static_cast<size_t>(first_row + r) * vocab_size_, vocab_size_);
}

int32_t MtpGenerator::DraftNextToken(OrtValue* /*unused*/, int32_t token, bool need_draft) {
  // hidden_slice_ already holds the hidden state paired with `token`. Feed (hidden, token) to the
  // MTP head; its KV cache accumulates, so this is an O(1) incremental draft step.
  mtp_->SetHiddenStates(hidden_slice_);
  std::array<int32_t, 1> tok{token};
  mtp_->AppendTokens(cpu_span<const int32_t>(tok));
  stats_.draft_forward_passes++;
  if (!need_draft) {
    // KV-advance only (e.g. after an accepted draft): skip the full-vocab argmax + stream sync.
    return 0;
  }
  auto logits_span = mtp_->GetLogits();  // fp32, last token, [1, V]
  int32_t draft = 0;
  if (mtp_model_.p_device_->ArgMax(logits_span.Span().data(), Ort::TypeToTensorType<float>, 1, vocab_size_, &draft))
    return draft;
  auto logits = logits_span.CopyDeviceToCpu();  // host fallback
  return ArgmaxRow(logits.data(), vocab_size_);
}

int32_t MtpGenerator::DraftTwo(OrtValue* hidden, int32_t tok0, int32_t tok1) {
  // Populate the [1,2,H] hidden buffer: row 0 = hidden@position L (pairs with tok0), row 1 =
  // hidden@position L+1 (pairs with tok1). `hidden` is the main model's [1,S,H] verify output.
  auto src = ByteWrapTensor(*main_model_.p_device_, *hidden);
  const size_t row_bytes = hidden_slice_->GetByteSpan().size();  // bytes of one [1,1,H] row
  auto dst = hidden_slice2_->GetByteSpan();
  dst.subspan(0, row_bytes).CopyFrom(src.subspan(0, row_bytes));                  // row 0 <- hidden@0
  dst.subspan(row_bytes, row_bytes).CopyFrom(src.subspan(row_bytes, row_bytes));  // row 1 <- hidden@1

  // One 2-token MTP forward: feeds tok0 (KV-advance) and tok1 (the next committed token); the
  // last-position logits give the draft for the token after tok1.
  mtp_->SetHiddenStates(hidden_slice2_);
  std::array<int32_t, 2> toks{tok0, tok1};
  mtp_->AppendTokens(cpu_span<const int32_t>(toks));
  stats_.draft_forward_passes++;
  auto logits_span = mtp_->GetLogits();  // fp32, last token, [1, V]
  int32_t draft = 0;
  if (mtp_model_.p_device_->ArgMax(logits_span.Span().data(), Ort::TypeToTensorType<float>, 1, vocab_size_, &draft))
    return draft;
  auto logits = logits_span.CopyDeviceToCpu();  // host fallback
  return ArgmaxRow(logits.data(), vocab_size_);
}

void MtpGenerator::AppendTokens(cpu_span<const int32_t> input_ids) {
  if (primed_)
    throw std::runtime_error("MtpGenerator: AppendTokens can only be called once");

  // Chunked prefill: bounds the ORT activation arena of the prompt forward (see the constructor).
  // Off (single forward) when the chunk size is 0 or the prompt fits in one chunk.
  const size_t total = input_ids.size();
  size_t tail = total;
  if (prefill_chunk_ > 0 && total > static_cast<size_t>(prefill_chunk_)) {
    const size_t chunk = static_cast<size_t>(prefill_chunk_);
    for (size_t off = 0; off < total; off += chunk) {
      const size_t n = std::min(chunk, total - off);
      main_->AppendTokens(cpu_span<const int32_t>(input_ids.data() + off, n));
      tail = n;  // outputs below are read from the final chunk's forward
    }
  } else {
    main_->AppendTokens(input_ids);
  }
  length_ = input_ids.size();
  sequence_.assign(input_ids.begin(), input_ids.end());
  emitted_sequence_ = sequence_;
  if (sequence_.size() >= static_cast<size_t>(max_length_)) {
    done_ = true;
    primed_ = true;
    return;
  }

  OrtValue* hidden = main_->state_->GetOutput(main_model_.config_->model.mtp.main_hidden_states.c_str());
  const int last = static_cast<int>(tail) - 1;
  ExtractHiddenPosition(hidden, last);  // h for the token we are about to predict
  if (sampling_) {
    // Sample the first generated token from the truncated target distribution at the last prompt
    // position. Use the on-device top-k over just that row (no full-vocab cast/copy of prefill logits).
    OrtValue* raw = main_->state_->GetOutput(main_model_.config_->model.decoder.outputs.logits.c_str());
    auto info = raw->GetTensorTypeAndShapeInfo();
    const int rtype = static_cast<int>(info->GetElementType());
    const size_t elem = Ort::SizeOf(info->GetElementType());
    const uint8_t* base = static_cast<const uint8_t*>(raw->GetTensorRawData());
    const void* row_ptr = base + static_cast<size_t>(last) * vocab_size_ * elem;
    std::vector<int32_t> idx;
    std::vector<float> prob;
    if (!main_logits_penalties_->IsActive() &&
        TopKScoresRows(row_ptr, rtype, 1, *main_model_.p_device_)) {
      SparseFromTopKRow(0, idx, prob);
    } else {
      const float* rowf = MainLogitsRowsCpu(last, 1);
      auto processed = ProcessMainLogitsRow(
          std::span<const float>(rowf, static_cast<size_t>(vocab_size_)), last);
      ComputeSampledCategorical(processed,
                                top_k_, top_p_, temperature_, sampled_scratch_);
      idx = sampled_scratch_.indices;
      prob = sampled_scratch_.probs;
    }
    next_token_ = SampleSparseToken(idx, prob, rng_);
  } else {
    ArgmaxMainRows(last, 1, &next_token_);  // token predicted for position length_
  }
  has_pending_draft_ = false;
  pending_refeed_count_ = -1;
  primed_ = true;
}

void MtpGenerator::RunRound() {
  if (done_) return;

  // Commit the token predicted for position length_.
  const int32_t t = next_token_;
  sequence_.push_back(t);
  if (IsEos(t) || sequence_.size() >= static_cast<size_t>(max_length_)) {
    done_ = true;
    return;
  }

  if (sampling_)
    GenerateStepMultiSample(t);
  else if (num_speculative_tokens_ == 1)
    GenerateStepSingle(t);
  else
    GenerateStepMulti(t);
}

void MtpGenerator::GenerateStepSingle(int32_t t) {
  stats_.rounds++;
  stats_.completed_rounds++;
  stats_.draft_tokens_proposed++;

  // 1. Draft the next token for t. After an accepted step the draft was already computed ahead
  //    (fused into that step's KV-advance as one 2-token MTP forward), so reuse it; otherwise the
  //    MTP head is at the right point and we issue a fresh single-token draft.
  int32_t d;
  if (has_pending_draft_) {
    d = pending_draft_;
    has_pending_draft_ = false;
  } else {
    d = DraftNextToken(nullptr, t);  // hidden_slice_ holds h paired with t
  }

  // 2. Snapshot the recurrent state at length L, then verify [t, d] in a single main forward.
  main_->SnapshotState();
  std::array<int32_t, 2> verify{t, d};
  main_->AppendTokens(cpu_span<const int32_t>(verify));
  stats_.target_forward_passes++;

  // Argmax both verify rows on-device in one launch: row 0 = main's real token after t,
  // row 1 = the free prediction harvested when the draft is accepted.
  int32_t verify_argmax[2];
  ArgmaxMainRows(0, 2, verify_argmax);
  const int32_t m = verify_argmax[0];
  stats_.draft_tokens_evaluated++;

  if (d == m) {
    // 2a. Accept: t and d are both correct. Commit d and harvest the free prediction at row 1.
    stats_.draft_tokens_accepted++;
    stats_.bonus_tokens++;
    sequence_.push_back(d);
    if (contains(main_model_.config_->model.eos_token_id, d) ||
        sequence_.size() >= static_cast<size_t>(max_length_)) {
      done_ = true;
      return;
    }
    OrtValue* hidden = main_->state_->GetOutput(main_model_.config_->model.mtp.main_hidden_states.c_str());
    // Next token to commit is argmax(logits@L+1) (harvested above).
    next_token_ = verify_argmax[1];
    // Fuse the post-accept KV-advance (hidden@L, d) and the next step's draft (hidden@L+1,
    // next_token_) into ONE 2-token MTP forward, and stash the resulting draft for the next step.
    pending_draft_ = DraftTwo(hidden, d, next_token_);
    has_pending_draft_ = true;
    length_ += 2;
  } else {
    // 2b. Reject: roll back the speculative forward (restore recurrent state + crop KV to L),
    //     then re-run the single correct token t. The pipelined draft (if any) is invalid.
    has_pending_draft_ = false;
    main_->RewindToLength(length_);
    std::array<int32_t, 1> rerun{t};
    main_->AppendTokens(cpu_span<const int32_t>(rerun));
    stats_.target_forward_passes++;
    stats_.correction_tokens++;
    OrtValue* hidden = main_->state_->GetOutput(main_model_.config_->model.mtp.main_hidden_states.c_str());
    ArgmaxMainRows(0, 1, &next_token_);
    ExtractHiddenPosition(hidden, 0);
    length_ += 1;
  }
}

void MtpGenerator::GenerateStepMulti(int32_t t) {
  const int N = std::min(num_speculative_tokens_, static_cast<int>(max_length_ - length_ - 1));
  stats_.rounds++;
  stats_.completed_rounds++;
  stats_.draft_tokens_proposed += static_cast<size_t>(N);
  const std::string& hs_name = main_model_.config_->model.mtp.main_hidden_states;

  // --- Draft phase: chain the single MTP module N times. ---
  // Step 0 feeds the main model's hidden (hidden_slice_ holds h paired with t) and appends the
  // committed token t to the head KV. Steps 1..N-1 feed the head's OWN post-norm hidden
  // (head_out_hidden_, captured by DraftHeadStep) + the previous draft -- speculative appends.
  //
  // When the previous step left an accepted prefix pending, its refeed is FUSED into step 0: the
  // accepted drafts and t are consecutive positions that all pair with MAIN-model hidden states,
  // so ONE (a+1)-token head forward both re-materializes the drafts in the head KV and produces
  // this step's first draft from its last row -- exactly what a separate refeed forward plus an
  // M=1 draft forward produced, at one forward instead of two.
  size_t head_start = 0;
  if (device_draft_chain_) {
    if (pending_refeed_count_ > 0) {
      const int a_prev = pending_refeed_count_;
      mtp_->RewindToLength(pending_refeed_head_len_);  // drop last step's speculative drafts
      head_len_ = pending_refeed_head_len_;
      merged_tokens_[a_prev] = t;  // last row: the token committed at the top of this step
      mtp_->SetHiddenStates(refeed_multi_[a_prev + 1]);
      DraftHeadStepMultiToDevice(merged_tokens_.data(), a_prev + 1, drafts_device_.subspan(0, 1));
    } else {
      if (pending_refeed_count_ == 0) {
        // Nothing was accepted last step: only the speculative drafts need dropping.
        mtp_->RewindToLength(pending_refeed_head_len_);
        head_len_ = pending_refeed_head_len_;
      }
      mtp_->SetHiddenStates(hidden_slice_);
      DraftHeadStepToDevice(t, drafts_device_.subspan(0, 1));
    }
    pending_refeed_count_ = -1;
    head_start = head_len_ - 1;  // head KV length before t was appended
    for (int k = 1; k < N; ++k) {
      mtp_->SetHiddenStates(head_out_hidden_);
      DraftHeadStepToDevice(drafts_device_.subspan(static_cast<size_t>(k - 1), 1),
                            drafts_device_.subspan(static_cast<size_t>(k), 1));
    }
    auto drafts_cpu = drafts_device_.CopyDeviceToCpu();
    std::copy_n(drafts_cpu.begin(), N, drafts_.begin());
  } else {
    if (pending_refeed_count_ > 0) {
      const int a_prev = pending_refeed_count_;
      mtp_->RewindToLength(pending_refeed_head_len_);
      head_len_ = pending_refeed_head_len_;
      merged_tokens_[a_prev] = t;
      mtp_->SetHiddenStates(refeed_multi_[a_prev + 1]);
      drafts_[0] = DraftHeadStepMulti(merged_tokens_.data(), a_prev + 1);
    } else {
      if (pending_refeed_count_ == 0) {
        mtp_->RewindToLength(pending_refeed_head_len_);
        head_len_ = pending_refeed_head_len_;
      }
      mtp_->SetHiddenStates(hidden_slice_);
      drafts_[0] = DraftHeadStep(t);
    }
    pending_refeed_count_ = -1;
    head_start = head_len_ - 1;
    for (int k = 1; k < N; ++k) {
      mtp_->SetHiddenStates(head_out_hidden_);
      drafts_[k] = DraftHeadStep(drafts_[k - 1]);
    }
  }

  // --- Verify [t, d0..d_{N-1}] in a single batched main forward (the whole point of MTP: one
  //     forward validates N+1 tokens). The batched (M=N+1) forward is numerically ~equal but not
  //     bit-identical to single-token decode (different GEMM tiling for M=1 vs M>1, plus XQA-vs-
  //     Flash attention), so a greedy argmax can occasionally differ on near-ties. This is the same
  //     tradeoff the N=1 verify already makes; the reject path below re-runs decode-consistently to
  //     bound divergence from plain greedy. ---
  // Snapshot the recurrent state so a rejected wide verify can replay the committed prefix with
  // decode-consistent numerics. The snapshot is also needed when no draft is accepted, where there
  // is no committed recurrent-state window slot to crop to.
  main_->SnapshotState();
  verify_tokens_[0] = t;
  for (int k = 0; k < N; ++k) verify_tokens_[k + 1] = drafts_[k];
  main_->AppendTokens(cpu_span<const int32_t>(verify_tokens_.data(), N + 1));
  stats_.target_forward_passes++;
  ArgmaxMainRows(0, N + 1, verify_argmax_.data());  // main's real token after each verify position
  OrtValue* vhidden = main_->state_->GetOutput(hs_name.c_str());

  // --- Longest accepted prefix (greedy match against the main model). ---
  int a = 0;
  while (a < N && drafts_[a] == verify_argmax_[a]) ++a;
  stats_.draft_tokens_evaluated +=
      (a < N) ? static_cast<size_t>(a + 1) : static_cast<size_t>(N);
  stats_.draft_tokens_accepted += static_cast<size_t>(a);
  if (a == N)
    stats_.bonus_tokens++;
  else
    stats_.correction_tokens++;

  // Commit the a accepted drafts (t was already committed by the caller). Stop at eos/max_length.
  for (int k = 0; k < a; ++k) {
    sequence_.push_back(drafts_[k]);
    if (contains(main_model_.config_->model.eos_token_id, drafts_[k]) ||
        sequence_.size() >= static_cast<size_t>(max_length_)) {
      done_ = true;
      return;  // generation finished; leftover main/head KV is irrelevant
    }
  }

  // --- Capture the head refeed payload, but DEFER the forward. Re-materializing the a accepted
  //     drafts in the head KV and drafting this step's successor both consume MAIN-model hidden
  //     states over consecutive positions, so they fuse into a single (a+1)-token head forward
  //     issued at the top of the next step (see the draft phase). Only the payload is captured
  //     here, because the accepted drafts' hidden rows must be read out of the verify output
  //     BEFORE a finalize replay overwrites it. The head KV rewind is deferred too -- nothing
  //     reads the head state between steps. ---
  pending_refeed_head_len_ = head_start + 1;  // keep t (fed with its main hidden), drop the drafts
  pending_refeed_count_ = a;
  if (a > 0) {
    // Rows 0..a-1 are contiguous in both the verify output and the merged buffer: one D2D copy.
    // Row a is filled in by the finalize phase below from hidden_slice_.
    Tensor& hbuf = *refeed_multi_[a + 1];
    const size_t row_bytes = refeed_hidden_->GetByteSpan().size();
    auto src = ByteWrapTensor(*main_model_.p_device_, *vhidden);
    hbuf.GetByteSpan()
        .subspan(0, static_cast<size_t>(a) * row_bytes)
        .CopyFrom(src.subspan(0, static_cast<size_t>(a) * row_bytes));
    for (int k = 0; k < a; ++k) merged_tokens_[k] = drafts_[k];
  }

  if (a == N) {
    // All drafts accepted: the batched verify already committed [t, d0..d_{N-1}] correctly, so the
    // main KV / recurrent state is exactly at L + (N+1). The bonus token is main's prediction at the
    // last verify row (mirrors the N=1 accept path, which likewise commits a batched-forward token).
    next_token_ = verify_argmax_[a];
    CopyHiddenRow(vhidden, a, *hidden_slice_);
    length_ += static_cast<size_t>(N) + 1;
  } else if (main_->CanCropRecurrentState() && a >= 1) {
    // Partial accept (a>=1), LOSSLESS CROP fast-path (model exported with state_window).
    // The batched verify's row a is an EARLY row of a wide (M=N+1) forward, whose argmax is NOT
    // decode-consistent (only the LAST row of a forward matches a 1-token decode; §13.1). So we
    // cannot take the bonus straight from the verify. Instead: crop the KV cache + recurrent state
    // to L+a (state AFTER verify tokens 0..a-1 == window slot for position a-1) -- avoiding the wide
    // (a+1)-token replay -- then M=1-decode the last committed token (d_{a-1}, at position L+a). Its
    // row-0 logits ARE decode-consistent, giving a lossless bonus. This replaces the §13.5 wide
    // replay (M=a+1) with a cheap M=1 forward.
    // NOTE (§14.3): this is NOT actually lossless in practice -- the cropped state is itself derived
    // from the wide batched verify and differs from sequential decode (~0.25 fp16), so greedy
    // near-ties still flip. Kept for history/reference; the fallback replay below is the lossless path.
    main_->CropToAccepted(length_ + static_cast<size_t>(a), static_cast<size_t>(a) - 1);
    std::array<int32_t, 1> last{drafts_[a - 1]};  // committed token at position L+a
    main_->AppendTokens(cpu_span<const int32_t>(last));
    stats_.target_forward_passes++;
    OrtValue* rhidden = main_->state_->GetOutput(hs_name.c_str());
    ArgmaxMainRows(0, 1, &next_token_);         // decode-consistent bonus (M=1 last row)
    CopyHiddenRow(rhidden, 0, *hidden_slice_);  // hidden paired with the bonus token
    length_ += static_cast<size_t>(a) + 1;
  } else {
    // Rejection at position a: the batched verify over-appended N-a wrong tokens and cannot be
    // partially cropped (the linear-attention recurrent state has no per-token rollback). Restore
    // the recurrent snapshot at L and re-run only the committed prefix [t, d0..d_{a-1}]. Reading the
    // bonus + its hidden from THIS forward keeps the carried state consistent with the committed
    // sequence (the a==0 case degenerates to the exact N=1 single-token decode re-run).
    main_->RewindToLength(length_);
    verify_tokens_[0] = t;
    for (int k = 0; k < a; ++k) verify_tokens_[k + 1] = drafts_[k];
    main_->AppendTokens(cpu_span<const int32_t>(verify_tokens_.data(), a + 1));
    stats_.target_forward_passes++;
    OrtValue* rhidden = main_->state_->GetOutput(hs_name.c_str());
    ArgmaxMainRows(a, 1, &next_token_);         // main's token after the committed prefix
    CopyHiddenRow(rhidden, a, *hidden_slice_);  // hidden paired with the bonus token
    length_ += static_cast<size_t>(a) + 1;
  }

  // Row a of the fused refeed buffer pairs with next_token_ (committed at the top of the next
  // step). Its hidden is whatever finalize produced in hidden_slice_ -- verify row a on the
  // all-accept path or the replay forward's row on the partial-accept paths.
  if (pending_refeed_count_ > 0) {
    Tensor& hbuf = *refeed_multi_[pending_refeed_count_ + 1];
    const size_t row_bytes = refeed_hidden_->GetByteSpan().size();
    hbuf.GetByteSpan()
        .subspan(static_cast<size_t>(pending_refeed_count_) * row_bytes, row_bytes)
        .CopyFrom(hidden_slice_->GetByteSpan());
  }
}

void MtpGenerator::GenerateStepMultiSample(int32_t t) {
  const int N = std::min(num_speculative_tokens_, static_cast<int>(max_length_ - length_ - 1));
  stats_.rounds++;
  stats_.completed_rounds++;
  stats_.draft_tokens_proposed += static_cast<size_t>(N);
  const std::string& hs_name = main_model_.config_->model.mtp.main_hidden_states;

  // --- Draft phase: chain the single MTP module N times, SAMPLING each draft d_k from its
  //     truncated distribution q_k (top_k/top_p/temperature) and recording q_k for the accept test.
  const size_t head_start = head_len_;
  mtp_->SetHiddenStates(hidden_slice_);
  drafts_[0] = DraftHeadStepSample(t, 0);
  for (int k = 1; k < N; ++k) {
    mtp_->SetHiddenStates(head_out_hidden_);
    drafts_[k] = DraftHeadStepSample(drafts_[k - 1], k);
  }

  // --- Verify [t, d0..d_{N-1}] in a single batched main forward. ---
  // Snapshot the recurrent state ONLY for the fallback re-run rollback (models WITHOUT
  // state_window). Snapshot() copies every linear-attn layer's conv+recurrent
  // state (2*num_layers D2D copies + launches) on EVERY step. When the crop fast-path is
  // available the reject branch uses the state window via CropToPosition and NEVER rewinds
  // the recurrent state (RewindTo/RestoreSnapshot is unreachable), so the snapshot is dead
  // overhead -- skip it. The predicate matches the reject-path crop-vs-fallback choice below,
  // so the fallback branch still has its snapshot when it needs one.
  if (!main_->CanCropRecurrentState()) main_->SnapshotState();
  verify_tokens_[0] = t;
  for (int k = 0; k < N; ++k) verify_tokens_[k + 1] = drafts_[k];
  main_->AppendTokens(cpu_span<const int32_t>(verify_tokens_.data(), N + 1));
  stats_.target_forward_passes++;
  OrtValue* vhidden = main_->state_->GetOutput(hs_name.c_str());

  // Build each verify row's truncated target distribution p_k. Row k is the target's prediction for
  // the position after [t, d0..d_{k-1}] -- it judges draft d_k (row N is the bonus position).
  // Prefer the on-device top-k (only k*(N+1) values leave the GPU); fall back to a host cast+copy.
  OrtValue* raw_logits = main_->state_->GetOutput(main_model_.config_->model.decoder.outputs.logits.c_str());
  const int rtype = static_cast<int>(raw_logits->GetTensorTypeAndShapeInfo()->GetElementType());
  if (!main_logits_penalties_->IsActive() &&
      TopKScoresRows(raw_logits->GetTensorRawData(), rtype, N + 1, *main_model_.p_device_)) {
    for (int kk = 0; kk <= N; ++kk) SparseFromTopKRow(kk, target_idx_[kk], target_prob_[kk]);
  } else {
    const float* logits = MainLogitsRowsCpu(0, N + 1);
    for (int kk = 0; kk <= N; ++kk) {
      auto processed = ProcessMainLogitsRow(
          std::span<const float>(logits + static_cast<size_t>(kk) * vocab_size_,
                                 static_cast<size_t>(vocab_size_)),
          kk);
      ComputeSampledCategorical(
          processed, top_k_, top_p_, temperature_, sampled_scratch_);
      target_idx_[kk] = sampled_scratch_.indices;
      target_prob_[kk] = sampled_scratch_.probs;
    }
  }

  // --- Speculative-sampling accept/reject over the N drafts. Accept d_a with probability
  //     min(1, p_a(d_a)/q_a(d_a)); on the first rejection draw a correction from the residual
  //     norm(max(0, p_a - q_a)). `a` is the number of accepted drafts. ---
  std::uniform_real_distribution<float> uni(0.0f, 1.0f);
  int a = 0;
  int32_t correction = -1;
  bool rejected = false;
  for (; a < N; ++a) {
    const int32_t d = drafts_[a];
    const float p_t = GetSparseTokenProbability(target_idx_[a], target_prob_[a], d);
    const float p_d = GetSparseTokenProbability(draft_idx_[a], draft_prob_[a], d);
    if (uni(rng_) < ComputeAcceptProb(p_t, p_d))
      continue;  // accept d_a
    // Reject at position a: sample the correction from the normalized residual max(0, p_a - q_a).
    // The residual lives only on the target's sparse support, so draw it over ~top_k entries
    // instead of densifying both distributions to full vocab and building a full-vocab
    // std::discrete_distribution on this (frequent) reject path.
    correction = SampleCorrectionToken(target_idx_[a], target_prob_[a],
                                       draft_idx_[a], draft_prob_[a], rng_);
    rejected = true;
    break;
  }

  stats_.draft_tokens_evaluated +=
      rejected ? static_cast<size_t>(a + 1) : static_cast<size_t>(N);
  stats_.draft_tokens_accepted += static_cast<size_t>(a);
  if (rejected)
    stats_.correction_tokens++;
  else
    stats_.bonus_tokens++;

  // Commit the a accepted drafts (t was already committed by the caller). Stop at eos/max_length.
  for (int k = 0; k < a; ++k) {
    sequence_.push_back(drafts_[k]);
    if (contains(main_model_.config_->model.eos_token_id, drafts_[k]) ||
        sequence_.size() >= static_cast<size_t>(max_length_)) {
      done_ = true;
      return;
    }
  }

  // --- Roll the MTP head KV back to the committed tokens: keep t, drop the N-1 speculative drafts,
  //     then re-materialize the a accepted drafts with the main model's hidden states in ONE batched
  //     head forward (instead of a separate forward per token). Read the head-refeed hiddens from
  //     the verify output BEFORE any main rewind overwrites the buffer. Under sampling the head KV
  //     only shapes the DRAFT distribution q; rejection sampling still corrects the output to p, so
  //     the batched (M=a) refeed does not change the output distribution. ---
  mtp_->RewindToLength(head_start + 1);
  head_len_ = head_start + 1;
  if (a > 0) {
    Tensor& hbuf = *refeed_multi_[a];
    const size_t row_bytes = refeed_hidden_->GetByteSpan().size();
    auto dst = hbuf.GetByteSpan();
    auto src = ByteWrapTensor(*main_model_.p_device_, *vhidden);
    for (int k = 0; k < a; ++k)
      dst.subspan(static_cast<size_t>(k) * row_bytes, row_bytes)
          .CopyFrom(src.subspan(static_cast<size_t>(k) * row_bytes, row_bytes));
    mtp_->SetHiddenStates(refeed_multi_[a]);
    mtp_->AppendTokens(cpu_span<const int32_t>(drafts_.data(), a));  // d0..d_{a-1} with main hiddens
    stats_.draft_forward_passes++;
    head_len_ = head_start + 1 + static_cast<size_t>(a);
  }

  if (!rejected) {
    // Every draft accepted: the batched verify already committed [t, d0..d_{N-1}] correctly, so the
    // main KV / recurrent state is exactly at L + (N+1). Draw the bonus token from the target's
    // next-position distribution p_N (verify row N), sampled directly from its sparse support.
    next_token_ = SampleSparseToken(target_idx_[N], target_prob_[N], rng_);
    CopyHiddenRow(vhidden, N, *hidden_slice_);  // hidden that predicted the bonus token
    length_ += static_cast<size_t>(N) + 1;
  } else if (main_->CanCropRecurrentState()) {
    // Rejected at position a, LOSSLESS-CROP fast path (model exported with state_window):
    // skip the full re-run main forward (~25% of N=3 step time). The batched verify already advanced
    // the recurrent state through every token (window slot for position a = state AFTER the committed
    // prefix [t, d0..d_{a-1}]) and computed row a's hidden -- which is exactly the hidden that
    // predicts the correction (a causal hidden is independent of the rejected drafts that follow it,
    // so verify row a == the re-run's row a). So crop the KV + recurrent state to L+a+1 and pair the
    // already-sampled correction with verify row a. Unlike the GREEDY path (which needs a decode-
    // consistent argmax bonus and therefore re-runs losslessly), rejection sampling corrects the
    // OUTPUT distribution regardless of the wide-verify state's small (~0.25 fp16) drift from
    // sequential decode -- and the accept-all sampling branch above already carries the wide-verify
    // recurrent numerics forward -- so the crop is distribution-safe here and avoids the replay.
    main_->CropToAccepted(length_ + static_cast<size_t>(a) + 1, static_cast<size_t>(a));
    next_token_ = correction;
    CopyHiddenRow(vhidden, a, *hidden_slice_);  // verify row a: hidden that predicts the correction
    length_ += static_cast<size_t>(a) + 1;
  } else {
    // Fallback (model without state_window): the recurrent state cannot be cropped, so
    // restore the snapshot at L and re-run only the committed prefix [t, d0..d_{a-1}] so the carried
    // recurrent/KV state is decode-consistent; pair the sampled correction (already drawn from the
    // batched verify's residual) with the re-run's hidden at row a.
    main_->RewindToLength(length_);
    verify_tokens_[0] = t;
    for (int k = 0; k < a; ++k) verify_tokens_[k + 1] = drafts_[k];
    main_->AppendTokens(cpu_span<const int32_t>(verify_tokens_.data(), a + 1));
    stats_.target_forward_passes++;
    OrtValue* rhidden = main_->state_->GetOutput(hs_name.c_str());
    next_token_ = correction;
    CopyHiddenRow(rhidden, a, *hidden_slice_);  // hidden paired with the correction token
    length_ += static_cast<size_t>(a) + 1;
  }
}

}  // namespace Generators
