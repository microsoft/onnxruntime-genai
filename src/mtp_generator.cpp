// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "search.h"
#include "constrained_logits_processor.h"
#include "models/model.h"
#include "mtp_generator.h"

namespace Generators {

namespace {
// Greedy argmax over a contiguous vocab row of fp32 logits on the CPU.
int32_t ArgmaxRow(const float* row, int vocab_size) {
  int32_t best = 0;
  float best_val = row[0];
  for (int i = 1; i < vocab_size; ++i) {
    if (row[i] > best_val) {
      best_val = row[i];
      best = i;
    }
  }
  return best;
}
}  // namespace

MtpGenerator::MtpGenerator(const Model& main_model, const Model& mtp_model, const GeneratorParams& params)
    : main_model_{main_model}, mtp_model_{mtp_model} {
  // MTP runs both a 1-token decode and a 2-token verify on the main model. Allow CUDA graph
  // capture of both shapes (each captured under its own annotation id with pre-sized static
  // buffers). Harmless for the MTP head, which only ever runs a single token per step.
  const_cast<GeneratorParams&>(params).max_graph_capture_length = 2;

  main_ = CreateGenerator(main_model_, params);
  mtp_ = CreateGenerator(mtp_model_, params);

  hidden_size_ = main_model_.config_->model.decoder.hidden_size;
  vocab_size_ = main_model_.config_->model.vocab_size;
  max_length_ = params.search.max_length;

  // Reusable [1, 1, hidden] device buffer for the on-device hidden-state handoff.
  hidden_slice_ = std::make_shared<Tensor>(
      main_model_.p_device_inputs_,
      main_model_.session_info_.GetOutputDataType(main_model_.config_->model.decoder.outputs.hidden_states));
  const std::array<int64_t, 3> slice_shape{1, 1, hidden_size_};
  hidden_slice_->CreateTensor(slice_shape);

  // Reusable [1, 2, hidden] device buffer for the batched 2-token draft (post-accept KV-advance
  // fused with the next step's draft into one MTP forward).
  hidden_slice2_ = std::make_shared<Tensor>(
      main_model_.p_device_inputs_,
      main_model_.session_info_.GetOutputDataType(main_model_.config_->model.decoder.outputs.hidden_states));
  const std::array<int64_t, 3> slice2_shape{1, 2, hidden_size_};
  hidden_slice2_->CreateTensor(slice2_shape);
}

void MtpGenerator::ExtractHiddenPosition(OrtValue* hidden, int position) {
  // hidden is [1, S, H] on the model device; copy row `position` into hidden_slice_ ([1,1,H]).
  auto src = ByteWrapTensor(*main_model_.p_device_, *hidden);
  const size_t row_bytes = hidden_slice_->GetByteSpan().size();
  auto src_row = src.subspan(static_cast<size_t>(position) * row_bytes, row_bytes);
  hidden_slice_->GetByteSpan().CopyFrom(src_row);
}

void MtpGenerator::ArgmaxMainRows(int first_row, int num_rows, int32_t* out) {
  OrtValue* raw = main_->state_->GetOutput(main_model_.config_->model.decoder.outputs.logits.c_str());
  auto info = raw->GetTensorTypeAndShapeInfo();
  const ONNXTensorElementDataType type = info->GetElementType();

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
  if (!need_draft) {
    // KV-advance only (e.g. after an accepted draft): skip the full-vocab argmax + stream sync.
    return 0;
  }
  auto logits_span = mtp_->GetLogits();              // fp32, last token, [1, V]
  int32_t draft = 0;
  if (mtp_model_.p_device_->ArgMax(logits_span.Span().data(), Ort::TypeToTensorType<float>, 1, vocab_size_, &draft))
    return draft;
  auto logits = logits_span.CopyDeviceToCpu();        // host fallback
  return ArgmaxRow(logits.data(), vocab_size_);
}

int32_t MtpGenerator::DraftTwo(OrtValue* hidden, int32_t tok0, int32_t tok1) {
  // Populate the [1,2,H] hidden buffer: row 0 = hidden@position L (pairs with tok0), row 1 =
  // hidden@position L+1 (pairs with tok1). `hidden` is the main model's [1,S,H] verify output.
  auto src = ByteWrapTensor(*main_model_.p_device_, *hidden);
  const size_t row_bytes = hidden_slice_->GetByteSpan().size();  // bytes of one [1,1,H] row
  auto dst = hidden_slice2_->GetByteSpan();
  dst.subspan(0, row_bytes).CopyFrom(src.subspan(0, row_bytes));            // row 0 <- hidden@0
  dst.subspan(row_bytes, row_bytes).CopyFrom(src.subspan(row_bytes, row_bytes));  // row 1 <- hidden@1

  // One 2-token MTP forward: feeds tok0 (KV-advance) and tok1 (the next committed token); the
  // last-position logits give the draft for the token after tok1.
  mtp_->SetHiddenStates(hidden_slice2_);
  std::array<int32_t, 2> toks{tok0, tok1};
  mtp_->AppendTokens(cpu_span<const int32_t>(toks));
  auto logits_span = mtp_->GetLogits();              // fp32, last token, [1, V]
  int32_t draft = 0;
  if (mtp_model_.p_device_->ArgMax(logits_span.Span().data(), Ort::TypeToTensorType<float>, 1, vocab_size_, &draft))
    return draft;
  auto logits = logits_span.CopyDeviceToCpu();        // host fallback
  return ArgmaxRow(logits.data(), vocab_size_);
}

void MtpGenerator::AppendTokens(cpu_span<const int32_t> input_ids) {
  main_->AppendTokens(input_ids);
  length_ = input_ids.size();
  for (auto t : input_ids) sequence_.push_back(t);

  OrtValue* hidden = main_->state_->GetOutput(main_model_.config_->model.decoder.outputs.hidden_states.c_str());
  const int last = static_cast<int>(input_ids.size()) - 1;
  ExtractHiddenPosition(hidden, last);             // h for the token we are about to predict
  ArgmaxMainRows(last, 1, &next_token_);           // token predicted for position length_
  has_pending_draft_ = false;
  primed_ = true;
}

void MtpGenerator::GenerateNextToken() {
  if (!primed_) throw std::runtime_error("MtpGenerator: AppendTokens must be called before GenerateNextToken");
  if (done_) return;

  // Commit the token predicted for position length_.
  const int32_t t = next_token_;
  sequence_.push_back(t);
  if (contains(main_model_.config_->model.eos_token_id, t) || sequence_.size() >= static_cast<size_t>(max_length_)) {
    done_ = true;
    return;
  }

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
  ++forwards_;

  // Argmax both verify rows on-device in one launch: row 0 = main's real token after t,
  // row 1 = the free prediction harvested when the draft is accepted.
  int32_t verify_argmax[2];
  ArgmaxMainRows(0, 2, verify_argmax);
  const int32_t m = verify_argmax[0];
  ++trials_;

  if (d == m) {
    // 2a. Accept: t and d are both correct. Commit d and harvest the free prediction at row 1.
    ++accepts_;
    sequence_.push_back(d);
    if (sequence_.size() >= static_cast<size_t>(max_length_)) {
      done_ = true;
      return;
    }
    OrtValue* hidden = main_->state_->GetOutput(main_model_.config_->model.decoder.outputs.hidden_states.c_str());
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
    ++forwards_;
    OrtValue* hidden = main_->state_->GetOutput(main_model_.config_->model.decoder.outputs.hidden_states.c_str());
    ArgmaxMainRows(0, 1, &next_token_);
    ExtractHiddenPosition(hidden, 0);
    length_ += 1;
  }
}

bool MtpGenerator::IsDone() const {
  return done_;
}

}  // namespace Generators
