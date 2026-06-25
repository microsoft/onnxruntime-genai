// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "../generators.h"
#include "moonshine_streaming.h"

namespace Generators {

void MoonshineConfig::PopulateFromConfig(const Config& config) {
  const auto& m = config.model;
  const auto& enc = m.encoder;
  const auto& dec = m.decoder;

  if (m.sample_rate > 0) sample_rate = m.sample_rate;
  if (m.chunk_samples > 0) chunk_samples = m.chunk_samples;
  if (m.bos_token_id > 0) bos_token_id = m.bos_token_id;
  if (!m.eos_token_id.empty()) eos_token_id = m.eos_token_id[0];

  if (enc.hidden_size > 0) encoder_dim = enc.hidden_size;
  if (dec.hidden_size > 0) decoder_dim = dec.hidden_size;
  if (dec.num_hidden_layers > 0) num_decoder_layers = dec.num_hidden_layers;
  if (dec.num_attention_heads > 0) num_decoder_heads = dec.num_attention_heads;
  if (dec.head_size > 0) decoder_head_size = dec.head_size;

  if (!enc.filename.empty()) frontend_filename = enc.filename;
  if (!dec.filename.empty()) decoder_kv_filename = dec.filename;
}

MoonshineStreamingModel::MoonshineStreamingModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  moonshine_config_.PopulateFromConfig(*config_);

  session_options_ = OrtSessionOptions::Create();
  CreateSessionOptionsFromConfig(config_->model.decoder.session_options,
                                 *session_options_, true);

  session_frontend_   = CreateSession(ort_env, moonshine_config_.frontend_filename,   session_options_.get());
  session_encoder_    = CreateSession(ort_env, moonshine_config_.encoder_filename,    session_options_.get());
  session_adapter_    = CreateSession(ort_env, moonshine_config_.adapter_filename,    session_options_.get());
  session_cross_kv_   = CreateSession(ort_env, moonshine_config_.cross_kv_filename,   session_options_.get());
  session_decoder_kv_ = CreateSession(ort_env, moonshine_config_.decoder_kv_filename, session_options_.get());

  session_info_.Add(*session_frontend_);
  session_info_.Add(*session_encoder_);
  session_info_.Add(*session_adapter_);
  session_info_.Add(*session_cross_kv_);
  session_info_.Add(*session_decoder_kv_);
}

std::unique_ptr<State> MoonshineStreamingModel::CreateState(DeviceSpan<int32_t> /*sequence_lengths*/,
                                                            const GeneratorParams& params) const {
  return std::make_unique<MoonshineStreamingState>(*this, params);
}

MoonshineStreamingState::MoonshineStreamingState(const MoonshineStreamingModel& model,
                                                 const GeneratorParams& params)
    : TransducerState{params, model},
      moonshine_model_{model} {
  config_ = model.moonshine_config_;

  // Idle until SetExtraInputs() supplies cross-KV for a chunk.
  chunk_done_ = true;

  // Pre-allocate single-token int64 tensor [1, 1].
  auto tok_shape = std::array<int64_t, 2>{1, 1};
  token_tensor_ = OrtValue::CreateTensor(model.allocator_cpu_, tok_shape,
                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

  ResetSelfKv();
}

MoonshineStreamingState::~MoonshineStreamingState() = default;

void MoonshineStreamingState::ResetSelfKv() {
  // [num_decoder_layers, 1, num_decoder_heads, 0, head_size] float32.
  auto shape = std::array<int64_t, 5>{
      config_.num_decoder_layers, 1, config_.num_decoder_heads, 0,
      config_.decoder_head_size};
  k_self_ = OrtValue::CreateTensor(moonshine_model_.allocator_cpu_, shape,
                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  v_self_ = OrtValue::CreateTensor(moonshine_model_.allocator_cpu_, shape,
                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
}

DeviceSpan<float> MoonshineStreamingState::Run(int /*total_length*/,
                                               DeviceSpan<int32_t>& /*next_tokens*/,
                                               DeviceSpan<int32_t> /*next_indices*/) {
  // Streaming ASR bypasses the standard search/logits loop; StepToken() drives execution.
  return {};
}

void MoonshineStreamingState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  std::shared_ptr<Tensor> k_cross;
  std::shared_ptr<Tensor> v_cross;
  bool is_final = false;
  for (const auto& input : extra_inputs) {
    if (input.name == "k_cross" || input.name == "out_k_cross") {
      k_cross = input.tensor;
    } else if (input.name == "v_cross" || input.name == "out_v_cross") {
      v_cross = input.tensor;
    } else if (input.name == "is_final") {
      if (input.tensor && input.tensor->ort_tensor_) {
        is_final = (*input.tensor->ort_tensor_->GetTensorData<int64_t>()) != 0;
      }
    }
  }
  if (!k_cross || !v_cross) return;  // No new chunk this call.

  k_cross_tensor_ = std::move(k_cross);
  v_cross_tensor_ = std::move(v_cross);

  // Reset per-pass decoder state. (Self-KV gets rebuilt fresh each pass since
  // we re-decode from BOS over the full accumulated memory.)
  ResetSelfKv();
  pending_tokens_.clear();
  pending_idx_ = 0;
  last_tokens_.clear();

  // cross-KV shape is [num_decoder_layers, 1, num_decoder_heads, memory_len, head_size].
  int64_t memory_len = 0;
  if (k_cross_tensor_ && k_cross_tensor_->ort_tensor_) {
    auto shape = k_cross_tensor_->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape();
    if (shape.size() >= 5) memory_len = shape[3];
  }

  // Detect new utterance: if memory_len shrinks vs the previous call, the
  // processor was Flush()'d (or this is the first chunk after construction).
  // Reset commit tracking and the running transcript so the next stream
  // starts clean.
  if (memory_len < previous_memory_len_) {
    previous_pass_tokens_.clear();
    emitted_count_ = 0;
    all_tokens_.clear();
  }
  previous_memory_len_ = memory_len;

  if (memory_len <= 0) {
    chunk_done_ = true;
    return;
  }

  // Per-chunk token cap (matches the official moonshine streaming impl).
  const float duration_sec =
      static_cast<float>(memory_len) * config_.seconds_per_memory_frame;
  const int cap = static_cast<int>(std::ceil(duration_sec * config_.tokens_per_second));
  const int max_tokens = std::min(cap, config_.max_seq_len);

  // Full AR decode from BOS to EOS-or-cap.
  std::vector<int32_t> current_pass;
  current_pass.reserve(static_cast<size_t>(max_tokens));
  int64_t input_tok = config_.bos_token_id;
  for (int i = 0; i < max_tokens; ++i) {
    int next_tok = RunDecoderStep(input_tok);
    if (next_tok == config_.eos_token_id) break;
    current_pass.push_back(static_cast<int32_t>(next_tok));
    input_tok = static_cast<int64_t>(next_tok);
  }

  // Compute the new "committed" frontier:
  //   * on Flush (is_final): commit every token of the current pass.
  //   * otherwise: commit up to the longest-common-prefix with the previous
  //     pass — beyond that point, the model might still rewrite, so hold
  //     those tokens back until a future chunk confirms them.
  size_t commit_end;
  if (is_final) {
    commit_end = current_pass.size();
  } else {
    size_t lcp = 0;
    while (lcp < previous_pass_tokens_.size() && lcp < current_pass.size() &&
           previous_pass_tokens_[lcp] == current_pass[lcp]) {
      ++lcp;
    }
    commit_end = lcp;
  }

  // Emit only the slice (emitted_count_ .. commit_end] we haven't emitted
  // yet. Guard against monotonicity violations (commit_end shouldn't go
  // backwards once tokens are emitted, but if it does — e.g. model retracts
  // an earlier token — we silently skip rather than try to "un-emit").
  if (commit_end > emitted_count_ && commit_end <= current_pass.size()) {
    pending_tokens_.assign(
        current_pass.begin() + static_cast<std::ptrdiff_t>(emitted_count_),
        current_pass.begin() + static_cast<std::ptrdiff_t>(commit_end));
    emitted_count_ = commit_end;
  }
  previous_pass_tokens_ = std::move(current_pass);
  chunk_done_ = pending_tokens_.empty();
}

void MoonshineStreamingState::StepToken() {
  last_tokens_.clear();
  if (pending_idx_ >= pending_tokens_.size()) {
    chunk_done_ = true;
    return;
  }
  const int32_t tok = pending_tokens_[pending_idx_++];
  last_tokens_.push_back(tok);
  all_tokens_.push_back(tok);
  if (pending_idx_ >= pending_tokens_.size()) {
    chunk_done_ = true;
  }
}

int MoonshineStreamingState::RunDecoderStep(int64_t input_token) {
  *token_tensor_->GetTensorMutableData<int64_t>() = input_token;

  // decoder_kv inputs: token, k_self, v_self, out_k_cross, out_v_cross.
  const char* in_names[] = {"token", "k_self", "v_self", "out_k_cross", "out_v_cross"};
  OrtValue* in_values[]  = {
      token_tensor_.get(),
      k_self_.get(),
      v_self_.get(),
      k_cross_tensor_->ort_tensor_.get(),
      v_cross_tensor_->ort_tensor_.get(),
  };

  // decoder_kv outputs: logits, out_k_self, out_v_self, out_k_cross, out_v_cross.
  // The cross-KV outputs are passthroughs of the inputs; we discard them and
  // keep the chunk-scoped input tensors alive ourselves.
  const char* out_names[]  = {"logits", "out_k_self", "out_v_self", "out_k_cross", "out_v_cross"};
  OrtValue*   out_values[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};

  moonshine_model_.session_decoder_kv_->Run(
      nullptr,
      in_names, in_values, 5,
      out_names, out_values, 5);

  auto logits = std::unique_ptr<OrtValue>(out_values[0]);
  k_self_ = std::unique_ptr<OrtValue>(out_values[1]);
  v_self_ = std::unique_ptr<OrtValue>(out_values[2]);
  (void)std::unique_ptr<OrtValue>(out_values[3]);
  (void)std::unique_ptr<OrtValue>(out_values[4]);

  // Greedy argmax over the last-row logits.
  auto lshape = logits->GetTensorTypeAndShapeInfo()->GetShape();
  const int64_t seq = (lshape.size() >= 3) ? lshape[1] : 1;
  const int64_t vocab = lshape.back();
  const float* ldata = logits->GetTensorData<float>() + (seq - 1) * vocab;
  float max_val = ldata[0];
  int max_idx = 0;
  for (int i = 1; i < vocab; ++i) {
    if (ldata[i] > max_val) {
      max_val = ldata[i];
      max_idx = i;
    }
  }
  return max_idx;
}

OrtValue* MoonshineStreamingState::GetInput(const char* /*name*/) { return nullptr; }
OrtValue* MoonshineStreamingState::GetOutput(const char* /*name*/) { return nullptr; }

}  // namespace Generators
