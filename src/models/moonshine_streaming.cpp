// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../generators.h"
#include "moonshine_streaming.h"

namespace Generators {

void MoonshineConfig::PopulateFromConfig(const Config& config) {
  const auto& m = config.model;
  const auto& enc = m.encoder;
  const auto& dec = m.decoder;
  const auto& ms  = m.moonshine;

  if (m.sample_rate > 0) sample_rate = m.sample_rate;
  if (m.chunk_samples > 0) chunk_samples = m.chunk_samples;
  if (m.bos_token_id > 0) bos_token_id = m.bos_token_id;
  if (!m.eos_token_id.empty()) eos_token_id = m.eos_token_id[0];

  if (enc.hidden_size > 0) {
    encoder_dim = enc.hidden_size;
    // Frontend geometry derives from the encoder hidden size. Upstream
    // Moonshine sets d_model_frontend = encoder_dim, and the frontend's
    // expansion factor is 2 (c1 = 2 * d_model_frontend).
    conv1_channels = enc.hidden_size;
    conv2_channels = 2 * enc.hidden_size;
  }
  if (dec.hidden_size > 0) decoder_dim = dec.hidden_size;
  if (dec.num_hidden_layers > 0) {
    num_decoder_layers = dec.num_hidden_layers;
    // Encoder left-context = total_lookahead * depth (depth == decoder layers
    // in moonshine since enc/dec depths match).
    left_context_frames = total_lookahead * dec.num_hidden_layers;
  }
  if (dec.num_attention_heads > 0) num_decoder_heads = dec.num_attention_heads;
  if (dec.head_size > 0) decoder_head_size = dec.head_size;

  // Per-variant tuning from the moonshine section. Sentinel 0 (the
  // struct's default) means "not present in JSON, keep the C++ default".
  if (ms.sample_buffer_size > 0)        sample_buffer_size        = ms.sample_buffer_size;
  if (ms.conv1_buffer_size > 0)         conv1_buffer_size         = ms.conv1_buffer_size;
  if (ms.conv2_buffer_size > 0)         conv2_buffer_size         = ms.conv2_buffer_size;
  if (ms.total_lookahead > 0)           total_lookahead           = ms.total_lookahead;
  if (ms.left_context_frames > 0)       left_context_frames       = ms.left_context_frames;
  if (ms.max_seq_len > 0)               max_seq_len               = ms.max_seq_len;
  if (ms.tokens_per_second > 0.0f)      tokens_per_second         = ms.tokens_per_second;
  if (ms.seconds_per_memory_frame > 0.0f) seconds_per_memory_frame = ms.seconds_per_memory_frame;
  if (ms.max_segment_memory_frames > 0) max_segment_memory_frames = ms.max_segment_memory_frames;
  if (ms.min_segment_memory_frames > 0) min_segment_memory_frames = ms.min_segment_memory_frames;

  // Pipeline filenames — all 5 must be present in the moonshine config block.
  auto require = [](const std::string& s, const char* field) {
    if (s.empty()) {
      throw std::runtime_error(
          std::string("moonshine streaming model: genai_config.json must set model.moonshine.") +
          field);
    }
  };
  require(ms.frontend_filename,   "frontend_filename");
  require(ms.encoder_filename,    "encoder_filename");
  require(ms.adapter_filename,    "adapter_filename");
  require(ms.cross_kv_filename,   "cross_kv_filename");
  require(ms.decoder_kv_filename, "decoder_kv_filename");
  frontend_filename   = ms.frontend_filename;
  encoder_filename    = ms.encoder_filename;
  adapter_filename    = ms.adapter_filename;
  cross_kv_filename   = ms.cross_kv_filename;
  decoder_kv_filename = ms.decoder_kv_filename;
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

// ----------------------------------------------------------------------------
// Stateless session-call wrappers. These exist so callers (Processor /
// State) don't have to know per-graph input and output names.
// ----------------------------------------------------------------------------

MoonshineStreamingModel::FrontendOutputs MoonshineStreamingModel::RunFrontend(
    OrtValue& audio_chunk,
    OrtValue& sample_buffer,
    OrtValue& sample_len,
    OrtValue& conv1_buffer,
    OrtValue& conv2_buffer,
    OrtValue& frame_count) const {
  const char* in_names[]  = {"audio_chunk", "sample_buffer", "sample_len",
                             "conv1_buffer", "conv2_buffer", "frame_count"};
  OrtValue*   in_values[] = {&audio_chunk, &sample_buffer, &sample_len,
                             &conv1_buffer, &conv2_buffer, &frame_count};
  const char* out_names[] = {"features", "sample_buffer_out", "sample_len_out",
                             "conv1_buffer_out", "conv2_buffer_out", "frame_count_out"};
  OrtValue*   out_values[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

  session_frontend_->Run(nullptr,
                         in_names, in_values, 6,
                         out_names, out_values, 6);

  return FrontendOutputs{
      std::unique_ptr<OrtValue>(out_values[0]),
      std::unique_ptr<OrtValue>(out_values[1]),
      std::unique_ptr<OrtValue>(out_values[2]),
      std::unique_ptr<OrtValue>(out_values[3]),
      std::unique_ptr<OrtValue>(out_values[4]),
      std::unique_ptr<OrtValue>(out_values[5]),
  };
}

std::unique_ptr<OrtValue> MoonshineStreamingModel::RunEncoder(OrtValue& features) const {
  const char* in_names[]  = {"features"};
  OrtValue*   in_values[] = {&features};
  const char* out_names[] = {"encoded"};
  OrtValue*   out_values[1] = {nullptr};
  session_encoder_->Run(nullptr,
                        in_names, in_values, 1,
                        out_names, out_values, 1);
  return std::unique_ptr<OrtValue>(out_values[0]);
}

std::unique_ptr<OrtValue> MoonshineStreamingModel::RunAdapter(OrtValue& encoded,
                                                              OrtValue& pos_offset) const {
  const char* in_names[]  = {"encoded", "pos_offset"};
  OrtValue*   in_values[] = {&encoded, &pos_offset};
  const char* out_names[] = {"memory"};
  OrtValue*   out_values[1] = {nullptr};
  session_adapter_->Run(nullptr,
                        in_names, in_values, 2,
                        out_names, out_values, 1);
  return std::unique_ptr<OrtValue>(out_values[0]);
}

MoonshineStreamingModel::CrossKvOutputs MoonshineStreamingModel::RunCrossKv(OrtValue& memory) const {
  const char* in_names[]  = {"memory"};
  OrtValue*   in_values[] = {&memory};
  const char* out_names[] = {"k_cross", "v_cross"};
  OrtValue*   out_values[2] = {nullptr, nullptr};
  session_cross_kv_->Run(nullptr,
                         in_names, in_values, 1,
                         out_names, out_values, 2);
  return CrossKvOutputs{
      std::unique_ptr<OrtValue>(out_values[0]),
      std::unique_ptr<OrtValue>(out_values[1]),
  };
}

MoonshineStreamingModel::DecoderKvOutputs MoonshineStreamingModel::RunDecoderKv(
    OrtValue& token, OrtValue& k_self, OrtValue& v_self,
    OrtValue& k_cross, OrtValue& v_cross) const {
  const char* in_names[]  = {"token", "k_self", "v_self", "out_k_cross", "out_v_cross"};
  OrtValue*   in_values[] = {&token, &k_self, &v_self, &k_cross, &v_cross};
  const char* out_names[] = {"logits", "out_k_self", "out_v_self",
                             "out_k_cross", "out_v_cross"};
  OrtValue*   out_values[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  session_decoder_kv_->Run(nullptr,
                           in_names, in_values, 5,
                           out_names, out_values, 5);
  // Discard the cross-KV pass-throughs; the caller already owns the inputs.
  (void)std::unique_ptr<OrtValue>(out_values[3]);
  (void)std::unique_ptr<OrtValue>(out_values[4]);
  return DecoderKvOutputs{
      std::unique_ptr<OrtValue>(out_values[0]),
      std::unique_ptr<OrtValue>(out_values[1]),
      std::unique_ptr<OrtValue>(out_values[2]),
  };
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

  // Detect new segment: if memory_len shrinks vs the previous call, the
  // processor was Flush()'d, hit its hard memory cap, or detected a
  // VAD-silence segment boundary — and reset its accumulated memory.
  // Drop the per-pass commit tracking so the next pass starts from BOS,
  // but keep `all_tokens_` so the running transcript spans segment
  // boundaries (callers see one continuous transcript). Users who want a
  // fresh transcript should create a new Generator.
  if (memory_len < previous_memory_len_) {
    previous_pass_tokens_.clear();
    emitted_count_ = 0;
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

  // ---- Decode pass (optimised) ----------------------------------------
  // Cross-KV changes every chunk (memory grew), so cached self-KV from the
  // previous pass is invalid. But the model accepts a dynamic seq dim on
  // its token input, so we can teacher-force the already-committed prefix
  // in ONE parallel decoder call instead of `emitted_count_` sequential AR
  // steps — turning the per-chunk O(emitted_count) decoder calls into O(1)
  // for the prefix, then a small AR loop for the new suffix.
  std::vector<int32_t> current_pass;
  current_pass.reserve(static_cast<size_t>(max_tokens));

  // Prefix: [BOS] + previously-emitted tokens. Even if emitted_count_ == 0
  // (first chunk of a stream), this is just [BOS] — same single-call cost
  // as the old AR-from-BOS first step.
  std::vector<int64_t> prefix;
  prefix.reserve(emitted_count_ + 1);
  prefix.push_back(config_.bos_token_id);
  for (size_t i = 0; i < emitted_count_; ++i) {
    prefix.push_back(static_cast<int64_t>(previous_pass_tokens_[i]));
    current_pass.push_back(previous_pass_tokens_[i]);
  }
  int next_tok = RunDecoderForward(prefix);

  // AR loop for the suffix only. Cap is the total max tokens, so we have
  // `max_tokens - emitted_count_` budget for new tokens.
  const int suffix_budget = max_tokens - static_cast<int>(emitted_count_);
  for (int i = 0; i < suffix_budget; ++i) {
    if (next_tok == config_.eos_token_id) break;
    current_pass.push_back(static_cast<int32_t>(next_tok));
    next_tok = RunDecoderForward(std::vector<int64_t>{static_cast<int64_t>(next_tok)});
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

int MoonshineStreamingState::RunDecoderForward(const std::vector<int64_t>& tokens) {
  const int64_t n = static_cast<int64_t>(tokens.size());
  // Build a [1, N] int64 token tensor. For N==1, reuse the preallocated
  // token_tensor_ to save an allocation per AR step.
  OrtValue* token_input_ptr;
  std::unique_ptr<OrtValue> owned_multi_tok;
  if (n == 1) {
    *token_tensor_->GetTensorMutableData<int64_t>() = tokens[0];
    token_input_ptr = token_tensor_.get();
  } else {
    auto shape = std::array<int64_t, 2>{1, n};
    owned_multi_tok = OrtValue::CreateTensor(moonshine_model_.allocator_cpu_, shape,
                                             ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    std::memcpy(owned_multi_tok->GetTensorMutableData<int64_t>(),
                tokens.data(), static_cast<size_t>(n) * sizeof(int64_t));
    token_input_ptr = owned_multi_tok.get();
  }

  // decoder_kv inputs: token[1,N], k_self, v_self, out_k_cross, out_v_cross.
  // The cross-KV input tensors are kept alive by us for the whole chunk;
  // RunDecoderKv discards the (pass-through) cross-KV outputs internally.
  auto dk = moonshine_model_.RunDecoderKv(*token_input_ptr,
                                          *k_self_,
                                          *v_self_,
                                          *k_cross_tensor_->ort_tensor_,
                                          *v_cross_tensor_->ort_tensor_);
  auto logits = std::move(dk.logits);
  k_self_ = std::move(dk.k_self_out);
  v_self_ = std::move(dk.v_self_out);

  // Greedy argmax over the LAST position's logits (predicting token N).
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
