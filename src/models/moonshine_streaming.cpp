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
  if (dec.num_hidden_layers > 0) num_decoder_layers = dec.num_hidden_layers;
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
// Sub-states. Each wraps a single ONNX session behind State::Run(). All
// persistent cross-chunk state lives in the orchestrator; these sub-states
// only register their I/O and expose setters for the per-run inputs. The
// orchestrator is a friend, so after each Run it takes ownership of the
// freshly-allocated output tensors (outputs_[i]) and nulls the slots.
// ----------------------------------------------------------------------------

// ---- frontend --------------------------------------------------------------

MoonshineFrontendSubState::MoonshineFrontendSubState(const MoonshineStreamingModel& model,
                                                     const GeneratorParams& params)
    : State{params, model},
      model_{model} {
  config_ = model.moonshine_config_;
  AllocateStateBuffers();

  // Inputs: audio_chunk (set per-run) + the 5 persistent state buffers.
  audio_input_idx_ = inputs_.size();
  input_names_.push_back("audio_chunk");
  inputs_.push_back(nullptr);

  sample_buffer_input_idx_ = inputs_.size();
  input_names_.push_back("sample_buffer");
  inputs_.push_back(sample_buffer_.get());

  sample_len_input_idx_ = inputs_.size();
  input_names_.push_back("sample_len");
  inputs_.push_back(sample_len_.get());

  conv1_buffer_input_idx_ = inputs_.size();
  input_names_.push_back("conv1_buffer");
  inputs_.push_back(conv1_buffer_.get());

  conv2_buffer_input_idx_ = inputs_.size();
  input_names_.push_back("conv2_buffer");
  inputs_.push_back(conv2_buffer_.get());

  frame_count_input_idx_ = inputs_.size();
  input_names_.push_back("frame_count");
  inputs_.push_back(frame_count_.get());

  // Outputs: features + the 5 updated state buffers.
  output_names_.push_back("features");
  outputs_.push_back(nullptr);
  output_names_.push_back("sample_buffer_out");
  outputs_.push_back(nullptr);
  output_names_.push_back("sample_len_out");
  outputs_.push_back(nullptr);
  output_names_.push_back("conv1_buffer_out");
  outputs_.push_back(nullptr);
  output_names_.push_back("conv2_buffer_out");
  outputs_.push_back(nullptr);
  output_names_.push_back("frame_count_out");
  outputs_.push_back(nullptr);
}

void MoonshineFrontendSubState::AllocateStateBuffers() {
  auto& alloc = model_.allocator_cpu_;
  {
    auto shape = std::array<int64_t, 2>{1, config_.sample_buffer_size};
    sample_buffer_ = OrtValue::CreateTensor(alloc, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::memset(sample_buffer_->GetTensorMutableData<float>(), 0,
                static_cast<size_t>(config_.sample_buffer_size) * sizeof(float));
  }
  {
    auto shape = std::array<int64_t, 1>{1};
    sample_len_ = OrtValue::CreateTensor(alloc, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    *sample_len_->GetTensorMutableData<int64_t>() = 0;
  }
  {
    auto shape = std::array<int64_t, 3>{1, config_.conv1_channels, config_.conv1_buffer_size};
    conv1_buffer_ = OrtValue::CreateTensor(alloc, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::memset(conv1_buffer_->GetTensorMutableData<float>(), 0,
                static_cast<size_t>(config_.conv1_channels) *
                    static_cast<size_t>(config_.conv1_buffer_size) * sizeof(float));
  }
  {
    auto shape = std::array<int64_t, 3>{1, config_.conv2_channels, config_.conv2_buffer_size};
    conv2_buffer_ = OrtValue::CreateTensor(alloc, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::memset(conv2_buffer_->GetTensorMutableData<float>(), 0,
                static_cast<size_t>(config_.conv2_channels) *
                    static_cast<size_t>(config_.conv2_buffer_size) * sizeof(float));
  }
  {
    auto shape = std::array<int64_t, 1>{1};
    frame_count_ = OrtValue::CreateTensor(alloc, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    *frame_count_->GetTensorMutableData<int64_t>() = 0;
  }
}

void MoonshineFrontendSubState::SetAudioInput(OrtValue* audio_chunk) {
  inputs_[audio_input_idx_] = audio_chunk;
}

void MoonshineFrontendSubState::UpdateStateInputs() {
  inputs_[sample_buffer_input_idx_] = sample_buffer_.get();
  inputs_[sample_len_input_idx_] = sample_len_.get();
  inputs_[conv1_buffer_input_idx_] = conv1_buffer_.get();
  inputs_[conv2_buffer_input_idx_] = conv2_buffer_.get();
  inputs_[frame_count_input_idx_] = frame_count_.get();
}

void MoonshineFrontendSubState::Reset() {
  AllocateStateBuffers();
  UpdateStateInputs();
}

DeviceSpan<float> MoonshineFrontendSubState::Run(int /*total_length*/,
                                                 DeviceSpan<int32_t>& /*next_tokens*/,
                                                 DeviceSpan<int32_t> /*next_indices*/) {
  State::Run(*model_.session_frontend_);
  return {};
}

// ---- encoder ---------------------------------------------------------------

MoonshineEncoderSubState::MoonshineEncoderSubState(const MoonshineStreamingModel& model,
                                                   const GeneratorParams& params)
    : State{params, model},
      model_{model} {
  input_names_.push_back("features");
  inputs_.push_back(nullptr);
  output_names_.push_back("encoded");
  outputs_.push_back(nullptr);
}

void MoonshineEncoderSubState::SetFeaturesInput(OrtValue* features) {
  inputs_[0] = features;
}

DeviceSpan<float> MoonshineEncoderSubState::Run(int /*total_length*/,
                                                DeviceSpan<int32_t>& /*next_tokens*/,
                                                DeviceSpan<int32_t> /*next_indices*/) {
  State::Run(*model_.session_encoder_);
  return {};
}

// ---- adapter ---------------------------------------------------------------

MoonshineAdapterSubState::MoonshineAdapterSubState(const MoonshineStreamingModel& model,
                                                   const GeneratorParams& params)
    : State{params, model},
      model_{model} {
  auto pos_shape = std::array<int64_t, 1>{1};
  pos_tensor_ = OrtValue::CreateTensor(model_.allocator_cpu_, pos_shape,
                                       ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *pos_tensor_->GetTensorMutableData<int64_t>() = 0;

  encoded_input_idx_ = inputs_.size();
  input_names_.push_back("encoded");
  inputs_.push_back(nullptr);

  pos_input_idx_ = inputs_.size();
  input_names_.push_back("pos_offset");
  inputs_.push_back(pos_tensor_.get());

  output_names_.push_back("memory");
  outputs_.push_back(nullptr);
}

void MoonshineAdapterSubState::SetInputs(OrtValue* encoded, int64_t pos_offset) {
  inputs_[encoded_input_idx_] = encoded;
  *pos_tensor_->GetTensorMutableData<int64_t>() = pos_offset;
  inputs_[pos_input_idx_] = pos_tensor_.get();
}

DeviceSpan<float> MoonshineAdapterSubState::Run(int /*total_length*/,
                                                DeviceSpan<int32_t>& /*next_tokens*/,
                                                DeviceSpan<int32_t> /*next_indices*/) {
  State::Run(*model_.session_adapter_);
  return {};
}

// ---- cross_kv --------------------------------------------------------------

MoonshineCrossKvSubState::MoonshineCrossKvSubState(const MoonshineStreamingModel& model,
                                                   const GeneratorParams& params)
    : State{params, model},
      model_{model} {
  input_names_.push_back("memory");
  inputs_.push_back(nullptr);
  output_names_.push_back("k_cross");
  outputs_.push_back(nullptr);
  output_names_.push_back("v_cross");
  outputs_.push_back(nullptr);
}

void MoonshineCrossKvSubState::SetMemoryInput(OrtValue* memory) {
  inputs_[0] = memory;
}

DeviceSpan<float> MoonshineCrossKvSubState::Run(int /*total_length*/,
                                                DeviceSpan<int32_t>& /*next_tokens*/,
                                                DeviceSpan<int32_t> /*next_indices*/) {
  State::Run(*model_.session_cross_kv_);
  return {};
}

// ---- decoder_kv ------------------------------------------------------------

MoonshineDecoderKvSubState::MoonshineDecoderKvSubState(const MoonshineStreamingModel& model,
                                                       const GeneratorParams& params)
    : State{params, model},
      model_{model} {
  token_input_idx_ = inputs_.size();
  input_names_.push_back("token");
  inputs_.push_back(nullptr);

  k_self_input_idx_ = inputs_.size();
  input_names_.push_back("k_self");
  inputs_.push_back(nullptr);

  v_self_input_idx_ = inputs_.size();
  input_names_.push_back("v_self");
  inputs_.push_back(nullptr);

  k_cross_input_idx_ = inputs_.size();
  input_names_.push_back("out_k_cross");
  inputs_.push_back(nullptr);

  v_cross_input_idx_ = inputs_.size();
  input_names_.push_back("out_v_cross");
  inputs_.push_back(nullptr);

  output_names_.push_back("logits");
  outputs_.push_back(nullptr);
  output_names_.push_back("out_k_self");
  outputs_.push_back(nullptr);
  output_names_.push_back("out_v_self");
  outputs_.push_back(nullptr);
  output_names_.push_back("out_k_cross");
  outputs_.push_back(nullptr);
  output_names_.push_back("out_v_cross");
  outputs_.push_back(nullptr);
}

void MoonshineDecoderKvSubState::SetInputs(OrtValue* token, OrtValue* k_self, OrtValue* v_self,
                                           OrtValue* k_cross, OrtValue* v_cross) {
  inputs_[token_input_idx_] = token;
  inputs_[k_self_input_idx_] = k_self;
  inputs_[v_self_input_idx_] = v_self;
  inputs_[k_cross_input_idx_] = k_cross;
  inputs_[v_cross_input_idx_] = v_cross;
}

DeviceSpan<float> MoonshineDecoderKvSubState::Run(int /*total_length*/,
                                                  DeviceSpan<int32_t>& /*next_tokens*/,
                                                  DeviceSpan<int32_t> /*next_indices*/) {
  State::Run(*model_.session_decoder_kv_);
  return {};
}


MoonshineStreamingState::MoonshineStreamingState(const MoonshineStreamingModel& model,
                                                 const GeneratorParams& params)
    : TransducerState{params, model},
      moonshine_model_{model} {
  config_ = model.moonshine_config_;

  // Create the five ONNX sub-states (each needs the real GeneratorParams).
  frontend_state_ = std::make_unique<MoonshineFrontendSubState>(model, params);
  encoder_state_ = std::make_unique<MoonshineEncoderSubState>(model, params);
  adapter_state_ = std::make_unique<MoonshineAdapterSubState>(model, params);
  cross_kv_state_ = std::make_unique<MoonshineCrossKvSubState>(model, params);
  decoder_kv_state_ = std::make_unique<MoonshineDecoderKvSubState>(model, params);

  // Idle until SetExtraInputs() supplies a chunk.
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
  // Only cache the chunk + signals here and flag a pending run. The Generator
  // pump may call SetExtraInputs more than once per chunk (twice on the very
  // first chunk), so the heavy, state-mutating pipeline runs exactly once from
  // the first StepToken() (gated by need_pipeline_run_), matching Nemotron.
  std::shared_ptr<Tensor> audio_chunk;
  bool is_silent = false;
  bool is_final = false;
  for (const auto& input : extra_inputs) {
    if (input.name == "audio_chunk") {
      audio_chunk = input.tensor;
    } else if (input.name == "is_silent") {
      if (input.tensor && input.tensor->ort_tensor_) {
        is_silent = (*input.tensor->ort_tensor_->GetTensorData<int64_t>()) != 0;
      }
    } else if (input.name == "is_final") {
      if (input.tensor && input.tensor->ort_tensor_) {
        is_final = (*input.tensor->ort_tensor_->GetTensorData<int64_t>()) != 0;
      }
    }
  }
  if (!audio_chunk) return;  // No new chunk this call (idempotent no-op).

  current_audio_ = std::move(audio_chunk);
  current_is_silent_ = is_silent;
  current_is_final_ = is_final;
  need_pipeline_run_ = true;
  chunk_done_ = false;
}

void MoonshineStreamingState::RunPipeline() {
  // Reset the per-chunk queue so any early return (dropped silence, no stable
  // frames yet) leaves the chunk with zero committed tokens.
  pending_tokens_.clear();
  pending_idx_ = 0;
  last_tokens_.clear();

  // Deferred reset from a previous segment close (hard cap / VAD / Flush).
  if (needs_reset_) {
    ResetAccumulation();
    needs_reset_ = false;
  }

  const float* audio = nullptr;
  size_t num = 0;
  if (current_audio_ && current_audio_->ort_tensor_) {
    auto shape = current_audio_->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape();
    num = shape.empty() ? 0 : static_cast<size_t>(shape.back());
    if (num > 0) audio = current_audio_->ort_tensor_->GetTensorData<float>();
  }

  const bool flush = current_is_final_;
  const bool silent = current_is_silent_;
  bool commit_all = flush;

  if (flush) {
    // Flush: release held-back lookahead and accumulate the tail.
    RunFrontendAndAccumulate(audio, num, /*is_final=*/true);
    if (memory_len_ == 0) { chunk_done_ = true; return; }
    RefreshCrossKv();
    needs_reset_ = true;
  } else if (silent) {
    // VAD-based segmentation (runs the per-chunk silence verdict computed by
    // the processor). Routing mirrors the previous processor logic:
    const bool past_min = config_.min_segment_memory_frames > 0 &&
                          memory_len_ >= config_.min_segment_memory_frames;
    if (past_min) {
      // Flush the previous speech's held-back lookahead, then break as final.
      RunFrontendAndAccumulate(nullptr, 0, /*is_final=*/true);
      if (memory_len_ == 0) { chunk_done_ = true; return; }
      RefreshCrossKv();
      needs_reset_ = true;
      commit_all = true;
    } else if (memory_len_ == 0) {
      // Pre-utterance silence: skip the encoder entirely.
      chunk_done_ = true;
      return;
    } else {
      // Mid-segment short silence (< min_segment): encode normally to keep the
      // lookahead boundary continuous.
      RunFrontendAndAccumulate(audio, num, /*is_final=*/false);
      if (memory_len_ == 0) { chunk_done_ = true; return; }
      RefreshCrossKv();
    }
  } else {
    // Speech (or VAD disabled).
    RunFrontendAndAccumulate(audio, num, /*is_final=*/false);
    if (memory_len_ == 0) { chunk_done_ = true; return; }
    RefreshCrossKv();
    // Hard-cap reset: if memory crossed the segment cap, commit ALL tokens and
    // schedule a reset for the next chunk.
    if (config_.max_segment_memory_frames > 0 &&
        memory_len_ >= config_.max_segment_memory_frames) {
      needs_reset_ = true;
      commit_all = true;
    }
  }

  DecodeAndQueue(commit_all);
}

void MoonshineStreamingState::ResetAccumulation() {
  frontend_state_->Reset();
  accumulated_features_.clear();
  total_features_ = 0;
  encoder_frames_emitted_ = 0;
  adapter_pos_offset_ = 0;
  accumulated_memory_.clear();
  memory_len_ = 0;
  cached_k_cross_.reset();
  cached_v_cross_.reset();
  memory_in_cross_kv_ = 0;
  cross_kv_valid_ = false;
}

void MoonshineStreamingState::RunFrontendAndAccumulate(const float* audio, size_t num,
                                                       bool is_final) {
  auto& alloc = moonshine_model_.allocator_cpu_;
  const int encoder_dim = config_.encoder_dim;
  const int decoder_dim = config_.decoder_dim;
  DeviceSpan<int32_t> dummy_tokens;

  // ----- frontend -------------------------------------------------------
  // Run only when there is audio to feed (frontend audio dim must be > 0).
  // On Flush with num==0 we skip the frontend but still release lookahead in
  // the encoder step below via is_final.
  if (num > 0) {
    auto audio_shape = std::array<int64_t, 2>{1, static_cast<int64_t>(num)};
    auto audio_tensor =
        OrtValue::CreateTensor(alloc, audio_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::memcpy(audio_tensor->GetTensorMutableData<float>(), audio, num * sizeof(float));

    frontend_state_->SetAudioInput(audio_tensor.get());
    frontend_state_->UpdateStateInputs();
    frontend_state_->Run(0, dummy_tokens);

    // Take ownership of the frontend outputs. outputs_[0] is the new feature
    // frames; outputs_[1..5] are the updated causal state buffers which cycle
    // back into the sub-state's inputs for the next chunk.
    std::unique_ptr<OrtValue> features{frontend_state_->outputs_[0]};
    frontend_state_->outputs_[0] = nullptr;
    frontend_state_->sample_buffer_.reset(frontend_state_->outputs_[1]);
    frontend_state_->outputs_[1] = nullptr;
    frontend_state_->sample_len_.reset(frontend_state_->outputs_[2]);
    frontend_state_->outputs_[2] = nullptr;
    frontend_state_->conv1_buffer_.reset(frontend_state_->outputs_[3]);
    frontend_state_->outputs_[3] = nullptr;
    frontend_state_->conv2_buffer_.reset(frontend_state_->outputs_[4]);
    frontend_state_->outputs_[4] = nullptr;
    frontend_state_->frame_count_.reset(frontend_state_->outputs_[5]);
    frontend_state_->outputs_[5] = nullptr;
    frontend_state_->UpdateStateInputs();

    auto fshape = features->GetTensorTypeAndShapeInfo()->GetShape();
    if (fshape.size() >= 3 && fshape[1] > 0) {
      const int new_features = static_cast<int>(fshape[1]);
      const float* fdata = features->GetTensorData<float>();
      accumulated_features_.insert(accumulated_features_.end(), fdata,
                                   fdata + static_cast<size_t>(new_features) * encoder_dim);
      total_features_ += new_features;
    }
  }

  // ----- encoder / adapter step -----------------------------------------
  // Hold back the lookahead window unless this is the final chunk.
  const int lookahead = config_.total_lookahead;
  const int stable_count = is_final ? total_features_
                                    : std::max(0, total_features_ - lookahead);
  const int new_frames = stable_count - encoder_frames_emitted_;
  if (new_frames <= 0) return;

  // Encoder runs on [window_start : total_features_] for left context.
  const int left_context = config_.left_context_frames;
  const int window_start = std::max(0, encoder_frames_emitted_ - left_context);
  const int window_size = total_features_ - window_start;
  const int start_idx = encoder_frames_emitted_ - window_start;  // new frames offset in encoded.

  auto enc_in_shape = std::array<int64_t, 3>{1, window_size, encoder_dim};
  auto enc_in_tensor =
      OrtValue::CreateTensor(alloc, enc_in_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memcpy(enc_in_tensor->GetTensorMutableData<float>(),
              accumulated_features_.data() + static_cast<size_t>(window_start) * encoder_dim,
              static_cast<size_t>(window_size) * encoder_dim * sizeof(float));

  encoder_state_->SetFeaturesInput(enc_in_tensor.get());
  encoder_state_->Run(0, dummy_tokens);
  std::unique_ptr<OrtValue> encoded{encoder_state_->outputs_[0]};
  encoder_state_->outputs_[0] = nullptr;

  // Slice [:, start_idx : start_idx + new_frames] from encoded.
  auto new_enc_shape = std::array<int64_t, 3>{1, new_frames, encoder_dim};
  auto new_enc_tensor =
      OrtValue::CreateTensor(alloc, new_enc_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memcpy(new_enc_tensor->GetTensorMutableData<float>(),
              encoded->GetTensorData<float>() +
                  static_cast<size_t>(start_idx) * encoder_dim,
              static_cast<size_t>(new_frames) * encoder_dim * sizeof(float));

  // ----- adapter --------------------------------------------------------
  adapter_state_->SetInputs(new_enc_tensor.get(), adapter_pos_offset_);
  adapter_state_->Run(0, dummy_tokens);
  std::unique_ptr<OrtValue> memory_tensor{adapter_state_->outputs_[0]};
  adapter_state_->outputs_[0] = nullptr;

  // Append adapter output to accumulated memory.
  auto mshape = memory_tensor->GetTensorTypeAndShapeInfo()->GetShape();
  const int produced_frames = (mshape.size() >= 3) ? static_cast<int>(mshape[1]) : 0;
  if (produced_frames > 0) {
    const float* mdata = memory_tensor->GetTensorData<float>();
    accumulated_memory_.insert(accumulated_memory_.end(), mdata,
                               mdata + static_cast<size_t>(produced_frames) * decoder_dim);
    memory_len_ += produced_frames;
  }

  encoder_frames_emitted_ = stable_count;
  adapter_pos_offset_ += new_frames;
  cross_kv_valid_ = false;  // memory grew; cached cross-KV needs to catch up.
}

void MoonshineStreamingState::RefreshCrossKv() {
  if (cross_kv_valid_ || memory_len_ == 0) return;

  auto& alloc = moonshine_model_.allocator_cpu_;
  const int decoder_dim = config_.decoder_dim;
  const int new_frames = memory_len_ - memory_in_cross_kv_;
  DeviceSpan<int32_t> dummy_tokens;

  // Run cross_kv on JUST the new memory frames (pure per-frame projection).
  auto mem_shape = std::array<int64_t, 3>{1, new_frames, decoder_dim};
  auto mem_tensor =
      OrtValue::CreateTensor(alloc, mem_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memcpy(mem_tensor->GetTensorMutableData<float>(),
              accumulated_memory_.data() +
                  static_cast<size_t>(memory_in_cross_kv_) * decoder_dim,
              static_cast<size_t>(new_frames) * decoder_dim * sizeof(float));

  cross_kv_state_->SetMemoryInput(mem_tensor.get());
  cross_kv_state_->Run(0, dummy_tokens);
  std::unique_ptr<OrtValue> k_new{cross_kv_state_->outputs_[0]};
  cross_kv_state_->outputs_[0] = nullptr;
  std::unique_ptr<OrtValue> v_new{cross_kv_state_->outputs_[1]};
  cross_kv_state_->outputs_[1] = nullptr;

  if (memory_in_cross_kv_ == 0) {
    // First chunk of the segment: nothing to concat with.
    cached_k_cross_ = std::make_shared<Tensor>(std::move(k_new));
    cached_v_cross_ = std::make_shared<Tensor>(std::move(v_new));
  } else {
    // Concat cached [L,1,H,M_old,D] + new [L,1,H,new_frames,D] along dim 3.
    const int L = config_.num_decoder_layers;
    const int H = config_.num_decoder_heads;
    const int D = config_.decoder_head_size;
    const int M_old = memory_in_cross_kv_;
    const int M_total = memory_len_;
    auto concat_shape = std::array<int64_t, 5>{L, 1, H, M_total, D};

    auto concat_one = [&](const float* old_data, const float* new_data) {
      auto out =
          OrtValue::CreateTensor(alloc, concat_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      float* dst = out->GetTensorMutableData<float>();
      const size_t row_old = static_cast<size_t>(M_old) * D;
      const size_t row_new = static_cast<size_t>(new_frames) * D;
      const size_t row_total = static_cast<size_t>(M_total) * D;
      for (int l = 0; l < L; ++l) {
        for (int h = 0; h < H; ++h) {
          const size_t lh = static_cast<size_t>(l) * H + h;
          std::memcpy(dst + lh * row_total,
                      old_data + lh * row_old,
                      row_old * sizeof(float));
          std::memcpy(dst + lh * row_total + row_old,
                      new_data + lh * row_new,
                      row_new * sizeof(float));
        }
      }
      return out;
    };

    auto k_full = concat_one(cached_k_cross_->ort_tensor_->GetTensorData<float>(),
                             k_new->GetTensorData<float>());
    auto v_full = concat_one(cached_v_cross_->ort_tensor_->GetTensorData<float>(),
                             v_new->GetTensorData<float>());
    cached_k_cross_ = std::make_shared<Tensor>(std::move(k_full));
    cached_v_cross_ = std::make_shared<Tensor>(std::move(v_full));
  }

  memory_in_cross_kv_ = memory_len_;
  cross_kv_valid_ = true;
}

void MoonshineStreamingState::DecodeAndQueue(bool is_final) {
  // Point the decoder at the current chunk's cross-KV for the whole pass.
  k_cross_tensor_ = cached_k_cross_;
  v_cross_tensor_ = cached_v_cross_;

  // Reset per-pass decoder state. (Self-KV gets rebuilt fresh each pass since
  // we re-decode from BOS over the full accumulated memory.)
  ResetSelfKv();
  pending_tokens_.clear();
  pending_idx_ = 0;
  last_tokens_.clear();

  int64_t memory_len = memory_len_;

  // Detect new segment: if memory_len shrinks vs the previous pass, a segment
  // was closed (hard cap / VAD silence / Flush) and accumulation was reset.
  // Drop the per-pass commit tracking so the next pass starts from BOS, but
  // keep `all_tokens_` so the running transcript spans segment boundaries
  // (callers see one continuous transcript). Users who want a fresh transcript
  // should create a new Generator.
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

  // Run the heavy pipeline exactly once per chunk, on the first StepToken.
  // The Generator pump may call SetExtraInputs more than once per chunk, so
  // SetExtraInputs only caches the chunk + sets need_pipeline_run_.
  if (need_pipeline_run_) {
    RunPipeline();
    need_pipeline_run_ = false;
  }

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
  // The cross-KV input tensors are kept alive by us for the whole chunk; the
  // decoder passes them straight through, so we discard those outputs.
  DeviceSpan<int32_t> dummy_tokens;
  decoder_kv_state_->SetInputs(token_input_ptr, k_self_.get(), v_self_.get(),
                               k_cross_tensor_->ort_tensor_.get(),
                               v_cross_tensor_->ort_tensor_.get());
  decoder_kv_state_->Run(0, dummy_tokens);

  // Outputs: logits, out_k_self, out_v_self, out_k_cross, out_v_cross.
  std::unique_ptr<OrtValue> logits{decoder_kv_state_->outputs_[0]};
  decoder_kv_state_->outputs_[0] = nullptr;
  k_self_.reset(decoder_kv_state_->outputs_[1]);
  decoder_kv_state_->outputs_[1] = nullptr;
  v_self_.reset(decoder_kv_state_->outputs_[2]);
  decoder_kv_state_->outputs_[2] = nullptr;
  // Discard the pass-through cross-KV outputs (freed at scope exit).
  std::unique_ptr<OrtValue> k_cross_out_owner{decoder_kv_state_->outputs_[3]};
  decoder_kv_state_->outputs_[3] = nullptr;
  std::unique_ptr<OrtValue> v_cross_out_owner{decoder_kv_state_->outputs_[4]};
  decoder_kv_state_->outputs_[4] = nullptr;

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

OrtValue* MoonshineStreamingState::GetInput(const char* name) {
  if (auto* v = frontend_state_->GetInput(name)) return v;
  if (auto* v = encoder_state_->GetInput(name)) return v;
  if (auto* v = adapter_state_->GetInput(name)) return v;
  if (auto* v = cross_kv_state_->GetInput(name)) return v;
  if (auto* v = decoder_kv_state_->GetInput(name)) return v;
  return State::GetInput(name);
}

OrtValue* MoonshineStreamingState::GetOutput(const char* name) {
  if (auto* v = frontend_state_->GetOutput(name)) return v;
  if (auto* v = encoder_state_->GetOutput(name)) return v;
  if (auto* v = adapter_state_->GetOutput(name)) return v;
  if (auto* v = cross_kv_state_->GetOutput(name)) return v;
  if (auto* v = decoder_kv_state_->GetOutput(name)) return v;
  return State::GetOutput(name);
}

}  // namespace Generators
