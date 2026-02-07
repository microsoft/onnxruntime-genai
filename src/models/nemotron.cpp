#include "nemotron.h"
#include <algorithm>
#include <cstring>
#include <numeric>

namespace Generators {

// ============================================================================
// NemotronModel
// ============================================================================

NemotronModel::NemotronModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  // Read dimensions from config
  encoder_hidden_size_ = config_->model.encoder.hidden_size;
  decoder_hidden_size_ = config_->model.decoder.hidden_size;
  num_encoder_layers_ = config_->model.encoder.num_hidden_layers;
  vocab_size_ = config_->model.vocab_size;
  blank_id_ = vocab_size_ - 1;  // RNNT blank is the last token
  num_decoder_layers_ = 2;      // Nemotron uses 2-layer LSTM

  // Read streaming config if present
  // genai_config.json can have: "streaming": { "enabled": true, "cache_last_channel_size": 70, ... }
  // For now, detect streaming by checking if encoder ONNX model has cache inputs
  // This is done after session creation by probing input names.

  // Create encoder session with its own options
  encoder_session_options_ = OrtSessionOptions::Create();
  CreateSessionOptionsFromConfig(
      config_->model.encoder.session_options.has_value()
          ? config_->model.encoder.session_options.value()
          : config_->model.decoder.session_options,
      *encoder_session_options_, true, false);
  session_encoder_ = CreateSession(ort_env, config_->model.encoder.filename, encoder_session_options_.get());

  // Create decoder session using base class session_options_ (same pattern as Whisper)
  session_decoder_ = CreateSession(ort_env, config_->model.decoder.filename, session_options_.get());

  // Load joint network from pipeline config
  if (!config_->model.decoder.pipeline.empty()) {
    auto& joint_config = config_->model.decoder.pipeline[0];
    session_joint_ = CreateSession(ort_env, joint_config.filename, session_options_.get());
  }

  // Detect streaming mode by checking if encoder has cache_last_channel input
  {
    auto input_count = session_encoder_->GetInputCount();
    for (size_t i = 0; i < input_count; i++) {
      auto name = session_encoder_->GetInputName(i);
      if (std::string(name) == "cache_last_channel") {
        streaming_enabled_ = true;
        break;
      }
    }
  }

  session_info_.Add(*session_encoder_);
  session_info_.Add(*session_decoder_);
  if (session_joint_)
    session_info_.Add(*session_joint_);
}

std::unique_ptr<State> NemotronModel::CreateState(
    DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<NemotronState>(*this, params, sequence_lengths);
}

// ============================================================================
// NemotronState
// ============================================================================

NemotronState::NemotronState(const NemotronModel& model, const GeneratorParams& params,
                             DeviceSpan<int32_t> sequence_lengths)
    : State{params, model},
      model_{model} {
  batch_size_ = 1;  // RNNT operates on single utterances
  last_decoder_token_ = model_.blank_id_;  // SOS = blank
}

void NemotronState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  // Find the audio features in the extra inputs
  for (auto& extra : extra_inputs) {
    if (extra.name == model_.config_->model.encoder.inputs.audio_features ||
        extra.name == "audio_signal") {
      // ExtraInput.tensor is a shared_ptr<Tensor>, get the underlying OrtValue
      OrtValue* ort_val = extra.tensor->GetOrtTensor();
      if (ort_val) {
        // Copy the OrtValue data into our own tensor
        auto type_info = ort_val->GetTensorTypeAndShapeInfo();
        auto shape = type_info->GetShape();
        auto type = type_info->GetElementType();

        audio_signal_ = OrtValue::CreateTensor(model_.allocator_cpu_, shape, type);
        auto src_size = type_info->GetElementCount() * sizeof(float);
        std::memcpy(audio_signal_->GetTensorMutableData<float>(),
                    ort_val->GetTensorData<float>(), src_size);
        has_new_audio_ = true;
      }
      break;
    }
  }

  if (!audio_signal_) {
    throw std::runtime_error("NemotronState: audio_signal not provided in extra inputs");
  }
}

// ============================================================================
// Encoder (Non-streaming / Batch mode)
// ============================================================================

void NemotronState::RunEncoder() {
  if (encoder_done_) return;

  // Get audio shape info
  auto type_info = audio_signal_->GetTensorTypeAndShapeInfo();
  auto shape = type_info->GetShape();
  int64_t num_frames = shape[2];  // [B, 128, T]

  // Create length input
  auto length_shape = std::array<int64_t, 1>{static_cast<int64_t>(batch_size_)};
  audio_length_ = OrtValue::CreateTensor<int64_t>(
      model_.allocator_cpu_, length_shape);
  auto* len_data = audio_length_->GetTensorMutableData<int64_t>();
  len_data[0] = num_frames;

  // Set up encoder inputs/outputs
  const char* enc_input_names[] = {"audio_signal", "length"};
  const OrtValue* enc_inputs[] = {audio_signal_.get(), audio_length_.get()};

  const char* enc_output_names[] = {"outputs", "encoded_lengths"};

  // Run encoder — let ONNX Runtime allocate outputs
  auto enc_outputs = model_.session_encoder_->Run(
      nullptr,  // run options
      enc_input_names, enc_inputs, 2,
      enc_output_names, 2);

  encoder_output_raw_ = std::move(enc_outputs[0]);
  encoded_lengths_ = std::move(enc_outputs[1]);

  // Read encoded length
  encoded_length_ = static_cast<int>(encoded_lengths_->GetTensorData<int64_t>()[0]);

  // Transpose encoder output: [B, 1024, T'] -> [B, T', 1024]
  TransposeEncoderOutput();

  encoder_done_ = true;
}

// ============================================================================
// Streaming Encoder (Cache-Aware)
// ============================================================================

void NemotronState::RunStreamingEncoder() {
  if (!has_new_audio_) return;
  has_new_audio_ = false;

  // Initialize caches on first call
  if (!caches_initialized_) {
    // Cache format: [B, n_layers, cache_len, d_model] — matches ONNX model I/O
    auto cache_ch_shape = std::array<int64_t, 4>{
        static_cast<int64_t>(batch_size_),
        static_cast<int64_t>(model_.num_encoder_layers_),
        static_cast<int64_t>(model_.cache_last_channel_size_),
        static_cast<int64_t>(model_.encoder_hidden_size_)};
    size_t cache_ch_size = batch_size_ * model_.num_encoder_layers_ *
                           model_.cache_last_channel_size_ * model_.encoder_hidden_size_;
    cache_last_channel_ = OrtValue::CreateTensor<float>(model_.allocator_cpu_, cache_ch_shape);
    std::memset(cache_last_channel_->GetTensorMutableData<float>(), 0, cache_ch_size * sizeof(float));

    auto cache_tm_shape = std::array<int64_t, 4>{
        static_cast<int64_t>(batch_size_),
        static_cast<int64_t>(model_.num_encoder_layers_),
        static_cast<int64_t>(model_.encoder_hidden_size_),
        static_cast<int64_t>(model_.conv_context_size_)};
    size_t cache_tm_size = batch_size_ * model_.num_encoder_layers_ *
                           model_.encoder_hidden_size_ * model_.conv_context_size_;
    cache_last_time_ = OrtValue::CreateTensor<float>(model_.allocator_cpu_, cache_tm_shape);
    std::memset(cache_last_time_->GetTensorMutableData<float>(), 0, cache_tm_size * sizeof(float));

    auto cache_len_shape = std::array<int64_t, 1>{static_cast<int64_t>(batch_size_)};
    cache_last_channel_len_ = OrtValue::CreateTensor<int64_t>(model_.allocator_cpu_, cache_len_shape);
    cache_last_channel_len_->GetTensorMutableData<int64_t>()[0] = 0;

    caches_initialized_ = true;
  }

  // Get audio shape info
  auto type_info = audio_signal_->GetTensorTypeAndShapeInfo();
  auto shape = type_info->GetShape();
  int64_t num_frames = shape[2];  // [B, 128, chunk_T]

  // Create length input
  auto length_shape = std::array<int64_t, 1>{static_cast<int64_t>(batch_size_)};
  audio_length_ = OrtValue::CreateTensor<int64_t>(
      model_.allocator_cpu_, length_shape);
  audio_length_->GetTensorMutableData<int64_t>()[0] = num_frames;

  // Set up streaming encoder inputs: audio + length + 3 caches
  const char* enc_input_names[] = {
      "audio_signal", "length",
      "cache_last_channel", "cache_last_time", "cache_last_channel_len"};
  const OrtValue* enc_inputs[] = {
      audio_signal_.get(), audio_length_.get(),
      cache_last_channel_.get(), cache_last_time_.get(), cache_last_channel_len_.get()};

  const char* enc_output_names[] = {
      "outputs", "encoded_lengths",
      "cache_last_channel_next", "cache_last_time_next", "cache_last_channel_len_next"};

  // Run streaming encoder
  auto enc_outputs = model_.session_encoder_->Run(
      nullptr,
      enc_input_names, enc_inputs, 5,
      enc_output_names, 5);

  encoder_output_raw_ = std::move(enc_outputs[0]);
  encoded_lengths_ = std::move(enc_outputs[1]);

  // Update caches for next chunk
  cache_last_channel_ = std::move(enc_outputs[2]);
  cache_last_time_ = std::move(enc_outputs[3]);
  cache_last_channel_len_ = std::move(enc_outputs[4]);

  // Read how many encoder output frames this chunk produced
  encoded_length_ = static_cast<int>(encoded_lengths_->GetTensorData<int64_t>()[0]);

  // Transpose encoder output: [B, 1024, T'] -> [B, T', 1024]
  TransposeEncoderOutput();
}

void NemotronState::TransposeEncoderOutput() {
  auto type_info = encoder_output_raw_->GetTensorTypeAndShapeInfo();
  auto shape = type_info->GetShape();
  // shape = [B, 1024, T']
  int64_t B = shape[0];
  int64_t C = shape[1];  // 1024
  int64_t T = shape[2];  // time frames

  const float* src = encoder_output_raw_->GetTensorData<float>();
  encoder_output_transposed_.resize(B * T * C);

  // Transpose [B, C, T] -> [B, T, C]
  for (int64_t b = 0; b < B; b++) {
    for (int64_t t = 0; t < T; t++) {
      for (int64_t c = 0; c < C; c++) {
        encoder_output_transposed_[b * T * C + t * C + c] = src[b * C * T + c * T + t];
      }
    }
  }
}

// ============================================================================
// Decoder (Stateful LSTM)
// ============================================================================

void NemotronState::ResetDecoderState() {
  // Initialize LSTM hidden and cell states to zeros: [num_layers, batch, hidden]
  auto state_shape = std::array<int64_t, 3>{
      static_cast<int64_t>(model_.num_decoder_layers_),
      static_cast<int64_t>(batch_size_),
      static_cast<int64_t>(model_.decoder_hidden_size_)};
  size_t state_size = model_.num_decoder_layers_ * batch_size_ * model_.decoder_hidden_size_;

  decoder_h_state_ = OrtValue::CreateTensor<float>(model_.allocator_cpu_, state_shape);
  decoder_c_state_ = OrtValue::CreateTensor<float>(model_.allocator_cpu_, state_shape);
  std::memset(decoder_h_state_->GetTensorMutableData<float>(), 0, state_size * sizeof(float));
  std::memset(decoder_c_state_->GetTensorMutableData<float>(), 0, state_size * sizeof(float));
}

void NemotronState::RunDecoder(int64_t token_id) {
  // Create target input: [1, 1]
  auto target_shape = std::array<int64_t, 2>{1, 1};
  decoder_targets_ = OrtValue::CreateTensor<int64_t>(
      model_.allocator_cpu_, target_shape);
  decoder_targets_->GetTensorMutableData<int64_t>()[0] = token_id;

  // Create target length: [1]
  auto len_shape = std::array<int64_t, 1>{1};
  decoder_target_length_ = OrtValue::CreateTensor<int64_t>(
      model_.allocator_cpu_, len_shape);
  decoder_target_length_->GetTensorMutableData<int64_t>()[0] = 1;

  // Stateful decoder: pass h_in and c_in as additional inputs
  const char* dec_input_names[] = {"targets", "target_length_orig", "h_in", "c_in"};
  const OrtValue* dec_inputs[] = {
      decoder_targets_.get(),
      decoder_target_length_.get(),
      decoder_h_state_.get(),
      decoder_c_state_.get()};

  const char* dec_output_names[] = {"decoder_output", "target_length", "h_out", "c_out"};

  auto dec_outputs = model_.session_decoder_->Run(
      nullptr,
      dec_input_names, dec_inputs, 4,
      dec_output_names, 4);

  // dec_outputs[0] = decoder_output [B, 640, 1] (stateful: seq_len=1)
  // Extract the hidden state for the Joint network: [B, 640, 1] -> [640]
  auto* dec_data = dec_outputs[0]->GetTensorData<float>();
  auto dec_type_info = dec_outputs[0]->GetTensorTypeAndShapeInfo();
  auto dec_shape = dec_type_info->GetShape();
  int64_t seq_len = dec_shape[2];  // should be 1

  decoder_hidden_out_.resize(model_.decoder_hidden_size_);
  // Extract last time step: for shape [1, 640, S], element [0, c, S-1] = data[c * S + (S-1)]
  for (int c = 0; c < model_.decoder_hidden_size_; c++) {
    decoder_hidden_out_[c] = dec_data[c * seq_len + (seq_len - 1)];
  }

  // Carry forward LSTM states for the next decode step
  decoder_h_state_ = std::move(dec_outputs[2]);
  decoder_c_state_ = std::move(dec_outputs[3]);
}

// ============================================================================
// Joint Network
// ============================================================================

int NemotronState::RunJoint(const float* enc_frame, const float* dec_hidden) {
  if (!model_.session_joint_) {
    throw std::runtime_error("Joint network session not loaded");
  }

  // Create joint encoder input: [1, 1, 1024]
  auto enc_shape = std::array<int64_t, 3>{1, 1, model_.encoder_hidden_size_};
  joint_enc_input_ = OrtValue::CreateTensor<float>(
      model_.allocator_cpu_, enc_shape);
  std::memcpy(joint_enc_input_->GetTensorMutableData<float>(), enc_frame,
              model_.encoder_hidden_size_ * sizeof(float));

  // Create joint decoder input: [1, 1, 640]
  auto dec_shape = std::array<int64_t, 3>{1, 1, model_.decoder_hidden_size_};
  joint_dec_input_ = OrtValue::CreateTensor<float>(
      model_.allocator_cpu_, dec_shape);
  std::memcpy(joint_dec_input_->GetTensorMutableData<float>(), dec_hidden,
              model_.decoder_hidden_size_ * sizeof(float));

  const char* joint_input_names[] = {"encoder_output", "decoder_output"};
  const OrtValue* joint_inputs[] = {joint_enc_input_.get(), joint_dec_input_.get()};

  const char* joint_output_names[] = {"joint_output"};

  auto joint_outputs = model_.session_joint_->Run(
      nullptr,
      joint_input_names, joint_inputs, 2,
      joint_output_names, 1);

  // joint_output shape: [1, 1, 1, 1025]
  const float* logits = joint_outputs[0]->GetTensorData<float>();

  // Argmax over vocab dimension
  int best_token = 0;
  float best_score = logits[0];
  for (int v = 1; v < model_.vocab_size_; v++) {
    if (logits[v] > best_score) {
      best_score = logits[v];
      best_token = v;
    }
  }

  return best_token;
}

// ============================================================================
// RNNT Greedy Decode (Full utterance)
// ============================================================================

void NemotronState::GreedyDecode() {
  decoded_tokens_.clear();

  // Initialize LSTM states to zeros for new utterance
  ResetDecoderState();

  const int max_symbols_per_step = 10;  // Prevent infinite non-blank emissions per frame
  int64_t current_token = model_.blank_id_;  // SOS = blank

  // Initial decoder run with SOS token
  RunDecoder(current_token);
  decoder_initialized_ = true;

  for (int t = 0; t < encoded_length_; t++) {
    // Get encoder frame at time t: pointer into transposed output [B, T', 1024]
    const float* enc_frame = encoder_output_transposed_.data() + t * model_.encoder_hidden_size_;

    for (int sym = 0; sym < max_symbols_per_step; sym++) {
      int next_token = RunJoint(enc_frame, decoder_hidden_out_.data());

      if (next_token == model_.blank_id_) {
        break;  // Advance to next encoder frame
      } else {
        decoded_tokens_.push_back(next_token);
        current_token = next_token;
        RunDecoder(current_token);
      }
    }
  }
  last_decoder_token_ = current_token;
}

// ============================================================================
// RNNT Greedy Decode (Incremental — for streaming)
//
// Decodes only the NEW encoder frames from the latest chunk.
// Decoder LSTM state is carried forward from the previous chunk.
// ============================================================================

void NemotronState::GreedyDecodeIncremental(int num_new_frames) {
  if (num_new_frames <= 0) return;

  // Initialize decoder on first-ever chunk
  if (!decoder_initialized_) {
    ResetDecoderState();
    RunDecoder(last_decoder_token_);
    decoder_initialized_ = true;
  }

  const int max_symbols_per_step = 10;

  for (int t = 0; t < num_new_frames; t++) {
    const float* enc_frame = encoder_output_transposed_.data() + t * model_.encoder_hidden_size_;

    for (int sym = 0; sym < max_symbols_per_step; sym++) {
      int next_token = RunJoint(enc_frame, decoder_hidden_out_.data());

      if (next_token == model_.blank_id_) {
        break;  // Advance to next encoder frame
      } else {
        decoded_tokens_.push_back(next_token);
        last_decoder_token_ = next_token;
        RunDecoder(last_decoder_token_);
      }
    }
  }
}

// ============================================================================
// State::Run — Main entry point
//
// Streaming mode:
//   Each call to Run() processes one chunk of audio through the streaming
//   encoder, then runs incremental RNNT decode on the new encoder frames.
//   Decoded tokens are emitted one at a time through the logits mechanism.
//
// Batch mode:
//   First call runs entire encoder + full RNNT decode. Subsequent calls
//   emit pre-decoded tokens one at a time.
// ============================================================================

DeviceSpan<float> NemotronState::Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                                     DeviceSpan<int32_t> next_indices) {
  if (model_.streaming_enabled_) {
    // --- Streaming mode ---
    // When new audio arrives via SetExtraInputs, run streaming encoder + incremental decode
    if (has_new_audio_) {
      // Run encoder on the new audio chunk (with cache carry-forward)
      RunStreamingEncoder();

      // Run RNNT decode on the NEW encoder output frames only
      GreedyDecodeIncremental(encoded_length_);

      // Update emit_index if new tokens were decoded
      // (emit_index tracks position within decoded_tokens_ for drip-feeding)
      // We don't reset it — just continue from where we left off
    }
  } else {
    // --- Batch mode ---
    // When new audio arrives, run full encoder + RNNT decode (independent per chunk)
    if (has_new_audio_) {
      has_new_audio_ = false;
      encoder_done_ = false;  // Allow re-running encoder for each new chunk
      RunEncoder();
      GreedyDecode();
      emit_index_ = 0;
      rnnt_done_ = false;
    } else if (!encoder_done_) {
      // First call without explicit SetExtraInputs (shouldn't happen normally)
      RunEncoder();
      GreedyDecode();
      emit_index_ = 0;
      rnnt_done_ = false;
    }
  }

  // Create logits tensor if not yet allocated
  if (!logits_tensor_) {
    auto logits_shape = std::array<int64_t, 2>{1, static_cast<int64_t>(model_.vocab_size_)};
    logits_tensor_ = OrtValue::CreateTensor<float>(
        model_.allocator_cpu_, logits_shape);
  }

  float* logits_data = logits_tensor_->GetTensorMutableData<float>();

  // Fill logits: make the target token have the highest score
  std::fill(logits_data, logits_data + model_.vocab_size_, -100.0f);

  if (emit_index_ < static_cast<int>(decoded_tokens_.size())) {
    // Emit the next decoded token
    int32_t token = decoded_tokens_[emit_index_];
    logits_data[token] = 10.0f;  // High score for the decoded token
    emit_index_++;
  } else {
    // All tokens emitted — signal EOS via eos_token_id (blank=1024)
    logits_data[model_.blank_id_] = 10.0f;
    rnnt_done_ = true;
  }

  // Wrap the logits tensor as a DeviceSpan for the framework
  logits_span_ = WrapTensor<float>(*model_.p_device_inputs_, *logits_tensor_);
  return logits_span_;
}

OrtValue* NemotronState::GetInput(const char* name) {
  if (audio_signal_ && std::strcmp(name, "audio_signal") == 0)
    return audio_signal_.get();
  if (audio_length_ && std::strcmp(name, "length") == 0)
    return audio_length_.get();
  if (cache_last_channel_ && std::strcmp(name, "cache_last_channel") == 0)
    return cache_last_channel_.get();
  if (cache_last_time_ && std::strcmp(name, "cache_last_time") == 0)
    return cache_last_time_.get();
  if (cache_last_channel_len_ && std::strcmp(name, "cache_last_channel_len") == 0)
    return cache_last_channel_len_.get();
  return nullptr;
}

OrtValue* NemotronState::GetOutput(const char* name) {
  if (encoder_output_raw_ && std::strcmp(name, "outputs") == 0)
    return encoder_output_raw_.get();
  if (encoded_lengths_ && std::strcmp(name, "encoded_lengths") == 0)
    return encoded_lengths_.get();
  if (logits_tensor_ && std::strcmp(name, "logits") == 0)
    return logits_tensor_.get();
  return nullptr;
}

}  // namespace Generators
