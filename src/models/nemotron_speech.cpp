// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>
#include <stdexcept>
#include <vector>

#include "../generators.h"
#include "nemo_mel_spectrogram.h"
#include "nemotron_speech.h"

namespace Generators {

void NemotronCacheConfig::PopulateFromConfig(const Config& config) {
  const auto& enc = config.model.encoder;
  const auto& dec = config.model.decoder;
  const auto& jo = config.model.joiner;

  // Encoder dimensions
  hidden_dim = enc.hidden_size;
  num_encoder_layers = enc.num_hidden_layers;

  // Decoder dimensions (LSTM)
  decoder_lstm_dim = dec.hidden_size;
  decoder_lstm_layers = dec.num_hidden_layers;

  // Speech / mel feature config (now at model level)
  num_mels = config.model.num_mels;
  fft_size = config.model.fft_size;
  hop_length = config.model.hop_length;
  win_length = config.model.win_length;
  preemph = config.model.preemph;
  log_eps = config.model.log_eps;
  subsampling_factor = config.model.subsampling_factor;
  left_context = config.model.left_context;
  conv_context = config.model.conv_context;
  pre_encode_cache_size = config.model.pre_encode_cache_size;
  sample_rate = config.model.sample_rate;
  chunk_samples = config.model.chunk_samples;
  blank_id = config.model.blank_id;
  max_symbols_per_step = config.model.max_symbols_per_step;

  // Vocab size from top-level config
  vocab_size = config.model.vocab_size;

  // Encoder I/O names
  enc_in_audio = enc.inputs.audio_features;
  enc_out_encoded = enc.outputs.encoder_outputs;

  // Encoder cache I/O names (from encoder inputs/outputs)
  enc_in_length = enc.inputs.input_lengths;
  enc_in_cache_channel = enc.inputs.cache_last_channel;
  enc_in_cache_time = enc.inputs.cache_last_time;
  enc_in_cache_channel_len = enc.inputs.cache_last_channel_len;
  enc_out_length = enc.outputs.output_lengths;
  enc_out_cache_channel = enc.outputs.cache_last_channel_next;
  enc_out_cache_time = enc.outputs.cache_last_time_next;
  enc_out_cache_channel_len = enc.outputs.cache_last_channel_len_next;

  // Joiner I/O names
  join_in_encoder = jo.inputs.encoder_outputs;
  join_in_decoder = jo.inputs.decoder_outputs;
  join_out_logits = jo.outputs.logits;

  // Decoder I/O names (RNNT prediction network)
  dec_in_targets = dec.inputs.targets;
  dec_in_lstm_hidden = dec.inputs.lstm_hidden_state;
  dec_in_lstm_cell = dec.inputs.lstm_cell_state;
  dec_out_outputs = dec.outputs.outputs;
  dec_out_lstm_hidden = dec.outputs.lstm_hidden_state;
  dec_out_lstm_cell = dec.outputs.lstm_cell_state;
}

void NemotronEncoderCache::Initialize(const NemotronCacheConfig& cfg, const SessionInfo& session_info, OrtAllocator& allocator, DeviceInterface& device) {
  auto cache_channel_type = session_info.GetInputDataType(cfg.enc_in_cache_channel);
  auto cache_time_type = session_info.GetInputDataType(cfg.enc_in_cache_time);
  auto cache_channel_len_type = session_info.GetInputDataType(cfg.enc_in_cache_channel_len);

  // cache_last_channel: [batch, num_layers, left_context, hidden_dim]
  auto ch_shape = std::array<int64_t, 4>{1, cfg.num_encoder_layers, cfg.left_context, cfg.hidden_dim};
  cache_last_channel = OrtValue::CreateTensor(allocator, ch_shape, cache_channel_type);
  ByteWrapTensor(*GetDeviceInterface(DeviceType::CPU), *cache_last_channel).Zero();

  // cache_last_time: [batch, num_layers, hidden_dim, conv_context]
  auto tm_shape = std::array<int64_t, 4>{1, cfg.num_encoder_layers, cfg.hidden_dim, cfg.conv_context};
  cache_last_time = OrtValue::CreateTensor(allocator, tm_shape, cache_time_type);
  ByteWrapTensor(*GetDeviceInterface(DeviceType::CPU), *cache_last_time).Zero();

  // cache_last_channel_len: [1]
  auto len_shape = std::array<int64_t, 1>{1};
  cache_last_channel_len = OrtValue::CreateTensor(allocator, len_shape, cache_channel_len_type);
  *cache_last_channel_len->GetTensorMutableData<int64_t>() = 0;
}

void NemotronEncoderCache::Reset(const NemotronCacheConfig& cfg, const SessionInfo& session_info, OrtAllocator& allocator, DeviceInterface& device) {
  Initialize(cfg, session_info, allocator, device);
}

void NemotronDecoderState::Initialize(const NemotronCacheConfig& cfg, const SessionInfo& session_info, OrtAllocator& allocator, DeviceInterface& device) {
  auto lstm_hidden_type = session_info.GetInputDataType(cfg.dec_in_lstm_hidden);
  auto lstm_cell_type = session_info.GetInputDataType(cfg.dec_in_lstm_cell);

  // LSTM states: [lstm_layers, 1, lstm_dim]
  auto state_shape = std::array<int64_t, 3>{cfg.decoder_lstm_layers, 1, cfg.decoder_lstm_dim};
  lstm_hidden_state = OrtValue::CreateTensor(allocator, state_shape, lstm_hidden_type);
  ByteWrapTensor(*GetDeviceInterface(DeviceType::CPU), *lstm_hidden_state).Zero();

  lstm_cell_state = OrtValue::CreateTensor(allocator, state_shape, lstm_cell_type);
  ByteWrapTensor(*GetDeviceInterface(DeviceType::CPU), *lstm_cell_state).Zero();

  last_token = cfg.blank_id;  // Start with blank/SOS token
}

void NemotronDecoderState::Reset(const NemotronCacheConfig& cfg, const SessionInfo& session_info, OrtAllocator& allocator, DeviceInterface& device) {
  Initialize(cfg, session_info, allocator, device);
}

NemotronSpeechModel::NemotronSpeechModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  cache_config_ = NemotronCacheConfig{};
  cache_config_.PopulateFromConfig(*config_);

  // Create session options
  encoder_session_options_ = OrtSessionOptions::Create();
  decoder_session_options_ = OrtSessionOptions::Create();
  joiner_session_options_ = OrtSessionOptions::Create();

  if (config_->model.encoder.session_options.has_value()) {
    CreateSessionOptionsFromConfig(config_->model.encoder.session_options.value(),
                                   *encoder_session_options_, true, false);
  } else {
    CreateSessionOptionsFromConfig(config_->model.decoder.session_options,
                                   *encoder_session_options_, true, false);
  }
  CreateSessionOptionsFromConfig(config_->model.decoder.session_options,
                                 *decoder_session_options_, true, false);
  if (config_->model.joiner.session_options.has_value()) {
    CreateSessionOptionsFromConfig(config_->model.joiner.session_options.value(),
                                   *joiner_session_options_, true, false);
  } else {
    CreateSessionOptionsFromConfig(config_->model.decoder.session_options,
                                   *joiner_session_options_, true, false);
  }

  // Load the three ONNX models
  std::string encoder_filename = config_->model.encoder.filename;
  if (encoder_filename.empty()) encoder_filename = "encoder.onnx";

  std::string decoder_filename = config_->model.decoder.filename;
  if (decoder_filename.empty()) decoder_filename = "decoder.onnx";

  std::string joiner_filename = config_->model.joiner.filename;
  if (joiner_filename.empty()) joiner_filename = "joiner.onnx";

  session_encoder_ = CreateSession(ort_env, encoder_filename, encoder_session_options_.get());
  session_decoder_ = CreateSession(ort_env, decoder_filename, decoder_session_options_.get());
  session_joiner_ = CreateSession(ort_env, joiner_filename, joiner_session_options_.get());

  session_info_.Add(*session_encoder_);
  session_info_.Add(*session_decoder_);
  session_info_.Add(*session_joiner_);
}

std::unique_ptr<State> NemotronSpeechModel::CreateState(DeviceSpan<int32_t> /*sequence_lengths*/,
                                                        const GeneratorParams& params) const {
  return std::make_unique<NemotronSpeechState>(*this, params);
}

NemotronEncoderSubState::NemotronEncoderSubState(const NemotronSpeechModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {
  auto& cfg = model_.cache_config_;
  auto& allocator = model_.allocator_cpu_;
  auto& device = *model_.p_device_;

  cache_.Initialize(cfg, model_.session_info_, allocator, device);

  has_length_input_ = model_.session_info_.HasInput(cfg.enc_in_length);

  // Create signal_length tensor if the encoder model expects it
  if (has_length_input_) {
    auto len_type = model_.session_info_.GetInputDataType(cfg.enc_in_length);
    auto len_shape = std::array<int64_t, 1>{1};
    signal_length_ = OrtValue::CreateTensor(allocator, len_shape, len_type);
  }

  // Register inputs: mel, [length], cache_channel, cache_time, cache_channel_len
  mel_input_idx_ = inputs_.size();
  input_names_.push_back(cfg.enc_in_audio.c_str());
  inputs_.push_back(nullptr);

  if (has_length_input_) {
    length_input_idx_ = inputs_.size();
    input_names_.push_back(cfg.enc_in_length.c_str());
    inputs_.push_back(signal_length_.get());
  }

  cache_channel_input_idx_ = inputs_.size();
  input_names_.push_back(cfg.enc_in_cache_channel.c_str());
  inputs_.push_back(cache_.cache_last_channel.get());

  cache_time_input_idx_ = inputs_.size();
  input_names_.push_back(cfg.enc_in_cache_time.c_str());
  inputs_.push_back(cache_.cache_last_time.get());

  cache_channel_len_input_idx_ = inputs_.size();
  input_names_.push_back(cfg.enc_in_cache_channel_len.c_str());
  inputs_.push_back(cache_.cache_last_channel_len.get());

  // Register outputs: encoded, length, cache_channel_next, cache_time_next, cache_channel_len_next
  output_names_.push_back(cfg.enc_out_encoded.c_str());
  outputs_.push_back(nullptr);

  output_names_.push_back(cfg.enc_out_length.c_str());
  outputs_.push_back(nullptr);

  output_names_.push_back(cfg.enc_out_cache_channel.c_str());
  outputs_.push_back(nullptr);

  output_names_.push_back(cfg.enc_out_cache_time.c_str());
  outputs_.push_back(nullptr);

  output_names_.push_back(cfg.enc_out_cache_channel_len.c_str());
  outputs_.push_back(nullptr);

  // Set run options from config
  if (model_.config_->model.encoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.encoder.run_options.value());
  }
}

void NemotronEncoderSubState::SetMelInput(OrtValue* mel_tensor, int64_t total_mel_frames) {
  inputs_[mel_input_idx_] = mel_tensor;
  if (has_length_input_) {
    *signal_length_->GetTensorMutableData<int64_t>() = total_mel_frames;
  }
}

void NemotronEncoderSubState::UpdateCacheInputs() {
  inputs_[cache_channel_input_idx_] = cache_.cache_last_channel.get();
  inputs_[cache_time_input_idx_] = cache_.cache_last_time.get();
  inputs_[cache_channel_len_input_idx_] = cache_.cache_last_channel_len.get();
}

DeviceSpan<float> NemotronEncoderSubState::Run(int /*total_length*/, DeviceSpan<int32_t>& /*next_tokens*/, DeviceSpan<int32_t> /*next_indices*/) {
  State::Run(*model_.session_encoder_);
  return {};
}

NemotronPredictionSubState::NemotronPredictionSubState(const NemotronSpeechModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {
  auto& cfg = model_.cache_config_;
  auto& allocator = model_.allocator_cpu_;
  auto& device = *model_.p_device_;

  lstm_state_.Initialize(cfg, model_.session_info_, allocator, device);

  // Create targets tensor
  auto targets_type = model_.session_info_.GetInputDataType(cfg.dec_in_targets);
  auto targets_shape = std::array<int64_t, 2>{1, 1};
  targets_ = OrtValue::CreateTensor(allocator, targets_shape, targets_type);

  // Register inputs
  targets_input_idx_ = inputs_.size();
  input_names_.push_back(cfg.dec_in_targets.c_str());
  inputs_.push_back(targets_.get());

  lstm_hidden_input_idx_ = inputs_.size();
  input_names_.push_back(cfg.dec_in_lstm_hidden.c_str());
  inputs_.push_back(lstm_state_.lstm_hidden_state.get());

  lstm_cell_input_idx_ = inputs_.size();
  input_names_.push_back(cfg.dec_in_lstm_cell.c_str());
  inputs_.push_back(lstm_state_.lstm_cell_state.get());

  // Register outputs: outputs, lstm_hidden, lstm_cell
  output_names_.push_back(cfg.dec_out_outputs.c_str());
  outputs_.push_back(nullptr);

  output_names_.push_back(cfg.dec_out_lstm_hidden.c_str());
  outputs_.push_back(nullptr);

  output_names_.push_back(cfg.dec_out_lstm_cell.c_str());
  outputs_.push_back(nullptr);

  // Set run options from config
  if (model_.config_->model.decoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.decoder.run_options.value());
  }
}

void NemotronPredictionSubState::UpdateInputs() {
  *targets_->GetTensorMutableData<int64_t>() = lstm_state_.last_token;
  inputs_[lstm_hidden_input_idx_] = lstm_state_.lstm_hidden_state.get();
  inputs_[lstm_cell_input_idx_] = lstm_state_.lstm_cell_state.get();
}

DeviceSpan<float> NemotronPredictionSubState::Run(int /*total_length*/, DeviceSpan<int32_t>& /*next_tokens*/, DeviceSpan<int32_t> /*next_indices*/) {
  State::Run(*model_.session_decoder_);
  return {};
}

NemotronJoinerSubState::NemotronJoinerSubState(const NemotronSpeechModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {
  auto& cfg = model_.cache_config_;

  // Register inputs
  encoder_input_idx_ = inputs_.size();
  input_names_.push_back(cfg.join_in_encoder.c_str());
  inputs_.push_back(nullptr);  // Set before each run

  decoder_input_idx_ = inputs_.size();
  input_names_.push_back(cfg.join_in_decoder.c_str());
  inputs_.push_back(nullptr);  // Set before each run

  // Register output
  output_names_.push_back(cfg.join_out_logits.c_str());
  outputs_.push_back(nullptr);

  // Set run options from config
  if (model_.config_->model.joiner.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.joiner.run_options.value());
  }
}

void NemotronJoinerSubState::SetInputFrames(OrtValue* encoder_frame, OrtValue* decoder_frame) {
  inputs_[encoder_input_idx_] = encoder_frame;
  inputs_[decoder_input_idx_] = decoder_frame;
}

DeviceSpan<float> NemotronJoinerSubState::Run(int /*total_length*/, DeviceSpan<int32_t>& /*next_tokens*/, DeviceSpan<int32_t> /*next_indices*/) {
  State::Run(*model_.session_joiner_);
  return {};
}

NemotronSpeechState::NemotronSpeechState(const NemotronSpeechModel& model,
                                         const GeneratorParams& params)
    : State{params, model},
      nemotron_model_{model} {
  cache_config_ = model.cache_config_;

  encoder_state_ = std::make_unique<NemotronEncoderSubState>(model, params);
  prediction_state_ = std::make_unique<NemotronPredictionSubState>(model, params);
  joiner_state_ = std::make_unique<NemotronJoinerSubState>(model, params);

  // Pre-allocate encoder frame for joiner input
  auto enc_out_type = model_.session_info_.GetOutputDataType(cache_config_.enc_out_encoded);
  auto frame_shape = std::array<int64_t, 3>{1, 1, cache_config_.hidden_dim};
  encoder_frame_ = OrtValue::CreateTensor(model_.allocator_cpu_, frame_shape, enc_out_type);
}

NemotronSpeechState::~NemotronSpeechState() = default;

DeviceSpan<float> NemotronSpeechState::Run(int /*total_length*/,
                                           DeviceSpan<int32_t>& /*next_tokens*/,
                                           DeviceSpan<int32_t> /*next_indices*/) {
  throw std::runtime_error(
      "NemotronSpeechState::Run() is not used directly. "
      "Use Generator::GenerateNextToken() with set_inputs.");
}

void NemotronSpeechState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  for (const auto& input : extra_inputs) {
    if (input.name == Config::Defaults::AudioFeaturesName || input.name == cache_config_.enc_in_audio) {
      current_mel_ = input.tensor;
      need_encoder_run_ = true;
      chunk_done_ = false;
    }
  }
}

OrtValue* NemotronSpeechState::GetInput(const char* name) {
  if (auto* val = encoder_state_->GetInput(name)) return val;
  if (auto* val = prediction_state_->GetInput(name)) return val;
  if (auto* val = joiner_state_->GetInput(name)) return val;
  return State::GetInput(name);
}

OrtValue* NemotronSpeechState::GetOutput(const char* name) {
  if (auto* val = encoder_state_->GetOutput(name)) return val;
  if (auto* val = prediction_state_->GetOutput(name)) return val;
  if (auto* val = joiner_state_->GetOutput(name)) return val;
  return State::GetOutput(name);
}

void NemotronSpeechState::ResetStreamingState() {
  auto& allocator = model_.allocator_cpu_;
  auto& device = *model_.p_device_;

  encoder_state_->cache_.Reset(cache_config_, model_.session_info_, allocator, device);
  encoder_state_->UpdateCacheInputs();
  encoder_state_->first_run_ = true;

  prediction_state_->lstm_state_.Reset(cache_config_, model_.session_info_, allocator, device);
  prediction_state_->UpdateInputs();
  prediction_state_->first_run_ = true;

  joiner_state_->first_run_ = true;

  current_mel_.reset();
  encoded_output_.reset();
  encoded_len_ = 0;
  time_step_ = 0;
  symbol_step_ = 0;
  need_encoder_run_ = false;
  chunk_done_ = true;
  last_tokens_.clear();
}

void NemotronSpeechState::RunEncoder() {
  if (!current_mel_ || !current_mel_->ort_tensor_)
    throw std::runtime_error("No mel input set. Call generator.set_model_input(\"audio_features\", mel) first.");

  OrtValue* mel_tensor = current_mel_->ort_tensor_.get();
  int64_t total_mel_frames = mel_tensor->GetTensorTypeAndShapeInfo()->GetShape()[1];

  encoder_state_->SetMelInput(mel_tensor, total_mel_frames);
  encoder_state_->UpdateCacheInputs();

  DeviceSpan<int32_t> dummy_tokens;
  encoder_state_->Run(0, dummy_tokens);

  // Grab encoder outputs
  encoded_output_.reset(encoder_state_->outputs_[0]);
  encoder_state_->outputs_[0] = nullptr;
  encoded_len_ = *encoder_state_->outputs_[1]->GetTensorData<int64_t>();

  // Cache outputs are already moved into encoder_state_->cache_ by Run()
  // But State::Run uses output pointers differently - the outputs are written to by ORT
  // We need to take ownership of the cache outputs
  encoder_state_->cache_.cache_last_channel.reset(encoder_state_->outputs_[2]);
  encoder_state_->outputs_[2] = nullptr;
  encoder_state_->cache_.cache_last_time.reset(encoder_state_->outputs_[3]);
  encoder_state_->outputs_[3] = nullptr;
  encoder_state_->cache_.cache_last_channel_len.reset(encoder_state_->outputs_[4]);
  encoder_state_->outputs_[4] = nullptr;

  current_mel_.reset();
}

std::span<const int32_t> NemotronSpeechState::StepToken() {
  if (need_encoder_run_) {
    RunEncoder();
    need_encoder_run_ = false;
    time_step_ = 0;
    symbol_step_ = 0;
  }

  last_tokens_.clear();

  auto enc_shape = encoded_output_->GetTensorTypeAndShapeInfo()->GetShape();
  int64_t time_steps = std::min(enc_shape[1], encoded_len_);
  int64_t hidden_dim = enc_shape[2];
  size_t frame_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);

  auto enc_span = ByteWrapTensor(*model_.p_device_, *encoded_output_);
  auto frame_span = ByteWrapTensor(*model_.p_device_, *encoder_frame_);
  auto& allocator = model_.allocator_cpu_;

  DeviceSpan<int32_t> dummy_tokens;

  while (time_step_ < time_steps) {
    // Copy current encoder frame
    auto src_frame = enc_span.subspan(static_cast<size_t>(time_step_) * frame_bytes, frame_bytes);
    frame_span.CopyFrom(src_frame);

    // Run prediction network
    prediction_state_->UpdateInputs();
    prediction_state_->Run(0, dummy_tokens);

    // Reshape decoder output for joiner: [1, dim] -> [1, 1, dim]
    auto dec_out_shape = prediction_state_->outputs_[0]->GetTensorTypeAndShapeInfo()->GetShape();
    auto decoder_frame_shape = std::array<int64_t, 3>{1, 1, dec_out_shape[1]};
    auto dec_out_type = model_.session_info_.GetOutputDataType(cache_config_.dec_out_outputs);
    auto decoder_frame = OrtValue::CreateTensor(allocator, decoder_frame_shape, dec_out_type);
    ByteWrapTensor(*model_.p_device_, *decoder_frame)
        .CopyFrom(ByteWrapTensor(*model_.p_device_, *prediction_state_->outputs_[0]));

    // Run joiner
    joiner_state_->SetInputFrames(encoder_frame_.get(), decoder_frame.get());
    joiner_state_->Run(0, dummy_tokens);

    // Argmax over logits
    const float* logits_data = joiner_state_->outputs_[0]->GetTensorData<float>();
    auto logits_shape = joiner_state_->outputs_[0]->GetTensorTypeAndShapeInfo()->GetShape();
    int total_logits = 1;
    for (auto d : logits_shape) total_logits *= static_cast<int>(d);

    int best_token = 0;
    float best_score = logits_data[0];
    for (int i = 1; i < total_logits; ++i) {
      if (logits_data[i] > best_score) {
        best_score = logits_data[i];
        best_token = i;
      }
    }

    if (best_token == cache_config_.blank_id) {
      time_step_++;
      symbol_step_ = 0;
      continue;
    }

    // Non-blank: emit token, update LSTM state from prediction outputs
    prediction_state_->lstm_state_.last_token = best_token;
    prediction_state_->lstm_state_.lstm_hidden_state.reset(prediction_state_->outputs_[1]);
    prediction_state_->outputs_[1] = nullptr;
    prediction_state_->lstm_state_.lstm_cell_state.reset(prediction_state_->outputs_[2]);
    prediction_state_->outputs_[2] = nullptr;

    symbol_step_++;
    if (symbol_step_ >= cache_config_.max_symbols_per_step) {
      time_step_++;
      symbol_step_ = 0;
    }

    last_tokens_.push_back(static_cast<int32_t>(best_token));
    token_count_++;
    return last_tokens_;
  }

  // Exhausted all time steps
  chunk_done_ = true;
  return last_tokens_;
}

}  // namespace Generators
