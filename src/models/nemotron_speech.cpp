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
  dec_in_target_length = dec.inputs.target_length;
  dec_in_lstm_hidden = dec.inputs.lstm_hidden_state;
  dec_in_lstm_cell = dec.inputs.lstm_cell_state;
  dec_out_outputs = dec.outputs.outputs;
  dec_out_prednet_lengths = dec.outputs.prednet_lengths;
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
  ByteWrapTensor(device, *cache_last_channel).Zero();

  // cache_last_time: [batch, num_layers, hidden_dim, conv_context]
  auto tm_shape = std::array<int64_t, 4>{1, cfg.num_encoder_layers, cfg.hidden_dim, cfg.conv_context};
  cache_last_time = OrtValue::CreateTensor(allocator, tm_shape, cache_time_type);
  ByteWrapTensor(device, *cache_last_time).Zero();

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
  ByteWrapTensor(device, *lstm_hidden_state).Zero();

  lstm_cell_state = OrtValue::CreateTensor(allocator, state_shape, lstm_cell_type);
  ByteWrapTensor(device, *lstm_cell_state).Zero();

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

NemotronSpeechState::NemotronSpeechState(const NemotronSpeechModel& model,
                                         const GeneratorParams& params)
    : State{params, model},
      nemotron_model_{model} {
  cache_config_ = model.cache_config_;

  auto& allocator = model_.allocator_cpu_;
  auto& device = *model_.p_device_;
  encoder_cache_.Initialize(cache_config_, model_.session_info_, allocator, device);
  decoder_state_.Initialize(cache_config_, model_.session_info_, allocator, device);

  // Pre-allocate run options per session
  encoder_run_options_ = OrtRunOptions::Create();
  decoder_run_options_ = OrtRunOptions::Create();
  joiner_run_options_ = OrtRunOptions::Create();

  if (model.config_->model.encoder.run_options.has_value()) {
    for (auto& entry : model.config_->model.encoder.run_options.value()) {
      encoder_run_options_->AddConfigEntry(entry.first.c_str(), entry.second.c_str());
    }
  }
  if (model.config_->model.decoder.run_options.has_value()) {
    for (auto& entry : model.config_->model.decoder.run_options.value()) {
      decoder_run_options_->AddConfigEntry(entry.first.c_str(), entry.second.c_str());
    }
  }
  if (model.config_->model.joiner.run_options.has_value()) {
    for (auto& entry : model.config_->model.joiner.run_options.value()) {
      joiner_run_options_->AddConfigEntry(entry.first.c_str(), entry.second.c_str());
    }
  }

  auto enc_out_type = model_.session_info_.GetOutputDataType(cache_config_.enc_out_encoded);
  auto frame_shape = std::array<int64_t, 3>{1, 1, cache_config_.hidden_dim};
  encoder_frame_ = OrtValue::CreateTensor(allocator, frame_shape, enc_out_type);

  auto targets_type = model_.session_info_.GetInputDataType(cache_config_.dec_in_targets);
  auto targets_shape = std::array<int64_t, 2>{1, 1};
  targets_ = OrtValue::CreateTensor(allocator, targets_shape, targets_type);

  auto tgt_len_type = model_.session_info_.GetInputDataType(cache_config_.dec_in_target_length);
  auto tgt_len_shape = std::array<int64_t, 1>{1};
  target_length_ = OrtValue::CreateTensor(allocator, tgt_len_shape, tgt_len_type);
  *target_length_->GetTensorMutableData<int64_t>() = 1;
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
    if (input.name == "audio_features" || input.name == cache_config_.enc_in_audio) {
      current_mel_ = input.tensor;
      need_encoder_run_ = true;
      chunk_done_ = false;
    }
  }
}

void NemotronSpeechState::ResetStreamingState() {
  auto& allocator = model_.allocator_cpu_;
  auto& device = *model_.p_device_;
  encoder_cache_.Reset(cache_config_, model_.session_info_, allocator, device);
  decoder_state_.Reset(cache_config_, model_.session_info_, allocator, device);
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

  auto& allocator = model_.allocator_cpu_;
  auto len_shape = std::array<int64_t, 1>{1};
  auto len_type = model_.session_info_.GetInputDataType(cache_config_.enc_in_length);
  auto signal_length = OrtValue::CreateTensor(allocator, len_shape, len_type);
  *signal_length->GetTensorMutableData<int64_t>() = total_mel_frames;

  const char* enc_input_names[] = {
      cache_config_.enc_in_audio.c_str(), cache_config_.enc_in_length.c_str(),
      cache_config_.enc_in_cache_channel.c_str(), cache_config_.enc_in_cache_time.c_str(),
      cache_config_.enc_in_cache_channel_len.c_str()};
  OrtValue* enc_inputs[] = {
      mel_tensor, signal_length.get(),
      encoder_cache_.cache_last_channel.get(),
      encoder_cache_.cache_last_time.get(),
      encoder_cache_.cache_last_channel_len.get()};
  const char* enc_output_names[] = {
      cache_config_.enc_out_encoded.c_str(), cache_config_.enc_out_length.c_str(),
      cache_config_.enc_out_cache_channel.c_str(), cache_config_.enc_out_cache_time.c_str(),
      cache_config_.enc_out_cache_channel_len.c_str()};

  auto enc_outputs = nemotron_model_.session_encoder_->Run(
      encoder_run_options_.get(), enc_input_names, enc_inputs, 5, enc_output_names, 5);

  encoded_output_ = std::move(enc_outputs[0]);
  encoded_len_ = *enc_outputs[1]->GetTensorData<int64_t>();
  encoder_cache_.cache_last_channel = std::move(enc_outputs[2]);
  encoder_cache_.cache_last_time = std::move(enc_outputs[3]);
  encoder_cache_.cache_last_channel_len = std::move(enc_outputs[4]);
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

  // Run decoder
  while (time_step_ < time_steps) {
    // Copy current encoder frame
    auto src_frame = enc_span.subspan(static_cast<size_t>(time_step_) * frame_bytes, frame_bytes);
    frame_span.CopyFrom(src_frame);

    // Run prediction network
    *targets_->GetTensorMutableData<int64_t>() = decoder_state_.last_token;

    const char* dec_input_names[] = {
        cache_config_.dec_in_targets.c_str(), cache_config_.dec_in_target_length.c_str(),
        cache_config_.dec_in_lstm_hidden.c_str(), cache_config_.dec_in_lstm_cell.c_str()};
    OrtValue* dec_inputs[] = {
        targets_.get(), target_length_.get(),
        decoder_state_.lstm_hidden_state.get(), decoder_state_.lstm_cell_state.get()};
    const char* dec_output_names[] = {
        cache_config_.dec_out_outputs.c_str(), cache_config_.dec_out_prednet_lengths.c_str(),
        cache_config_.dec_out_lstm_hidden.c_str(), cache_config_.dec_out_lstm_cell.c_str()};

    auto dec_outputs = nemotron_model_.session_decoder_->Run(
        decoder_run_options_.get(), dec_input_names, dec_inputs, 4, dec_output_names, 4);

    // Reshape decoder output for joiner
    auto dec_out_shape = dec_outputs[0]->GetTensorTypeAndShapeInfo()->GetShape();
    auto decoder_frame_shape = std::array<int64_t, 3>{1, 1, dec_out_shape[1]};
    auto dec_out_type = model_.session_info_.GetOutputDataType(cache_config_.dec_out_outputs);
    auto decoder_frame = OrtValue::CreateTensor(allocator, decoder_frame_shape, dec_out_type);
    ByteWrapTensor(*model_.p_device_, *decoder_frame).CopyFrom(ByteWrapTensor(*model_.p_device_, *dec_outputs[0]));

    // Run joiner
    const char* join_input_names[] = {
        cache_config_.join_in_encoder.c_str(), cache_config_.join_in_decoder.c_str()};
    OrtValue* join_inputs[] = {encoder_frame_.get(), decoder_frame.get()};
    const char* join_output_names[] = {cache_config_.join_out_logits.c_str()};

    auto join_outputs = nemotron_model_.session_joiner_->Run(
        joiner_run_options_.get(), join_input_names, join_inputs, 2, join_output_names, 1);

    // Argmax
    const float* logits_data = join_outputs[0]->GetTensorData<float>();
    auto logits_shape = join_outputs[0]->GetTensorTypeAndShapeInfo()->GetShape();
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

    // Non-blank: emit token, update LSTM state
    decoder_state_.last_token = best_token;
    decoder_state_.lstm_hidden_state = std::move(dec_outputs[2]);
    decoder_state_.lstm_cell_state = std::move(dec_outputs[3]);

    symbol_step_++;
    if (symbol_step_ >= cache_config_.max_symbols_per_step) {
      time_step_++;
      symbol_step_ = 0;
    }

    last_tokens_.push_back(static_cast<int32_t>(best_token));
    return last_tokens_;
  }

  // Exhausted all time steps
  chunk_done_ = true;
  return last_tokens_;
}

}  // namespace Generators
