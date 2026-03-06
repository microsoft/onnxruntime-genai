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
  const auto& sp = config.model.speech;
  const auto& jo = config.model.joiner;

  // Encoder dimensions
  hidden_dim = enc.hidden_size;
  num_encoder_layers = enc.num_hidden_layers;

  // Decoder dimensions (LSTM)
  decoder_lstm_dim = dec.hidden_size;
  decoder_lstm_layers = dec.num_hidden_layers;

  // Speech / mel feature config
  num_mels = sp.num_mels;
  fft_size = sp.fft_size;
  hop_length = sp.hop_length;
  win_length = sp.win_length;
  preemph = sp.preemph;
  log_eps = sp.log_eps;
  subsampling_factor = sp.subsampling_factor;
  left_context = sp.left_context;
  conv_context = sp.conv_context;
  pre_encode_cache_size = sp.pre_encode_cache_size;
  sample_rate = sp.sample_rate;
  chunk_samples = sp.chunk_samples;
  blank_id = sp.blank_id;
  max_symbols_per_step = sp.max_symbols_per_step;

  // Vocab size from top-level config (includes blank)
  vocab_size = config.model.vocab_size - 1;

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

void NemotronEncoderCache::Initialize(const NemotronCacheConfig& cfg, OrtAllocator& allocator, DeviceInterface& device) {
  // cache_last_channel: [batch, num_layers, left_context, hidden_dim]
  auto ch_shape = std::array<int64_t, 4>{1, cfg.num_encoder_layers, cfg.left_context, cfg.hidden_dim};
  cache_last_channel = OrtValue::CreateTensor(allocator, ch_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  ByteWrapTensor(device, *cache_last_channel).Zero();

  // cache_last_time: [batch, num_layers, hidden_dim, conv_context]
  auto tm_shape = std::array<int64_t, 4>{1, cfg.num_encoder_layers, cfg.hidden_dim, cfg.conv_context};
  cache_last_time = OrtValue::CreateTensor(allocator, tm_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  ByteWrapTensor(device, *cache_last_time).Zero();

  // cache_last_channel_len: [1]
  auto len_shape = std::array<int64_t, 1>{1};
  cache_last_channel_len = OrtValue::CreateTensor(allocator, len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *cache_last_channel_len->GetTensorMutableData<int64_t>() = 0;
}

void NemotronEncoderCache::Reset(const NemotronCacheConfig& cfg, OrtAllocator& allocator, DeviceInterface& device) {
  Initialize(cfg, allocator, device);
}

void NemotronDecoderState::Initialize(const NemotronCacheConfig& cfg, OrtAllocator& allocator, DeviceInterface& device) {
  // LSTM states: [lstm_layers, 1, lstm_dim]
  auto state_shape = std::array<int64_t, 3>{cfg.decoder_lstm_layers, 1, cfg.decoder_lstm_dim};
  lstm_hidden_state = OrtValue::CreateTensor(allocator, state_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  ByteWrapTensor(device, *lstm_hidden_state).Zero();

  lstm_cell_state = OrtValue::CreateTensor(allocator, state_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  ByteWrapTensor(device, *lstm_cell_state).Zero();

  last_token = cfg.blank_id;  // Start with blank/SOS token
}

void NemotronDecoderState::Reset(const NemotronCacheConfig& cfg, OrtAllocator& allocator, DeviceInterface& device) {
  Initialize(cfg, allocator, device);
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

// ===========================================================================
// NemotronSpeechState implementation
// ===========================================================================

NemotronSpeechState::NemotronSpeechState(const NemotronSpeechModel& model,
                                         const GeneratorParams& params)
    : State{params, model},
      nemotron_model_{model} {
  cache_config_ = model.cache_config_;

  // Initialize streaming state
  auto& allocator = model_.allocator_cpu_;
  auto& device = *model_.p_device_;
  encoder_cache_.Initialize(cache_config_, allocator, device);
  decoder_state_.Initialize(cache_config_, allocator, device);
}

NemotronSpeechState::~NemotronSpeechState() = default;

DeviceSpan<float> NemotronSpeechState::Run(int /*total_length*/,
                                           DeviceSpan<int32_t>& /*next_tokens*/,
                                           DeviceSpan<int32_t> /*next_indices*/) {
  throw std::runtime_error(
      "NemotronSpeechState::Run() is not used directly. "
      "Use Generator::GenerateNextTokens() which calls ProcessChunk() internally.");
}

void NemotronSpeechState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  for (const auto& input : extra_inputs) {
    // Accept both the nominal name "audio_features" and the model's actual graph input name
    if (input.name == "audio_features" || input.name == cache_config_.enc_in_audio) {
      current_mel_ = input.tensor;
    }
  }
}

void NemotronSpeechState::LoadVocab() {
  if (vocab_loaded_) return;

  auto tokenizer = model_.CreateTokenizer();
  vocab_.resize(cache_config_.vocab_size);
  for (int i = 0; i < cache_config_.vocab_size; ++i) {
    try {
      std::vector<int32_t> ids = {static_cast<int32_t>(i)};
      vocab_[i] = tokenizer->Decode(ids);
    } catch (...) {
      vocab_[i] = "";
    }
  }

  // Pre-process sentencepiece space markers
  for (auto& tok : vocab_) {
    size_t pos = 0;
    while ((pos = tok.find("\xe2\x96\x81", pos)) != std::string::npos) {
      tok.replace(pos, 3, " ");
      pos += 1;
    }
  }
  vocab_loaded_ = true;
}

void NemotronSpeechState::ResetStreamingState() {
  auto& allocator = model_.allocator_cpu_;
  auto& device = *model_.p_device_;
  encoder_cache_.Reset(cache_config_, allocator, device);
  decoder_state_.Reset(cache_config_, allocator, device);
  full_transcript_.clear();
  current_mel_.reset();
}

std::string NemotronSpeechState::ProcessChunk() {
  LoadVocab();

  if (!current_mel_ || !current_mel_->ort_tensor_) {
    throw std::runtime_error("No mel input set. Call generator.set_model_input(\"audio_features\", mel) before GenerateNextTokens().");
  }

  OrtValue* mel_tensor = current_mel_->ort_tensor_.get();
  auto mel_info = mel_tensor->GetTensorTypeAndShapeInfo();
  auto mel_shape = mel_info->GetShape();
  // mel_tensor shape: [1, total_frames, num_mels]
  int64_t total_mel_frames = mel_shape[1];

  auto& allocator = model_.allocator_cpu_;

  // Create processed_signal_length: [1]
  auto len_shape = std::array<int64_t, 1>{1};
  auto len_type = model_.session_info_.GetInputDataType(cache_config_.enc_in_length);
  auto signal_length = OrtValue::CreateTensor(allocator, len_shape, len_type);
  *signal_length->GetTensorMutableData<int64_t>() = total_mel_frames;

  // Encoder inputs
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

  auto run_options = OrtRunOptions::Create();

  // Run encoder
  auto enc_outputs = nemotron_model_.session_encoder_->Run(
      run_options.get(),
      enc_input_names, enc_inputs, 5,
      enc_output_names, 5);

  // Parse encoder outputs
  auto* encoded = enc_outputs[0].get();
  int64_t encoded_len = *enc_outputs[1]->GetTensorData<int64_t>();

  // Update encoder cache
  encoder_cache_.cache_last_channel = std::move(enc_outputs[2]);
  encoder_cache_.cache_last_time = std::move(enc_outputs[3]);
  encoder_cache_.cache_last_channel_len = std::move(enc_outputs[4]);

  // Run RNNT decoder
  std::string chunk_text = RunRNNTDecoder(encoded, encoded_len);
  full_transcript_ += chunk_text;

  // Clear current mel after processing
  current_mel_.reset();

  return chunk_text;
}

std::string NemotronSpeechState::RunRNNTDecoder(OrtValue* encoder_output, int64_t encoded_len) {
  auto& allocator = model_.allocator_cpu_;
  std::string result;

  auto enc_info = encoder_output->GetTensorTypeAndShapeInfo();
  auto enc_shape = enc_info->GetShape();
  int64_t time_steps = std::min(enc_shape[1], encoded_len);
  int64_t hidden_dim = enc_shape[2];

  auto run_options = OrtRunOptions::Create();

  // Pre-allocate reusable tensors
  auto enc_out_type = model_.session_info_.GetOutputDataType(cache_config_.enc_out_encoded);
  auto frame_shape = std::array<int64_t, 3>{1, 1, hidden_dim};
  auto encoder_frame = OrtValue::CreateTensor(allocator, frame_shape, enc_out_type);

  auto enc_span = ByteWrapTensor(*model_.p_device_, *encoder_output);
  auto frame_span = ByteWrapTensor(*model_.p_device_, *encoder_frame);
  const size_t frame_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);

  auto targets_shape = std::array<int64_t, 2>{1, 1};
  auto targets_type = model_.session_info_.GetInputDataType(cache_config_.dec_in_targets);
  auto targets = OrtValue::CreateTensor(allocator, targets_shape, targets_type);
  int64_t* targets_data = targets->GetTensorMutableData<int64_t>();

  auto tgt_len_shape = std::array<int64_t, 1>{1};
  auto tgt_len_type = model_.session_info_.GetInputDataType(cache_config_.dec_in_target_length);
  auto target_length = OrtValue::CreateTensor(allocator, tgt_len_shape, tgt_len_type);
  *target_length->GetTensorMutableData<int64_t>() = 1;

  const int max_sym = cache_config_.max_symbols_per_step;

  for (int64_t t = 0; t < time_steps; ++t) {
    auto src_frame = enc_span.subspan(static_cast<size_t>(t) * frame_bytes, frame_bytes);
    frame_span.CopyFrom(src_frame);

    for (int sym = 0; sym < max_sym; ++sym) {
      *targets_data = decoder_state_.last_token;

      const char* dec_input_names[] = {
          cache_config_.dec_in_targets.c_str(), cache_config_.dec_in_target_length.c_str(),
          cache_config_.dec_in_lstm_hidden.c_str(), cache_config_.dec_in_lstm_cell.c_str()};
      OrtValue* dec_inputs[] = {
          targets.get(), target_length.get(),
          decoder_state_.lstm_hidden_state.get(), decoder_state_.lstm_cell_state.get()};

      const char* dec_output_names[] = {
          cache_config_.dec_out_outputs.c_str(), cache_config_.dec_out_prednet_lengths.c_str(),
          cache_config_.dec_out_lstm_hidden.c_str(), cache_config_.dec_out_lstm_cell.c_str()};

      auto dec_outputs = nemotron_model_.session_decoder_->Run(
          run_options.get(),
          dec_input_names, dec_inputs, 4,
          dec_output_names, 4);

      auto dec_out_shape = dec_outputs[0]->GetTensorTypeAndShapeInfo()->GetShape();
      auto decoder_frame_shape = std::array<int64_t, 3>{1, 1, dec_out_shape[1]};
      auto dec_out_type = model_.session_info_.GetOutputDataType(cache_config_.dec_out_outputs);
      auto decoder_frame = OrtValue::CreateTensor(allocator, decoder_frame_shape, dec_out_type);
      auto source_span = ByteWrapTensor(*model_.p_device_, *dec_outputs[0]);
      auto destination_span = ByteWrapTensor(*model_.p_device_, *decoder_frame);
      destination_span.CopyFrom(source_span);

      // Run joiner
      const char* join_input_names[] = {
          cache_config_.join_in_encoder.c_str(), cache_config_.join_in_decoder.c_str()};
      OrtValue* join_inputs[] = {
          encoder_frame.get(), decoder_frame.get()};

      const char* join_output_names[] = {cache_config_.join_out_logits.c_str()};

      auto join_outputs = nemotron_model_.session_joiner_->Run(
          run_options.get(),
          join_input_names, join_inputs, 2,
          join_output_names, 1);

      // Find argmax
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

      // Blank means current frame is done
      if (best_token == cache_config_.blank_id || best_token >= cache_config_.vocab_size) {
        break;
      }

      // Emit token & update state
      decoder_state_.last_token = best_token;
      decoder_state_.lstm_hidden_state = std::move(dec_outputs[2]);
      decoder_state_.lstm_cell_state = std::move(dec_outputs[3]);

      result += vocab_[best_token];
    }
  }

  return result;
}

}  // namespace Generators
