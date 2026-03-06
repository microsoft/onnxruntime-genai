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
                                                        const GeneratorParams& /*params*/) const {
  throw std::runtime_error(
      "NemotronSpeechModel does not support the Generator pipeline. "
      "Use the StreamingASR API instead.");
}

}  // namespace Generators
