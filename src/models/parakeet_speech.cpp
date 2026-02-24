// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Parakeet TDT Speech Model — non-cache-aware encoder + TDT decoder+joiner.

#include <cstring>
#include <stdexcept>
#include <vector>

#include "../generators.h"
#include "parakeet_speech.h"

namespace Generators {

// ─── ParakeetConfig ─────────────────────────────────────────────────────────

void ParakeetConfig::PopulateFromConfig(const Config& config) {
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
  sample_rate = sp.sample_rate;
  blank_id = sp.blank_id;
  max_symbols_per_step = sp.max_symbols_per_step;

  // Chunk samples (for streaming)
  if (sp.chunk_samples > 0) {
    chunk_samples = sp.chunk_samples;
  }

  // TDT parameters
  tdt_durations = sp.tdt_durations;
  tdt_num_extra_outputs = sp.tdt_num_extra_outputs;

  // Vocab size from top-level config (includes blank)
  vocab_size = config.model.vocab_size;

  // Encoder I/O names
  enc_in_audio = enc.inputs.audio_features;
  enc_in_length = sp.enc_in_length;
  enc_out_encoded = enc.outputs.encoder_outputs;
  enc_out_length = sp.enc_out_length;

  // Joiner I/O names
  join_in_encoder = jo.inputs.encoder_outputs;
  join_in_decoder = jo.inputs.decoder_outputs;
  join_out_logits = jo.outputs.logits;

  // Decoder I/O names (prediction network)
  dec_in_targets = dec.inputs.targets;
  dec_in_target_length = dec.inputs.target_length;
  dec_in_states_1 = dec.inputs.states_1;
  dec_in_states_2 = dec.inputs.states_2;
  dec_out_outputs = dec.outputs.outputs;
  dec_out_prednet_lengths = dec.outputs.prednet_lengths;
  dec_out_states_1 = dec.outputs.states_1;
  dec_out_states_2 = dec.outputs.states_2;
}

// ─── ParakeetDecoderState ───────────────────────────────────────────────────

void ParakeetDecoderState::Initialize(const ParakeetConfig& cfg, OrtAllocator& allocator) {
  // LSTM states: [lstm_layers, 1, lstm_dim]
  auto state_shape = std::array<int64_t, 3>{cfg.decoder_lstm_layers, 1, cfg.decoder_lstm_dim};
  state_h = OrtValue::CreateTensor(allocator, state_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memset(state_h->GetTensorMutableRawData(), 0,
              cfg.decoder_lstm_layers * 1 * cfg.decoder_lstm_dim * sizeof(float));

  state_c = OrtValue::CreateTensor(allocator, state_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memset(state_c->GetTensorMutableRawData(), 0,
              cfg.decoder_lstm_layers * 1 * cfg.decoder_lstm_dim * sizeof(float));

  last_token = cfg.blank_id;  // Start with blank token
}

void ParakeetDecoderState::Reset(const ParakeetConfig& cfg, OrtAllocator& allocator) {
  Initialize(cfg, allocator);
}

// ─── ParakeetSpeechModel ────────────────────────────────────────────────────

ParakeetSpeechModel::ParakeetSpeechModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  parakeet_config_ = ParakeetConfig{};
  parakeet_config_.PopulateFromConfig(*config_);

  // Create session options from config.
  // Each section uses its own session_options if present, otherwise falls back
  // to the decoder's session_options. EP selection is driven by genai_config.json.
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
}

std::unique_ptr<State> ParakeetSpeechModel::CreateState(DeviceSpan<int32_t> /*sequence_lengths*/,
                                                        const GeneratorParams& /*params*/) const {
  throw std::runtime_error(
      "ParakeetSpeechModel does not support the Generator pipeline. "
      "Use the StreamingASR API instead.");
}

}  // namespace Generators
