// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Parakeet TDT speech recognition model — Whisper-style integration.
//
// The chunked encoder + TDT decoder pipeline is preserved verbatim from the
// original ParakeetTdtStreamingASR implementation; the only difference is that
// it is now driven by State::SetExtraInputs / State::Run instead of an
// external StreamingASR object, so it can be used through the standard
// Generator / MultiModalProcessor public API.
//
// All mel-spectrogram extraction and per-feature normalization are delegated
// to onnxruntime-extensions (nemo_mel::NemoStreamingMelExtractor and
// ort_extensions::PerFeatureNormalize). No mel/FFT/normalization math is
// implemented here.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "../generators.h"
#include "parakeet.h"

namespace Generators {

// ─── ParakeetTdtConfig ─────────────────────────────────────────────────────────

void ParakeetTdtConfig::PopulateFromConfig(const Config& config) {
  const auto& enc = config.model.encoder;
  const auto& dec = config.model.decoder;
  const auto& m = config.model;
  const auto& jo = config.model.joiner;

  hidden_dim = enc.hidden_size;
  num_encoder_layers = enc.num_hidden_layers;

  decoder_lstm_dim = dec.hidden_size;
  decoder_lstm_layers = dec.num_hidden_layers;

  subsampling_factor = m.subsampling_factor;
  sample_rate = m.sample_rate;
  chunk_samples = m.chunk_samples;
  blank_id = m.blank_id;
  max_symbols_per_step = m.max_symbols_per_step;
  left_context_samples = m.left_context_samples;
  right_context_samples = m.right_context_samples;

  tdt_durations = m.tdt_durations;

  vocab_size = m.vocab_size;

  enc_in_audio = enc.inputs.audio_features;
  enc_out_encoded = enc.outputs.encoder_outputs;
  enc_in_length = enc.inputs.input_lengths;
  enc_out_length = enc.outputs.output_lengths;

  join_in_encoder = jo.inputs.encoder_outputs;
  join_in_decoder = jo.inputs.decoder_outputs;
  join_out_logits = jo.outputs.logits;

  dec_in_targets = dec.inputs.targets;
  dec_in_targets_length = dec.inputs.targets_length;
  dec_in_lstm_hidden_state = dec.inputs.lstm_hidden_state;
  dec_in_lstm_cell_state = dec.inputs.lstm_cell_state;
  dec_out_outputs = dec.outputs.outputs;
  dec_out_outputs_length = dec.outputs.outputs_length;
  dec_out_lstm_hidden_state = dec.outputs.lstm_hidden_state;
  dec_out_lstm_cell_state = dec.outputs.lstm_cell_state;
}

// ─── ParakeetTdtModel ──────────────────────────────────────────────────────────

ParakeetTdtModel::ParakeetTdtModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  parakeet_config_.PopulateFromConfig(*config_);

  // The TDT joiner produces logits over [vocab_size + 1] tokens (last index is
  // the blank/eos). The genai_config.json typically reports vocab_size = 8192
  // (real tokens) and eos_token_id = 8192 (the blank). Bump the search-visible
  // vocab_size by one so the eos token is reachable through normal argmax.
  // Bump only the search-visible vocab_size so the eos/blank token id is
  // reachable through the standard search. parakeet_config_.vocab_size must
  // stay equal to the *real* number of non-blank tokens, otherwise the joiner
  // argmax loop below would read one slot past the token-logit region (into
  // the TDT duration logits) and could return token id == real_vocab, which
  // is out of range for the decoder embedding.
  if (parakeet_config_.blank_id >= config_->model.vocab_size) {
    config_->model.vocab_size = parakeet_config_.blank_id + 1;
  }

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

  std::string encoder_filename = config_->model.encoder.filename;
  if (encoder_filename.empty()) encoder_filename = "encoder.onnx";

  std::string decoder_filename = config_->model.decoder.filename;
  if (decoder_filename.empty()) decoder_filename = "decoder.onnx";

  std::string joiner_filename = config_->model.joiner.filename;
  if (joiner_filename.empty()) joiner_filename = "joint.onnx";

  session_encoder_ = CreateSession(ort_env, encoder_filename, encoder_session_options_.get());
  session_decoder_ = CreateSession(ort_env, decoder_filename, decoder_session_options_.get());
  session_joiner_ = CreateSession(ort_env, joiner_filename, joiner_session_options_.get());

  session_info_.Add(*session_encoder_);
  session_info_.Add(*session_decoder_);
  session_info_.Add(*session_joiner_);
}

std::unique_ptr<State> ParakeetTdtModel::CreateState(DeviceSpan<int32_t> /*sequence_lengths*/,
                                                     const GeneratorParams& params) const {
  return std::make_unique<ParakeetTdtState>(*this, params);
}

// ─── ParakeetTdtState ──────────────────────────────────────────────────────────

ParakeetTdtState::ParakeetTdtState(const ParakeetTdtModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      cfg_{model.parakeet_config_} {
  // Use the (possibly bumped) vocab_size so the eos token id is in range.
  logits_size_ = model_.config_->model.vocab_size;
  eos_token_id_ = static_cast<int32_t>(cfg_.blank_id);

  // Allocate the persistent logits buffer (CPU-resident, one-hot).
  logits_buffer_.assign(static_cast<size_t>(logits_size_), 0.0f);
}

void ParakeetTdtState::InitializeDecoderState() {
  auto& allocator = model_.allocator_cpu_;

  auto state_shape = std::array<int64_t, 3>{cfg_.decoder_lstm_layers, 1, cfg_.decoder_lstm_dim};
  dec_.state_h = OrtValue::CreateTensor(allocator, state_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memset(dec_.state_h->GetTensorMutableRawData(), 0,
              cfg_.decoder_lstm_layers * cfg_.decoder_lstm_dim * sizeof(float));
  dec_.state_c = OrtValue::CreateTensor(allocator, state_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memset(dec_.state_c->GetTensorMutableRawData(), 0,
              cfg_.decoder_lstm_layers * cfg_.decoder_lstm_dim * sizeof(float));

  // Prime the decoder with the blank token to obtain the initial decoder_output.
  StepDecoder(static_cast<int32_t>(cfg_.blank_id));
  dec_.last_token = cfg_.blank_id;
}

void ParakeetTdtState::StepDecoder(int32_t token_id) {
  auto& allocator = model_.allocator_cpu_;
  auto run_options = OrtRunOptions::Create();

  auto targets_type = model_.session_info_.GetInputDataType(cfg_.dec_in_targets);
  auto tgt_len_type = model_.session_info_.GetInputDataType(cfg_.dec_in_targets_length);

  auto targets_shape = std::array<int64_t, 2>{1, 1};
  auto targets = OrtValue::CreateTensor(allocator, targets_shape, targets_type);
  if (targets_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    *targets->GetTensorMutableData<int64_t>() = token_id;
  } else {
    *targets->GetTensorMutableData<int32_t>() = token_id;
  }

  auto tgt_len_shape = std::array<int64_t, 1>{1};
  auto targets_length = OrtValue::CreateTensor(allocator, tgt_len_shape, tgt_len_type);
  if (tgt_len_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    *targets_length->GetTensorMutableData<int64_t>() = 1;
  } else {
    *targets_length->GetTensorMutableData<int32_t>() = 1;
  }

  const char* dec_input_names[] = {
      cfg_.dec_in_targets.c_str(), cfg_.dec_in_targets_length.c_str(),
      cfg_.dec_in_lstm_hidden_state.c_str(), cfg_.dec_in_lstm_cell_state.c_str()};
  OrtValue* dec_inputs[] = {targets.get(), targets_length.get(),
                            dec_.state_h.get(), dec_.state_c.get()};

  const char* dec_output_names[] = {
      cfg_.dec_out_outputs.c_str(), cfg_.dec_out_outputs_length.c_str(),
      cfg_.dec_out_lstm_hidden_state.c_str(), cfg_.dec_out_lstm_cell_state.c_str()};

  auto dec_outputs = model_.session_decoder_->Run(
      run_options.get(),
      dec_input_names, dec_inputs, 4,
      dec_output_names, 4);

  // Decoder is run with targets_length=1, so its output already has shape
  // [1, dec_dim, 1] — exactly what the joiner expects. Just take ownership.
  dec_.decoder_output = std::move(dec_outputs[0]);
  dec_.state_h = std::move(dec_outputs[2]);
  dec_.state_c = std::move(dec_outputs[3]);
  dec_.last_token = token_id;
}

void ParakeetTdtState::TranscribeAll() {
  if (total_mel_frames_ <= 0) return;

  InitializeDecoderState();

  // The processor has already produced `full_mel_` (shape
  // [num_mels, total_mel_frames_], globally normalized). We just walk it in
  // chunks defined by `cfg_.chunk_samples` (audio-sample space) to keep the
  // existing chunk_samples / left_context_samples / right_context_samples
  // configuration semantics.
  const auto& m = model_.config_->model;
  const int hop = m.hop_length;
  const size_t total_audio = static_cast<size_t>(total_mel_frames_) * hop;
  const size_t chunk_sz = static_cast<size_t>(cfg_.chunk_samples);

  size_t processed = 0;
  while (processed < total_audio) {
    size_t chunk_end = std::min(processed + chunk_sz, total_audio);
    bool is_last = (chunk_end >= total_audio);
    ProcessChunk(total_audio, processed, chunk_end, is_last);
    processed = chunk_end;
  }
}

void ParakeetTdtState::ProcessChunk(size_t total_audio,
                                    size_t chunk_start, size_t chunk_end, bool is_last) {
  auto& allocator = model_.allocator_cpu_;

  const auto& m = model_.config_->model;
  const int hop = m.hop_length;
  const int sub = cfg_.subsampling_factor;
  const int num_mels = m.num_mels;

  const size_t left_samples = static_cast<size_t>(cfg_.left_context_samples);
  const size_t right_samples = static_cast<size_t>(cfg_.right_context_samples);

  // Window in audio-sample space (preserved so that the mel-frame slice
  // matches what we'd get if the encoder were fed the same audio range).
  size_t win_left = (chunk_start > left_samples) ? (chunk_start - left_samples) : 0;
  size_t win_right = std::min(chunk_end + right_samples, total_audio);

  // Convert window + chunk boundaries from sample-space to mel-frame space.
  const int64_t win_left_mel = static_cast<int64_t>(win_left / hop);
  int64_t win_right_mel = static_cast<int64_t>(win_right / hop);
  if (win_right_mel > total_mel_frames_) win_right_mel = total_mel_frames_;
  int64_t num_mel_frames = win_right_mel - win_left_mel;
  if (num_mel_frames <= 0) return;

  // Slice the cached, globally-normalized mel ([num_mels, T_full]) into a
  // contiguous [1, num_mels, num_mel_frames] tensor for the encoder.
  auto signal_shape = std::array<int64_t, 3>{1, num_mels, num_mel_frames};
  auto& allocator_ref = allocator;
  auto processed_signal = OrtValue::CreateTensor(allocator_ref, signal_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  float* dst = processed_signal->GetTensorMutableData<float>();
  for (int b = 0; b < num_mels; ++b) {
    const float* src = full_mel_.data() + static_cast<size_t>(b) * total_mel_frames_ + win_left_mel;
    std::memcpy(dst + static_cast<size_t>(b) * num_mel_frames, src,
                static_cast<size_t>(num_mel_frames) * sizeof(float));
  }

  auto len_shape = std::array<int64_t, 1>{1};
  auto signal_length = OrtValue::CreateTensor(allocator_ref, len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *signal_length->GetTensorMutableData<int64_t>() = num_mel_frames;

  const char* enc_input_names[] = {cfg_.enc_in_audio.c_str(), cfg_.enc_in_length.c_str()};
  OrtValue* enc_inputs[] = {processed_signal.get(), signal_length.get()};
  const char* enc_output_names[] = {cfg_.enc_out_encoded.c_str(), cfg_.enc_out_length.c_str()};

  auto run_options = OrtRunOptions::Create();
  auto enc_outputs = model_.session_encoder_->Run(
      run_options.get(),
      enc_input_names, enc_inputs, 2,
      enc_output_names, 2);

  auto* encoded = enc_outputs[0].get();
  int64_t enc_total = *enc_outputs[1]->GetTensorData<int64_t>();

  // Map chunk start/end (sample-space) to encoder-frame indices within this
  // window.  encoder_frame_index = mel_frame_index / subsampling_factor.
  int64_t chunk_start_mel = static_cast<int64_t>(chunk_start / hop) - win_left_mel;
  int64_t chunk_end_mel = static_cast<int64_t>(chunk_end / hop) - win_left_mel;
  int64_t decode_start = chunk_start_mel / sub;
  int64_t decode_end;
  if (is_last) {
    decode_end = enc_total;
  } else {
    decode_end = std::min(chunk_end_mel / sub, enc_total);
  }
  if (decode_start < 0) decode_start = 0;
  if (decode_end > enc_total) decode_end = enc_total;
  if (decode_end <= decode_start) return;

  RunTDTDecoder(encoded, decode_start, decode_end);
}

void ParakeetTdtState::RunTDTDecoder(OrtValue* encoder_output,
                                     int64_t start_frame,
                                     int64_t end_frame) {
  auto& allocator = model_.allocator_cpu_;
  auto run_options = OrtRunOptions::Create();

  auto enc_info = encoder_output->GetTensorTypeAndShapeInfo();
  auto enc_shape = enc_info->GetShape();
  // Encoder output: [1, hidden_dim, T']
  const int64_t hidden_dim = enc_shape[1];
  const int64_t enc_time = enc_shape[2];
  const float* enc_data = encoder_output->GetTensorData<float>();

  const int num_durations = static_cast<int>(cfg_.tdt_durations.size());
  const int vocab_size = cfg_.vocab_size;
  const int blank_id = cfg_.blank_id;
  const int max_sym = cfg_.max_symbols_per_step;

  int symbols_this_frame = 0;
  int64_t t = start_frame;

  while (t < end_frame) {
    // Joiner expects feature-last: encoder_output [1, 1, hidden_dim],
    // decoder_output [1, 1, dec_dim].
    auto frame_shape = std::array<int64_t, 3>{1, 1, hidden_dim};
    auto encoder_frame = OrtValue::CreateTensor(allocator, frame_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    float* frame_data = encoder_frame->GetTensorMutableData<float>();
    for (int64_t d = 0; d < hidden_dim; ++d) {
      frame_data[d] = enc_data[d * enc_time + t];
    }

    // Decoder output is [1, dec_dim, 1]; reshape to [1, 1, dec_dim] for the
    // joiner (same memory layout, only the shape descriptor changes).
    const float* dec_data = dec_.decoder_output->GetTensorData<float>();
    int64_t dec_dim = cfg_.decoder_lstm_dim;
    auto dec_shape = std::array<int64_t, 3>{1, 1, dec_dim};
    auto decoder_frame = OrtValue::CreateTensor(allocator, dec_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::memcpy(decoder_frame->GetTensorMutableData<float>(), dec_data,
                static_cast<size_t>(dec_dim) * sizeof(float));

    const char* join_input_names[] = {cfg_.join_in_encoder.c_str(), cfg_.join_in_decoder.c_str()};
    OrtValue* join_inputs[] = {encoder_frame.get(), decoder_frame.get()};
    const char* join_output_names[] = {cfg_.join_out_logits.c_str()};

    auto join_outputs = model_.session_joiner_->Run(
        run_options.get(),
        join_input_names, join_inputs, 2,
        join_output_names, 1);

    const float* logits_data = join_outputs[0]->GetTensorData<float>();

    // Token argmax over [0..vocab_size] (inclusive of blank at index vocab_size).
    const int num_tok_logits = vocab_size + 1;
    int best_token = 0;
    float best_score = logits_data[0];
    for (int i = 1; i < num_tok_logits; ++i) {
      if (logits_data[i] > best_score) {
        best_score = logits_data[i];
        best_token = i;
      }
    }

    // Duration argmax. When the predicted token is the blank, forbid
    // duration index 0 (zero-duration blank) — that combination would emit
    // nothing and stall on the same frame, which is the known v3 "sentence
    // dropping" failure mode (sherpa-onnx reports the same fix).
    int dur_idx = 0;
    if (num_durations > 0) {
      const int dur_off = num_tok_logits;
      const int start_i = (best_token == blank_id) ? 1 : 0;
      float best_dur_score = -std::numeric_limits<float>::infinity();
      for (int i = start_i; i < num_durations; ++i) {
        if (logits_data[dur_off + i] > best_dur_score) {
          best_dur_score = logits_data[dur_off + i];
          dur_idx = i;
        }
      }
    }
    int skip = (dur_idx < static_cast<int>(cfg_.tdt_durations.size()))
                   ? cfg_.tdt_durations[dur_idx]
                   : dur_idx;

    if (best_token != blank_id) {
      symbols_this_frame++;
      decoded_tokens_.push_back(static_cast<int32_t>(best_token));
      StepDecoder(static_cast<int32_t>(best_token));
    }

    if (skip > 0) symbols_this_frame = 0;
    if (symbols_this_frame >= max_sym) {
      symbols_this_frame = 0;
      skip = 1;
    }

    t += skip;
  }
}

void ParakeetTdtState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  if (decoded_) return;

  // Locate the pre-computed mel-features tensor produced by ParakeetTdtProcessor.
  // Expected shape: [1, num_mels, total_frames], already globally normalized.
  const Tensor* mel_tensor = nullptr;
  for (const auto& ei : extra_inputs) {
    if (ei.name == "mel_features") {
      mel_tensor = ei.tensor.get();
      break;
    }
  }
  if (!mel_tensor) {
    throw std::runtime_error("ParakeetTdtState::SetExtraInputs: 'mel_features' input is missing.");
  }

  auto info = mel_tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
  auto shape = info->GetShape();
  if (shape.size() != 3 || shape[0] != 1) {
    throw std::runtime_error("ParakeetTdtState::SetExtraInputs: mel_features must have shape [1, num_mels, T].");
  }
  const int num_mels = static_cast<int>(shape[1]);
  total_mel_frames_ = static_cast<int>(shape[2]);
  if (num_mels != model_.config_->model.num_mels) {
    throw std::runtime_error("ParakeetTdtState::SetExtraInputs: mel num_mels mismatch with model config.");
  }

  const float* mel_src = mel_tensor->ort_tensor_->GetTensorData<float>();
  full_mel_.assign(mel_src,
                   mel_src + static_cast<size_t>(num_mels) * total_mel_frames_);

  TranscribeAll();
  decoded_ = true;
}

DeviceSpan<float> ParakeetTdtState::Run(int total_length,
                                        DeviceSpan<int32_t>& /*next_tokens*/,
                                        DeviceSpan<int32_t> /*next_indices*/) {
  // total_length = current sequence length AFTER the just-appended tokens.
  // The processor inserts a single placeholder token at index 0 (decoder_start),
  // so the next emitted token corresponds to index (total_length - 1).
  size_t emit_index = (total_length > 0) ? static_cast<size_t>(total_length - 1) : 0;

  // Reset buffer to all-zeros, place a high score at the desired token id.
  std::fill(logits_buffer_.begin(), logits_buffer_.end(), 0.0f);

  int32_t next_token;
  if (emit_index < decoded_tokens_.size()) {
    next_token = decoded_tokens_[emit_index];
  } else {
    next_token = eos_token_id_;
  }

  if (next_token >= 0 && next_token < logits_size_) {
    logits_buffer_[static_cast<size_t>(next_token)] = 100.0f;
  }

  // Wrap the CPU buffer into a DeviceSpan<float> the search can consume.
  auto* cpu_device = GetDeviceInterface(DeviceType::CPU);
  logits_device_ = cpu_device->WrapMemory(std::span<float>{logits_buffer_.data(), logits_buffer_.size()});
  return logits_device_;
}

}  // namespace Generators
