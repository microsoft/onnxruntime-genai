// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Parakeet TDT speech recognition model — Whisper-style integration.
//
// The chunked encoder + TDT decoder pipeline is preserved verbatim from the
// original ParakeetStreamingASR implementation; the only difference is that
// it is now driven by State::SetExtraInputs / State::Run instead of an
// external StreamingASR object, so it can be used through the standard
// Generator / MultiModalProcessor public API.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "../generators.h"
#include "../parakeet_mel.h"
#include "parakeet.h"

namespace Generators {

// ─── ParakeetConfig ─────────────────────────────────────────────────────────

void ParakeetConfig::PopulateFromConfig(const Config& config) {
  const auto& enc = config.model.encoder;
  const auto& dec = config.model.decoder;
  const auto& m = config.model;
  const auto& jo = config.model.joiner;

  hidden_dim = enc.hidden_size;
  num_encoder_layers = enc.num_hidden_layers;

  decoder_lstm_dim = dec.hidden_size;
  decoder_lstm_layers = dec.num_hidden_layers;

  num_mels = m.num_mels;
  fft_size = m.fft_size;
  hop_length = m.hop_length;
  win_length = m.win_length;
  preemph = m.preemph;
  log_eps = m.log_eps;
  subsampling_factor = m.subsampling_factor;
  sample_rate = m.sample_rate;
  chunk_samples = m.chunk_samples;
  blank_id = m.blank_id;
  max_symbols_per_step = m.max_symbols_per_step;
  left_context_samples = m.left_context_samples;
  right_context_samples = m.right_context_samples;

  tdt_durations = m.tdt_durations;
  tdt_num_extra_outputs = m.tdt_num_extra_outputs;

  vocab_size = m.vocab_size;

  enc_in_audio = enc.inputs.audio_features;
  enc_out_encoded = enc.outputs.encoder_outputs;
  enc_in_length = m.enc_in_length;
  enc_out_length = m.enc_out_length;

  join_in_encoder = jo.inputs.encoder_outputs;
  join_in_decoder = jo.inputs.decoder_outputs;
  join_out_logits = jo.outputs.logits;

  dec_in_targets = dec.inputs.targets;
  dec_in_target_length = dec.inputs.target_length;
  dec_in_states_1 = dec.inputs.states_1;
  dec_in_states_2 = dec.inputs.states_2;
  dec_out_outputs = dec.outputs.outputs;
  dec_out_prednet_lengths = dec.outputs.prednet_lengths;
  dec_out_states_1 = dec.outputs.states_1;
  dec_out_states_2 = dec.outputs.states_2;
}

// ─── ParakeetModel ──────────────────────────────────────────────────────────

ParakeetModel::ParakeetModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
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
}

std::unique_ptr<State> ParakeetModel::CreateState(DeviceSpan<int32_t> /*sequence_lengths*/,
                                                   const GeneratorParams& params) const {
  return std::make_unique<ParakeetState>(*this, params);
}

// ─── ParakeetState ──────────────────────────────────────────────────────────

ParakeetState::ParakeetState(const ParakeetModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      cfg_{model.parakeet_config_} {
  // Use the (possibly bumped) vocab_size so the eos token id is in range.
  logits_size_ = model_.config_->model.vocab_size;
  eos_token_id_ = static_cast<int32_t>(cfg_.blank_id);

  // Allocate the persistent logits buffer (CPU-resident, one-hot).
  logits_buffer_.assign(static_cast<size_t>(logits_size_), 0.0f);
}

void ParakeetState::InitializeDecoderState() {
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

void ParakeetState::StepDecoder(int32_t token_id) {
  auto& allocator = model_.allocator_cpu_;
  auto run_options = OrtRunOptions::Create();

  auto targets_shape = std::array<int64_t, 2>{1, 1};
  auto targets = OrtValue::CreateTensor(allocator, targets_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  *targets->GetTensorMutableData<int32_t>() = token_id;

  auto tgt_len_shape = std::array<int64_t, 1>{1};
  auto target_length = OrtValue::CreateTensor(allocator, tgt_len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  *target_length->GetTensorMutableData<int32_t>() = 1;

  const char* dec_input_names[] = {
      cfg_.dec_in_targets.c_str(), cfg_.dec_in_target_length.c_str(),
      cfg_.dec_in_states_1.c_str(), cfg_.dec_in_states_2.c_str()};
  OrtValue* dec_inputs[] = {targets.get(), target_length.get(),
                            dec_.state_h.get(), dec_.state_c.get()};

  const char* dec_output_names[] = {
      cfg_.dec_out_outputs.c_str(), cfg_.dec_out_prednet_lengths.c_str(),
      cfg_.dec_out_states_1.c_str(), cfg_.dec_out_states_2.c_str()};

  auto dec_outputs = model_.session_decoder_->Run(
      run_options.get(),
      dec_input_names, dec_inputs, 4,
      dec_output_names, 4);

  auto dec_out_shape = dec_outputs[0]->GetTensorTypeAndShapeInfo()->GetShape();
  int64_t dec_dim = dec_out_shape[1];
  int64_t dec_time = dec_out_shape[2];

  auto frame_shape = std::array<int64_t, 3>{1, dec_dim, 1};
  dec_.decoder_output = OrtValue::CreateTensor(allocator, frame_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  const float* src = dec_outputs[0]->GetTensorData<float>();
  float* dst = dec_.decoder_output->GetTensorMutableData<float>();
  int64_t t_last = dec_time - 1;
  for (int64_t d = 0; d < dec_dim; ++d) {
    dst[d] = src[d * dec_time + t_last];
  }

  dec_.state_h = std::move(dec_outputs[2]);
  dec_.state_c = std::move(dec_outputs[3]);
  dec_.last_token = token_id;
}

void ParakeetState::TranscribeAll(const float* audio, size_t num_samples) {
  if (num_samples == 0) return;

  InitializeDecoderState();

  const size_t chunk_sz = static_cast<size_t>(cfg_.chunk_samples);
  size_t processed = 0;

  // Stream-style chunking (preserve original behaviour): full chunks plus
  // right context first, then a final partial chunk at the tail.
  while (processed + chunk_sz <= num_samples) {
    size_t chunk_end = processed + chunk_sz;
    bool is_last = (chunk_end + chunk_sz > num_samples);
    ProcessChunk(audio, num_samples, processed, chunk_end, is_last);
    processed = chunk_end;
  }

  if (processed < num_samples) {
    ProcessChunk(audio, num_samples, processed, num_samples, /*is_last=*/true);
  }
}

void ParakeetState::ProcessChunk(const float* audio, size_t total_audio,
                                  size_t chunk_start, size_t chunk_end, bool is_last) {
  auto& allocator = model_.allocator_cpu_;

  const int hop = cfg_.hop_length;
  const int sub = cfg_.subsampling_factor;
  const int encoder_frame_samples = hop * sub;

  const size_t left_samples = static_cast<size_t>(cfg_.left_context_samples);
  const size_t right_samples = static_cast<size_t>(cfg_.right_context_samples);
  const size_t chunk_samples_aligned = static_cast<size_t>(
      (static_cast<int>(chunk_end - chunk_start) / encoder_frame_samples) * encoder_frame_samples);

  size_t win_left = (chunk_start > left_samples) ? (chunk_start - left_samples) : 0;
  size_t win_right = std::min(chunk_end + right_samples, total_audio);

  if (is_last) {
    size_t target_buf_size = left_samples + chunk_samples_aligned + right_samples;
    size_t actual_size = win_right - win_left;
    if (actual_size < target_buf_size) {
      win_left = (win_right > target_buf_size) ? (win_right - target_buf_size) : 0;
    }
  }

  const float* window_audio = audio + win_left;
  size_t window_len = win_right - win_left;

  parakeet_mel::ParakeetMelConfig mel_cfg;
  mel_cfg.num_mels = cfg_.num_mels;
  mel_cfg.fft_size = cfg_.fft_size;
  mel_cfg.hop_length = cfg_.hop_length;
  mel_cfg.win_length = cfg_.win_length;
  mel_cfg.sample_rate = cfg_.sample_rate;
  mel_cfg.preemph = cfg_.preemph;

  int num_mel_frames = 0;
  auto raw_mel = parakeet_mel::ComputeLogMel(window_audio, window_len, mel_cfg, num_mel_frames);
  if (num_mel_frames <= 0) return;

  // Per-feature normalization (Bessel's correction), matches NeMo normalize_batch.
  const int num_mels = cfg_.num_mels;
  std::vector<float> mel_normalized(static_cast<size_t>(num_mels) * num_mel_frames);
  for (int m = 0; m < num_mels; ++m) {
    const float* row = raw_mel.data() + static_cast<size_t>(m) * num_mel_frames;
    double sum = 0.0;
    for (int t = 0; t < num_mel_frames; ++t) sum += row[t];
    float mean = static_cast<float>(sum / num_mel_frames);
    double var_sum = 0.0;
    for (int t = 0; t < num_mel_frames; ++t) {
      double diff = row[t] - mean;
      var_sum += diff * diff;
    }
    float std_val = (num_mel_frames > 1)
                        ? std::sqrt(static_cast<float>(var_sum / (num_mel_frames - 1))) + 1e-5f
                        : 1e-5f;
    float* out_row = mel_normalized.data() + static_cast<size_t>(m) * num_mel_frames;
    for (int t = 0; t < num_mel_frames; ++t) {
      out_row[t] = (row[t] - mean) / std_val;
    }
  }

  auto signal_shape = std::array<int64_t, 3>{1, num_mels, num_mel_frames};
  auto processed_signal = OrtValue::CreateTensor(allocator, signal_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memcpy(processed_signal->GetTensorMutableData<float>(), mel_normalized.data(),
              static_cast<size_t>(num_mels) * num_mel_frames * sizeof(float));

  auto len_shape = std::array<int64_t, 1>{1};
  auto signal_length = OrtValue::CreateTensor(allocator, len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *signal_length->GetTensorMutableData<int64_t>() = static_cast<int64_t>(num_mel_frames);

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

  size_t left_ctx_samples = chunk_start - win_left;
  int64_t left_ctx_mel = static_cast<int64_t>(left_ctx_samples) / hop;
  int64_t left_enc = left_ctx_mel / sub;

  int64_t decode_start = left_enc;
  int64_t decode_end;
  if (is_last) {
    decode_end = enc_total;
  } else {
    int64_t chunk_actual = static_cast<int64_t>(chunk_end - chunk_start);
    int64_t chunk_mel = chunk_actual / hop;
    int64_t chunk_enc = chunk_mel / sub;
    decode_end = std::min(left_enc + chunk_enc, enc_total);
  }

  if (decode_end <= decode_start) return;

  RunTDTDecoder(encoded, decode_start, decode_end);
}

void ParakeetState::RunTDTDecoder(OrtValue* encoder_output,
                                   int64_t start_frame,
                                   int64_t end_frame) {
  auto& allocator = model_.allocator_cpu_;
  auto run_options = OrtRunOptions::Create();

  auto enc_info = encoder_output->GetTensorTypeAndShapeInfo();
  auto enc_shape = enc_info->GetShape();
  // [1, hidden_dim, T']
  int64_t hidden_dim = enc_shape[1];
  int64_t enc_time = enc_shape[2];
  const float* enc_data = encoder_output->GetTensorData<float>();

  const int num_durations = cfg_.tdt_num_extra_outputs;
  const int vocab_size = cfg_.vocab_size;
  const int blank_id = cfg_.blank_id;
  const int max_sym = cfg_.max_symbols_per_step;

  int symbols_this_frame = 0;
  int64_t t = start_frame;

  while (t < end_frame) {
    auto frame_shape = std::array<int64_t, 3>{1, hidden_dim, 1};
    auto encoder_frame = OrtValue::CreateTensor(allocator, frame_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    float* frame_data = encoder_frame->GetTensorMutableData<float>();
    for (int64_t d = 0; d < hidden_dim; ++d) {
      frame_data[d] = enc_data[d * enc_time + t];
    }

    const char* join_input_names[] = {cfg_.join_in_encoder.c_str(), cfg_.join_in_decoder.c_str()};
    OrtValue* join_inputs[] = {encoder_frame.get(), dec_.decoder_output.get()};
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

    int skip = 0;
    if (num_durations > 0) {
      const int dur_off = num_tok_logits;
      float best_dur_score = logits_data[dur_off];
      for (int i = 1; i < num_durations; ++i) {
        if (logits_data[dur_off + i] > best_dur_score) {
          best_dur_score = logits_data[dur_off + i];
          skip = i;
        }
      }
      if (skip < static_cast<int>(cfg_.tdt_durations.size())) {
        skip = cfg_.tdt_durations[skip];
      }
    }

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
    if (best_token == blank_id && skip == 0) {
      symbols_this_frame = 0;
      skip = 1;
    }

    t += skip;
  }
}

void ParakeetState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  if (decoded_) return;

  // Locate the raw PCM tensor produced by ParakeetProcessor.
  const Tensor* pcm_tensor = nullptr;
  for (const auto& ei : extra_inputs) {
    if (ei.name == "audio_pcm" || ei.name == cfg_.enc_in_audio) {
      pcm_tensor = ei.tensor.get();
      break;
    }
  }
  if (!pcm_tensor) {
    throw std::runtime_error("ParakeetState::SetExtraInputs: 'audio_pcm' input is missing.");
  }

  auto info = pcm_tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
  auto shape = info->GetShape();
  const float* pcm = pcm_tensor->ort_tensor_->GetTensorData<float>();
  size_t num_samples = info->GetElementCount();
  // Shape may be [1, num_samples] or [num_samples] — element_count is what matters.
  (void)shape;

  TranscribeAll(pcm, num_samples);
  decoded_ = true;
}

DeviceSpan<float> ParakeetState::Run(int total_length,
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
