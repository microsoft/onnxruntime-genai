// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// ParakeetStreamingASR — matches sherpa-onnx's DecodeOneTDT exactly.
// Sliding window: encode last 8s of audio, TDT greedy decode,
// commit stable tokens (>2s old) with timestamp-based stitching.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

#include "generators.h"
#include "parakeet_streaming_asr.h"

namespace Generators {

// ─── Vocabulary loading ─────────────────────────────────────────────────────

void ParakeetStreamingASR::LoadVocab() {
  if (vocab_loaded_) return;

  auto vocab_path = model_.config_->config_path / "vocab.txt";
  std::ifstream vocab_file(vocab_path.string());
  if (vocab_file.is_open()) {
    std::string line;
    while (std::getline(vocab_file, line)) {
      vocab_.push_back(line);
    }
    while (static_cast<int>(vocab_.size()) < config_.vocab_size) {
      vocab_.push_back("");
    }
  } else {
    auto tokenizer = model_.CreateTokenizer();
    vocab_.resize(config_.vocab_size);
    for (int i = 0; i < config_.vocab_size; ++i) {
      try {
        std::vector<int32_t> ids = {static_cast<int32_t>(i)};
        vocab_[i] = tokenizer->Decode(ids);
      } catch (...) {
        vocab_[i] = "";
      }
    }
  }
  vocab_loaded_ = true;
}

// ─── Per-feature normalization ──────────────────────────────────────────────

void ParakeetStreamingASR::NormalizePerFeature(float* data, int num_mels, int num_frames) {
  // Matches sherpa's NemoNormalizePerFeature exactly:
  // For each mel bin: mean = mean(x), var = mean(x^2) - mean(x)^2, normalize
  for (int m = 0; m < num_mels; ++m) {
    float* row = data + m * num_frames;
    float sum = 0.0f, sq_sum = 0.0f;
    for (int t = 0; t < num_frames; ++t) {
      sum += row[t];
      sq_sum += row[t] * row[t];
    }
    float mean = sum / static_cast<float>(num_frames);
    float var = sq_sum / static_cast<float>(num_frames) - mean * mean;
    if (var < 0.0f) var = 0.0f;
    float inv_std = 1.0f / (std::sqrt(var) + 1e-5f);
    for (int t = 0; t < num_frames; ++t) {
      row[t] = (row[t] - mean) * inv_std;
    }
  }
}

// ─── Constructor / Reset ────────────────────────────────────────────────────

ParakeetStreamingASR::ParakeetStreamingASR(Model& model)
    : model_{model} {
  auto* parakeet_model = dynamic_cast<ParakeetSpeechModel*>(&model);
  if (!parakeet_model) {
    throw std::runtime_error("ParakeetStreamingASR requires a parakeet_tdt model type.");
  }

  encoder_session_ = parakeet_model->session_encoder_.get();
  decoder_session_ = parakeet_model->session_decoder_.get();
  joiner_session_ = parakeet_model->session_joiner_.get();
  config_ = parakeet_model->parakeet_config_;

  // Detect decoder integer input dtype from ONNX model metadata.
  // FP32 NeMo exports use int64 for targets/target_length; sherpa int8 models use int32.
  {
    auto type_info = decoder_session_->GetInputTypeInfo(0);  // first input = targets
    decoder_int_dtype_ = type_info->GetTensorTypeAndShapeInfo().GetElementType();
  }

  // Detect joiner input layout from ONNX metadata.
  // Channel-first [B, dim, T]: sherpa int8 models have dim[1] == hidden_dim (1024)
  // Channel-last  [B, T, dim]: FP32 NeMo exports have dim[1] == dynamic/-1
  {
    auto type_info = joiner_session_->GetInputTypeInfo(0);
    auto shape = type_info->GetTensorTypeAndShapeInfo().GetShape();
    joiner_channel_first_ = (shape.size() >= 3 && shape[1] == config_.hidden_dim);
  }
}

ParakeetStreamingASR::~ParakeetStreamingASR() = default;

void ParakeetStreamingASR::Reset() {
  full_transcript_.clear();
  audio_buffer_.clear();
  committed_tokens_.clear();
  total_audio_sec_ = 0.0f;
  chunk_index_ = 0;
}

// ─── TDT Greedy Decode (matches sherpa's DecodeOneTDT) ──────────────────────

std::vector<ParakeetStreamingASR::TimestampedToken>
ParakeetStreamingASR::EncodeAndDecodeTDT(
    const float* audio, size_t num_samples, float window_start_sec) {
  auto& allocator = model_.allocator_cpu_;
  const int num_mels = config_.num_mels;

  // Compute mel features using NeMo-compatible mel spectrogram
  parakeet_mel::MelConfig mel_cfg;
  mel_cfg.num_mels = num_mels;
  mel_cfg.fft_size = config_.fft_size;
  mel_cfg.hop_length = config_.hop_length;
  mel_cfg.win_length = config_.win_length;
  mel_cfg.sample_rate = config_.sample_rate;
  mel_cfg.preemph = config_.preemph;

  int mel_frames = 0;
  auto raw_mel = parakeet_mel::ComputeLogMel(audio, num_samples, mel_cfg, mel_frames);
  // raw_mel is [num_mels, mel_frames] row-major

  if (mel_frames == 0) return {};

  // Copy into ORT tensor [1, num_mels, mel_frames]
  auto signal_shape = std::array<int64_t, 3>{1, num_mels, mel_frames};
  auto processed_signal = OrtValue::CreateTensor(allocator, signal_shape,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  float* signal_data = processed_signal->GetTensorMutableData<float>();
  std::memcpy(signal_data, raw_mel.data(), num_mels * mel_frames * sizeof(float));

  // Per-feature normalization (matches sherpa's NemoNormalizePerFeature)
  NormalizePerFeature(signal_data, num_mels, mel_frames);

  // Length tensor
  auto len_shape = std::array<int64_t, 1>{1};
  auto signal_length = OrtValue::CreateTensor(allocator, len_shape,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *signal_length->GetTensorMutableData<int64_t>() = mel_frames;

  // Run encoder
  const char* enc_input_names[] = {
      config_.enc_in_audio.c_str(), config_.enc_in_length.c_str()};
  OrtValue* enc_inputs[] = {processed_signal.get(), signal_length.get()};
  const char* enc_output_names[] = {
      config_.enc_out_encoded.c_str(), config_.enc_out_length.c_str()};

  auto run_options = OrtRunOptions::Create();
  auto enc_outputs = encoder_session_->Run(
      run_options.get(), enc_input_names, enc_inputs, 2, enc_output_names, 2);

  auto* encoded = enc_outputs[0].get();
  int64_t enc_len = *enc_outputs[1]->GetTensorData<int64_t>();
  auto enc_shape = encoded->GetTensorTypeAndShapeInfo()->GetShape();
  int64_t hidden_dim = enc_shape[1];
  const float* enc_data = encoded->GetTensorData<float>();

  // ── TDT Greedy Decode (exact match to sherpa's DecodeOneTDT) ──
  std::vector<TimestampedToken> result;

  // Initial decoder input: blank_id (same as sherpa)
  // dtype detected at construction: int32 for int8 models, int64 for FP32 models
  const bool use_int64 = (decoder_int_dtype_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  auto targets_shape = std::array<int64_t, 2>{1, 1};
  auto targets = OrtValue::CreateTensor(allocator, targets_shape, decoder_int_dtype_);
  if (use_int64)
    *targets->GetTensorMutableData<int64_t>() = static_cast<int64_t>(config_.blank_id);
  else
    *targets->GetTensorMutableData<int32_t>() = static_cast<int32_t>(config_.blank_id);

  auto tgt_len_shape = std::array<int64_t, 1>{1};
  auto target_length = OrtValue::CreateTensor(allocator, tgt_len_shape, decoder_int_dtype_);
  if (use_int64)
    *target_length->GetTensorMutableData<int64_t>() = 1;
  else
    *target_length->GetTensorMutableData<int32_t>() = 1;

  // Initialize decoder state
  auto state_shape = std::array<int64_t, 3>{
      config_.decoder_lstm_layers, 1, config_.decoder_lstm_dim};
  auto state_h = OrtValue::CreateTensor(allocator, state_shape,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memset(state_h->GetTensorMutableRawData(), 0,
      config_.decoder_lstm_layers * config_.decoder_lstm_dim * sizeof(float));
  auto state_c = OrtValue::CreateTensor(allocator, state_shape,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memset(state_c->GetTensorMutableRawData(), 0,
      config_.decoder_lstm_layers * config_.decoder_lstm_dim * sizeof(float));

  // Run decoder once with blank_id
  const char* dec_input_names[] = {
      config_.dec_in_targets.c_str(), config_.dec_in_target_length.c_str(),
      config_.dec_in_states_1.c_str(), config_.dec_in_states_2.c_str()};
  OrtValue* dec_inputs[] = {
      targets.get(), target_length.get(), state_h.get(), state_c.get()};
  const char* dec_output_names[] = {
      config_.dec_out_outputs.c_str(), config_.dec_out_prednet_lengths.c_str(),
      config_.dec_out_states_1.c_str(), config_.dec_out_states_2.c_str()};

  auto dec_outputs = decoder_session_->Run(run_options.get(),
      dec_input_names, dec_inputs, 4, dec_output_names, 4);

  // dec_out = decoder output [1, 640, target_len]
  auto dec_shape = dec_outputs[0]->GetTensorTypeAndShapeInfo()->GetShape();
  int64_t dec_dim = dec_shape[1];

  // Extract decoder output for joiner
  // Channel-first: [1, dec_dim, 1], Channel-last: [1, 1, dec_dim]
  auto dec_frame_shape = joiner_channel_first_
      ? std::array<int64_t, 3>{1, dec_dim, 1}
      : std::array<int64_t, 3>{1, 1, dec_dim};
  auto dec_out = OrtValue::CreateTensor(allocator, dec_frame_shape,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  // Copy last frame from decoder output [1, 640, T] -> take last time step for each channel
  const float* dec_raw = dec_outputs[0]->GetTensorData<float>();
  int64_t dec_T = dec_shape[2];
  float* dec_out_data = dec_out->GetTensorMutableData<float>();
  for (int64_t d = 0; d < dec_dim; ++d) {
    dec_out_data[d] = dec_raw[d * dec_T + (dec_T - 1)];
  }

  state_h = std::move(dec_outputs[2]);
  state_c = std::move(dec_outputs[3]);

  const int max_tokens_per_frame = 5;
  int tokens_this_frame = 0;
  int skip = 0;
  int64_t t = 0;

  const auto& durations = config_.tdt_durations;
  const int token_vocab = config_.vocab_size;
  const int num_extra = config_.tdt_num_extra_outputs;

  while (t < enc_len) {
    // Extract encoder frame for joiner
    // Channel-first: [1, hidden_dim, 1], Channel-last: [1, 1, hidden_dim]
    auto enc_frame_shape = joiner_channel_first_
        ? std::array<int64_t, 3>{1, hidden_dim, 1}
        : std::array<int64_t, 3>{1, 1, hidden_dim};
    auto encoder_frame = OrtValue::CreateTensor(allocator, enc_frame_shape,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    float* frame_data = encoder_frame->GetTensorMutableData<float>();
    // Encoder output is [1, hidden_dim, time], extract column t
    for (int64_t d = 0; d < hidden_dim; ++d) {
      frame_data[d] = enc_data[d * enc_len + t];
    }

    // Run joiner
    const char* join_input_names[] = {
        config_.join_in_encoder.c_str(), config_.join_in_decoder.c_str()};
    OrtValue* join_inputs[] = {encoder_frame.get(), dec_out.get()};
    const char* join_output_names[] = {config_.join_out_logits.c_str()};

    auto join_outputs = joiner_session_->Run(run_options.get(),
        join_input_names, join_inputs, 2, join_output_names, 1);

    const float* logits = join_outputs[0]->GetTensorData<float>();
    auto logits_shape = join_outputs[0]->GetTensorTypeAndShapeInfo()->GetShape();
    int total_logits = 1;
    for (auto d : logits_shape) total_logits *= static_cast<int>(d);

    // Argmax over token logits
    int best_token = 0;
    float best_score = logits[0];
    for (int i = 1; i < token_vocab; ++i) {
      if (logits[i] > best_score) {
        best_score = logits[i];
        best_token = i;
      }
    }

    // Argmax over duration logits (skip = duration index)
    skip = 0;
    if (num_extra > 0) {
      float best_dur_score = logits[token_vocab];
      for (int i = 1; i < num_extra; ++i) {
        if (logits[token_vocab + i] > best_dur_score) {
          best_dur_score = logits[token_vocab + i];
          skip = i;
        }
      }
    }

    if (best_token != config_.blank_id) {
      float abs_time = window_start_sec + t * kFrameSec;
      result.push_back({best_token, abs_time});
      tokens_this_frame++;

      // Run decoder with new token
      auto new_targets = OrtValue::CreateTensor(allocator, targets_shape, decoder_int_dtype_);
      if (use_int64)
        *new_targets->GetTensorMutableData<int64_t>() = static_cast<int64_t>(best_token);
      else
        *new_targets->GetTensorMutableData<int32_t>() = static_cast<int32_t>(best_token);

      auto new_tgt_len = OrtValue::CreateTensor(allocator, tgt_len_shape, decoder_int_dtype_);
      if (use_int64)
        *new_tgt_len->GetTensorMutableData<int64_t>() = 1;
      else
        *new_tgt_len->GetTensorMutableData<int32_t>() = 1;

      OrtValue* new_dec_inputs[] = {
          new_targets.get(), new_tgt_len.get(), state_h.get(), state_c.get()};
      auto new_dec_outputs = decoder_session_->Run(run_options.get(),
          dec_input_names, new_dec_inputs, 4, dec_output_names, 4);

      // Update decoder output for joiner (reuses existing dec_out shape)
      const float* new_dec_raw = new_dec_outputs[0]->GetTensorData<float>();
      auto new_dec_shape = new_dec_outputs[0]->GetTensorTypeAndShapeInfo()->GetShape();
      int64_t new_dec_T = new_dec_shape[2];
      float* upd_data = dec_out->GetTensorMutableData<float>();
      for (int64_t d = 0; d < new_dec_shape[1]; ++d) {
        upd_data[d] = new_dec_raw[d * new_dec_T + (new_dec_T - 1)];
      }

      state_h = std::move(new_dec_outputs[2]);
      state_c = std::move(new_dec_outputs[3]);
    }

    // TDT advance logic (exact match to sherpa)
    if (skip > 0) tokens_this_frame = 0;
    if (tokens_this_frame >= max_tokens_per_frame) { tokens_this_frame = 0; skip = 1; }
    if (best_token == config_.blank_id && skip == 0) { tokens_this_frame = 0; skip = 1; }
    t += skip;
  }

  return result;
}

// ─── TranscribeChunk ────────────────────────────────────────────────────────

std::string ParakeetStreamingASR::TranscribeChunk(const float* audio_data, size_t num_samples) {
  LoadVocab();

  // Append to audio buffer
  audio_buffer_.insert(audio_buffer_.end(), audio_data, audio_data + num_samples);
  total_audio_sec_ = static_cast<float>(audio_buffer_.size()) / config_.sample_rate;

  // Cap audio buffer at max window
  size_t max_samples = static_cast<size_t>(kMaxWindowSec * config_.sample_rate);
  float window_start_sec = 0.0f;
  const float* segment = audio_buffer_.data();
  size_t segment_len = audio_buffer_.size();
  if (audio_buffer_.size() > max_samples) {
    segment = audio_buffer_.data() + audio_buffer_.size() - max_samples;
    segment_len = max_samples;
    window_start_sec = total_audio_sec_ - kMaxWindowSec;
  }

  // Encode and decode
  auto hyp = EncodeAndDecodeTDT(segment, segment_len, window_start_sec);

  if (hyp.empty()) return "";

  // Split into stable (old enough) and unstable (recent)
  float stable_cutoff = total_audio_sec_ - kStableDelaySec;
  float last_committed_time = committed_tokens_.empty() ? -1.0f
      : committed_tokens_.back().abs_time;

  std::string new_text;
  for (auto& tok : hyp) {
    if (tok.abs_time <= stable_cutoff && tok.abs_time > last_committed_time) {
      // Check for token-level dedup
      bool is_dup = false;
      if (!committed_tokens_.empty()) {
        size_t n = committed_tokens_.size();
        for (size_t k = 1; k <= std::min(n, size_t(5)); ++k) {
          // This simple check just skips exact timestamp matches
          if (committed_tokens_[n - 1].token_id == tok.token_id &&
              std::abs(committed_tokens_[n - 1].abs_time - tok.abs_time) < 0.16f) {
            is_dup = true;
            break;
          }
        }
      }
      if (!is_dup) {
        committed_tokens_.push_back(tok);
        // Convert token to text
        if (tok.token_id < static_cast<int>(vocab_.size())) {
          std::string token_str = vocab_[tok.token_id];
          size_t pos = 0;
          while ((pos = token_str.find("\xe2\x96\x81", pos)) != std::string::npos) {
            token_str.replace(pos, 3, " ");
            pos += 1;
          }
          new_text += token_str;
        }
      }
    }
  }

  full_transcript_ += new_text;
  chunk_index_++;
  return new_text;
}

// ─── Flush ──────────────────────────────────────────────────────────────────

std::string ParakeetStreamingASR::Flush() {
  LoadVocab();

  if (audio_buffer_.empty()) return "";

  // Final decode on remaining audio
  float window_start_sec = 0.0f;
  size_t max_samples = static_cast<size_t>(kMaxWindowSec * config_.sample_rate);
  const float* segment = audio_buffer_.data();
  size_t segment_len = audio_buffer_.size();
  if (audio_buffer_.size() > max_samples) {
    segment = audio_buffer_.data() + audio_buffer_.size() - max_samples;
    segment_len = max_samples;
    window_start_sec = total_audio_sec_ - kMaxWindowSec;
  }

  auto hyp = EncodeAndDecodeTDT(segment, segment_len, window_start_sec);

  // Commit ALL remaining tokens (no stable delay check)
  float last_committed_time = committed_tokens_.empty() ? -1.0f
      : committed_tokens_.back().abs_time;

  std::string new_text;
  for (auto& tok : hyp) {
    if (tok.abs_time > last_committed_time) {
      committed_tokens_.push_back(tok);
      if (tok.token_id < static_cast<int>(vocab_.size())) {
        std::string token_str = vocab_[tok.token_id];
        size_t pos = 0;
        while ((pos = token_str.find("\xe2\x96\x81", pos)) != std::string::npos) {
          token_str.replace(pos, 3, " ");
          pos += 1;
        }
        new_text += token_str;
      }
    }
  }

  full_transcript_ += new_text;
  return new_text;
}

}  // namespace Generators
