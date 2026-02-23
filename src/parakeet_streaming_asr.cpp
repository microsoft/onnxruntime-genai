// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// ParakeetStreamingASR — streaming ASR for Parakeet FastConformer + TDT models.
//
// The encoder is non-cache-aware. We accumulate all mel features across chunks,
// re-normalize and re-encode the full mel each time, but only TDT-decode the
// NEW encoder frames (those beyond what we decoded last time).
// The decoder LSTM state is maintained across chunks for continuity.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>

#include "generators.h"
#include "parakeet_streaming_asr.h"

namespace Generators {

// ─── Vocabulary loading ─────────────────────────────────────────────────────

void ParakeetStreamingASR::LoadVocab() {
  if (vocab_loaded_) return;

  // Load vocab directly from vocab.txt to preserve ▁ (sentencepiece space marker)
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
    // Fallback: use tokenizer
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

// ─── ParakeetStreamingASR ────────────────────────────────────────────────────

ParakeetStreamingASR::ParakeetStreamingASR(Model& model)
    : model_{model} {
  auto* parakeet_model = dynamic_cast<ParakeetSpeechModel*>(&model);
  if (!parakeet_model) {
    throw std::runtime_error("ParakeetStreamingASR requires a parakeet_tdt model type. Got: " + model.config_->model.type);
  }

  encoder_session_ = parakeet_model->session_encoder_.get();
  decoder_session_ = parakeet_model->session_decoder_.get();
  joiner_session_ = parakeet_model->session_joiner_.get();
  config_ = parakeet_model->parakeet_config_;

  // Initialize mel extractor from config
  nemo_mel::NemoMelConfig mel_cfg{
      config_.num_mels, config_.fft_size,
      config_.hop_length, config_.win_length,
      config_.sample_rate,
      config_.preemph, config_.log_eps};
  mel_extractor_ = nemo_mel::NemoStreamingMelExtractor{mel_cfg};

  // Initialize accumulated mel storage (one vector per mel bin)
  accumulated_mel_.resize(config_.num_mels);

  // Initialize decoder state
  auto& allocator = model_.allocator_cpu_;
  decoder_state_.Initialize(config_, allocator);
}

ParakeetStreamingASR::~ParakeetStreamingASR() = default;

void ParakeetStreamingASR::Reset() {
  auto& allocator = model_.allocator_cpu_;
  decoder_state_.Reset(config_, allocator);
  full_transcript_.clear();
  mel_extractor_.Reset();
  audio_buffer_.clear();
  accumulated_mel_.clear();
  accumulated_mel_.resize(config_.num_mels);
  total_mel_frames_ = 0;
  prev_decoded_frames_ = 0;
  chunk_index_ = 0;
}

std::string ParakeetStreamingASR::TranscribeChunk(const float* audio_data, size_t num_samples) {
  LoadVocab();

  // Append incoming audio to accumulation buffer
  audio_buffer_.insert(audio_buffer_.end(), audio_data, audio_data + num_samples);

  std::string result;
  const size_t chunk_sz = static_cast<size_t>(config_.chunk_samples);

  // Process complete chunks of audio
  while (audio_buffer_.size() >= chunk_sz) {
    // Compute mel for this chunk (streaming mel extractor handles overlap)
    auto [mel_data, num_frames] = mel_extractor_.Process(audio_buffer_.data(), chunk_sz);

    // Append new mel frames to accumulated buffer
    // mel_data is [num_mels, num_frames] row-major
    for (int m = 0; m < config_.num_mels; ++m) {
      const float* row = mel_data.data() + m * num_frames;
      accumulated_mel_[m].insert(accumulated_mel_[m].end(), row, row + num_frames);
    }
    total_mel_frames_ += num_frames;

    // Advance past processed audio samples
    audio_buffer_.erase(audio_buffer_.begin(),
                        audio_buffer_.begin() + static_cast<ptrdiff_t>(chunk_sz));

    // Re-encode ALL accumulated mel and decode only NEW frames
    result += EncodeAndDecode();
  }

  return result;
}

std::string ParakeetStreamingASR::Flush() {
  LoadVocab();

  std::string result;
  const size_t chunk_sz = static_cast<size_t>(config_.chunk_samples);

  // Process any remaining audio (pad to full chunk with silence)
  if (!audio_buffer_.empty()) {
    audio_buffer_.resize(chunk_sz, 0.0f);

    auto [mel_data, num_frames] = mel_extractor_.Process(audio_buffer_.data(), chunk_sz);

    for (int m = 0; m < config_.num_mels; ++m) {
      const float* row = mel_data.data() + m * num_frames;
      accumulated_mel_[m].insert(accumulated_mel_[m].end(), row, row + num_frames);
    }
    total_mel_frames_ += num_frames;

    audio_buffer_.clear();
    result += EncodeAndDecode();
  }

  return result;
}

std::string ParakeetStreamingASR::EncodeAndDecode() {
  auto& allocator = model_.allocator_cpu_;
  const int num_mels = config_.num_mels;
  const int T = total_mel_frames_;

  if (T == 0) return "";

  // Build mel tensor [1, num_mels, T] with per-feature normalization
  auto signal_shape = std::array<int64_t, 3>{1, num_mels, T};
  auto processed_signal = OrtValue::CreateTensor(allocator, signal_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  float* signal_data = processed_signal->GetTensorMutableData<float>();

  for (int m = 0; m < num_mels; ++m) {
    float* row = signal_data + m * T;
    // Copy accumulated mel for this bin
    std::memcpy(row, accumulated_mel_[m].data(), T * sizeof(float));

    // Per-feature normalization: zero-mean, unit-variance per mel bin
    float mean = 0.0f;
    for (int t = 0; t < T; ++t) mean += row[t];
    mean /= static_cast<float>(T);

    float var = 0.0f;
    for (int t = 0; t < T; ++t) {
      float d = row[t] - mean;
      var += d * d;
    }
    float inv_std = 1.0f / (std::sqrt(var / static_cast<float>(T)) + 1e-5f);
    for (int t = 0; t < T; ++t) {
      row[t] = (row[t] - mean) * inv_std;
    }
  }

  // Create length tensor
  auto len_shape = std::array<int64_t, 1>{1};
  auto signal_length = OrtValue::CreateTensor(allocator, len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *signal_length->GetTensorMutableData<int64_t>() = static_cast<int64_t>(T);

  // Run encoder on full accumulated mel
  const char* enc_input_names[] = {
      config_.enc_in_audio.c_str(), config_.enc_in_length.c_str()};
  OrtValue* enc_inputs[] = {
      processed_signal.get(), signal_length.get()};
  const char* enc_output_names[] = {
      config_.enc_out_encoded.c_str(), config_.enc_out_length.c_str()};

  auto run_options = OrtRunOptions::Create();
  auto enc_outputs = encoder_session_->Run(
      run_options.get(),
      enc_input_names, enc_inputs, 2,
      enc_output_names, 2);

  auto* encoded = enc_outputs[0].get();
  int64_t encoded_len = *enc_outputs[1]->GetTensorData<int64_t>();

  // Only decode frames beyond what we already decoded
  int64_t start_frame = prev_decoded_frames_;
  std::string chunk_text = RunTDTDecoder(encoded, encoded_len, start_frame);
  prev_decoded_frames_ = encoded_len;

  full_transcript_ += chunk_text;
  chunk_index_++;

  return chunk_text;
}

std::string ParakeetStreamingASR::RunTDTDecoder(OrtValue* encoder_output,
                                                 int64_t encoded_len,
                                                 int64_t start_frame) {
  auto& allocator = model_.allocator_cpu_;
  std::string result;

  auto enc_info = encoder_output->GetTensorTypeAndShapeInfo();
  auto enc_shape = enc_info->GetShape();
  const float* enc_data = encoder_output->GetTensorData<float>();

  int64_t total_time, hidden_dim;
  bool is_hidden_first;

  if (enc_shape[2] == encoded_len || (enc_shape[1] > enc_shape[2] && enc_shape[2] <= encoded_len)) {
    hidden_dim = enc_shape[1];
    total_time = std::min(enc_shape[2], encoded_len);
    is_hidden_first = true;
  } else {
    total_time = std::min(enc_shape[1], encoded_len);
    hidden_dim = enc_shape[2];
    is_hidden_first = false;
  }

  // Only decode from start_frame to total_time
  if (start_frame >= total_time) return result;

  const auto& durations = config_.tdt_durations;
  const int num_extra = config_.tdt_num_extra_outputs;
  const int token_vocab = config_.vocab_size;

  auto run_options = OrtRunOptions::Create();

  int64_t t = start_frame;
  while (t < total_time) {
    auto enc_frame_shape = std::array<int64_t, 3>{1, 1, hidden_dim};
    auto encoder_frame = OrtValue::CreateTensor(allocator, enc_frame_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    float* frame_data = encoder_frame->GetTensorMutableData<float>();

    if (is_hidden_first) {
      for (int64_t d = 0; d < hidden_dim; ++d) {
        frame_data[d] = enc_data[d * total_time + t];
      }
    } else {
      std::memcpy(frame_data, enc_data + t * hidden_dim, hidden_dim * sizeof(float));
    }

    const int max_sym = config_.max_symbols_per_step;
    for (int sym = 0; sym < max_sym; ++sym) {
      auto targets_shape = std::array<int64_t, 2>{1, 1};
      auto targets = OrtValue::CreateTensor(allocator, targets_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
      *targets->GetTensorMutableData<int64_t>() = decoder_state_.last_token;

      auto tgt_len_shape = std::array<int64_t, 1>{1};
      auto target_length = OrtValue::CreateTensor(allocator, tgt_len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
      *target_length->GetTensorMutableData<int64_t>() = 1;

      const char* dec_input_names[] = {
          config_.dec_in_targets.c_str(), config_.dec_in_target_length.c_str(),
          config_.dec_in_states_1.c_str(), config_.dec_in_states_2.c_str()};
      OrtValue* dec_inputs[] = {
          targets.get(), target_length.get(),
          decoder_state_.state_h.get(), decoder_state_.state_c.get()};

      const char* dec_output_names[] = {
          config_.dec_out_outputs.c_str(), config_.dec_out_prednet_lengths.c_str(),
          config_.dec_out_states_1.c_str(), config_.dec_out_states_2.c_str()};

      auto dec_outputs = decoder_session_->Run(
          run_options.get(),
          dec_input_names, dec_inputs, 4,
          dec_output_names, 4);

      auto dec_shape = dec_outputs[0]->GetTensorTypeAndShapeInfo()->GetShape();
      int64_t dec_dim = dec_shape[1];
      auto dec_frame_shape = std::array<int64_t, 3>{1, 1, dec_dim};
      auto decoder_frame = OrtValue::CreateTensor(allocator, dec_frame_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      std::memcpy(decoder_frame->GetTensorMutableData<float>(),
                  dec_outputs[0]->GetTensorData<float>(),
                  dec_dim * sizeof(float));

      const char* join_input_names[] = {
          config_.join_in_encoder.c_str(), config_.join_in_decoder.c_str()};
      OrtValue* join_inputs[] = {
          encoder_frame.get(), decoder_frame.get()};
      const char* join_output_names[] = {config_.join_out_logits.c_str()};

      auto join_outputs = joiner_session_->Run(
          run_options.get(),
          join_input_names, join_inputs, 2,
          join_output_names, 1);

      const float* logits_data = join_outputs[0]->GetTensorData<float>();

      int best_token = 0;
      float best_score = logits_data[0];
      for (int i = 1; i < token_vocab; ++i) {
        if (logits_data[i] > best_score) {
          best_score = logits_data[i];
          best_token = i;
        }
      }

      int best_dur_idx = 0;
      if (num_extra > 0 && !durations.empty()) {
        float best_dur_score = logits_data[token_vocab];
        for (int i = 1; i < num_extra; ++i) {
          if (logits_data[token_vocab + i] > best_dur_score) {
            best_dur_score = logits_data[token_vocab + i];
            best_dur_idx = i;
          }
        }
      }

      if (best_token == config_.blank_id) {
        t += 1;
        break;
      }

      decoder_state_.last_token = best_token;
      decoder_state_.state_h = std::move(dec_outputs[2]);
      decoder_state_.state_c = std::move(dec_outputs[3]);

      if (best_token < static_cast<int>(vocab_.size())) {
        std::string token_str = vocab_[best_token];
        size_t pos = 0;
        while ((pos = token_str.find("\xe2\x96\x81", pos)) != std::string::npos) {
          token_str.replace(pos, 3, " ");
          pos += 1;
        }
        result += token_str;
      }

      int predicted_duration = (best_dur_idx < static_cast<int>(durations.size()))
                                   ? durations[best_dur_idx]
                                   : 1;
      int advance = std::max(1, predicted_duration);
      t += advance;
      break;
    }
  }

  return result;
}

}  // namespace Generators
