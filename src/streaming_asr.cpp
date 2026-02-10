// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// StreamingASR implementation — high-level streaming speech recognition.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>

#include "generators.h"
#include "streaming_asr.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Generators {

// ─── Mel spectrogram utilities ──────────────────────────────────────────────

static float HzToMel(float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); }
static float MelToHz(float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); }

void StreamingASR::InitMelFilterbank() {
  int num_bins = kFFTSize / 2 + 1;
  float mel_low = HzToMel(0.0f);
  float mel_high = HzToMel(static_cast<float>(kSampleRate) / 2.0f);

  std::vector<float> mel_points(kNumMels + 2);
  for (int i = 0; i < kNumMels + 2; ++i) {
    mel_points[i] = MelToHz(mel_low + (mel_high - mel_low) * i / (kNumMels + 1));
  }

  std::vector<float> bin_points(kNumMels + 2);
  for (int i = 0; i < kNumMels + 2; ++i) {
    bin_points[i] = (kFFTSize + 1) * mel_points[i] / kSampleRate;
  }

  mel_filters_.resize(kNumMels, std::vector<float>(num_bins, 0.0f));
  for (int m = 0; m < kNumMels; ++m) {
    for (int k = 0; k < num_bins; ++k) {
      float fk = static_cast<float>(k);
      if (fk >= bin_points[m] && fk <= bin_points[m + 1]) {
        mel_filters_[m][k] = (fk - bin_points[m]) / (bin_points[m + 1] - bin_points[m] + 1e-10f);
      } else if (fk >= bin_points[m + 1] && fk <= bin_points[m + 2]) {
        mel_filters_[m][k] = (bin_points[m + 2] - fk) / (bin_points[m + 2] - bin_points[m + 1] + 1e-10f);
      }
    }
  }

  hann_window_.resize(kWinLength);
  for (int i = 0; i < kWinLength; ++i) {
    hann_window_[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / kWinLength));
  }
}

std::pair<std::vector<float>, int> StreamingASR::ComputeLogMel(const float* audio, size_t num_samples) {
  // Pad if too short
  std::vector<float> padded(audio, audio + num_samples);
  if (static_cast<int>(num_samples) < kWinLength) {
    padded.resize(kWinLength, 0.0f);
  }

  int num_frames = static_cast<int>((padded.size() - kWinLength) / kHopLength) + 1;
  int num_bins = kFFTSize / 2 + 1;

  std::vector<float> mel_spec(kNumMels * num_frames);

  for (int t = 0; t < num_frames; ++t) {
    // Compute power spectrum for this frame
    std::vector<float> magnitudes(num_bins);
    const float* frame = padded.data() + t * kHopLength;

    for (int k = 0; k < num_bins; ++k) {
      float real_sum = 0.0f, imag_sum = 0.0f;
      for (int n = 0; n < kWinLength; ++n) {
        float val = frame[n] * hann_window_[n];
        float angle = 2.0f * static_cast<float>(M_PI) * k * n / kFFTSize;
        real_sum += val * std::cos(angle);
        imag_sum -= val * std::sin(angle);
      }
      magnitudes[k] = real_sum * real_sum + imag_sum * imag_sum;
    }

    // Apply mel filterbank
    for (int m = 0; m < kNumMels; ++m) {
      float val = 0.0f;
      for (int k = 0; k < num_bins; ++k) {
        val += mel_filters_[m][k] * magnitudes[k];
      }
      mel_spec[m * num_frames + t] = std::log(std::max(val, 1e-10f));
    }
  }

  return {mel_spec, num_frames};
}

// ─── Vocabulary loading ─────────────────────────────────────────────────────

void StreamingASR::LoadVocab() {
  if (vocab_loaded_) return;

  auto config_path = model_.config_->config_path;

  // Try tokens.txt
  auto tokens_path = config_path / "tokens.txt";
  std::ifstream tokens_file(tokens_path.string());
  if (tokens_file.is_open()) {
    std::string line;
    while (std::getline(tokens_file, line)) {
      auto tab_pos = line.find('\t');
      if (tab_pos != std::string::npos) {
        vocab_.push_back(line.substr(0, tab_pos));
      } else {
        vocab_.push_back(line);
      }
    }
    vocab_loaded_ = true;
    return;
  }

  // Fallback: use tokenizer
  try {
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
    vocab_loaded_ = true;
    return;
  } catch (...) {
  }

  // Last resort
  vocab_.resize(cache_config_.vocab_size);
  for (int i = 0; i < cache_config_.vocab_size; ++i) {
    vocab_[i] = "<" + std::to_string(i) + ">";
  }
  vocab_loaded_ = true;
}

// ─── StreamingASR ───────────────────────────────────────────────────────────

StreamingASR::StreamingASR(Model& model)
    : model_{model} {
  // Get the NemotronSpeechModel to access its sessions
  auto* nemotron_model = dynamic_cast<NemotronSpeechModel*>(&model);
  if (!nemotron_model) {
    throw std::runtime_error("StreamingASR requires a nemotron_speech model type. Got: " + model.config_->model.type);
  }

  encoder_session_ = nemotron_model->session_encoder_.get();
  decoder_session_ = nemotron_model->session_decoder_joint_.get();
  cache_config_ = nemotron_model->cache_config_;

  // Initialize mel filterbank
  InitMelFilterbank();

  // Initialize streaming state
  auto& allocator = model_.allocator_cpu_;
  encoder_cache_.Initialize(cache_config_, allocator);
  decoder_state_.Initialize(cache_config_, allocator);
}

StreamingASR::~StreamingASR() = default;

void StreamingASR::Reset() {
  auto& allocator = model_.allocator_cpu_;
  encoder_cache_.Reset(cache_config_, allocator);
  decoder_state_.Reset(cache_config_, allocator);
  full_transcript_.clear();
}

std::string StreamingASR::TranscribeChunk(const float* audio_data, size_t num_samples) {
  LoadVocab();

  // Compute log-mel spectrogram
  auto [mel_data, num_frames] = ComputeLogMel(audio_data, num_samples);

  auto& allocator = model_.allocator_cpu_;

  // Create processed_signal: [1, num_mels, num_frames]
  auto signal_shape = std::array<int64_t, 3>{1, kNumMels, num_frames};
  auto processed_signal = OrtValue::CreateTensor(allocator, signal_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memcpy(processed_signal->GetTensorMutableRawData(), mel_data.data(), mel_data.size() * sizeof(float));

  // Create processed_signal_length: [1]
  auto len_shape = std::array<int64_t, 1>{1};
  auto signal_length = OrtValue::CreateTensor(allocator, len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *signal_length->GetTensorMutableData<int64_t>() = static_cast<int64_t>(num_frames);

  // Encoder inputs
  const char* enc_input_names[] = {
      "processed_signal", "processed_signal_length",
      "cache_last_channel", "cache_last_time", "cache_last_channel_len"};
  OrtValue* enc_inputs[] = {
      processed_signal.get(), signal_length.get(),
      encoder_cache_.cache_last_channel.get(),
      encoder_cache_.cache_last_time.get(),
      encoder_cache_.cache_last_channel_len.get()};

  // Encoder outputs — let ORT allocate
  const char* enc_output_names[] = {
      "encoded", "encoded_len",
      "cache_last_channel_next", "cache_last_time_next", "cache_last_channel_len_next"};

  // Run encoder
  auto run_options = OrtRunOptions::Create();
  auto enc_outputs = encoder_session_->Run(
      run_options.get(),
      enc_input_names, enc_inputs, 5,
      enc_output_names, 5);

  // Parse encoder outputs
  auto* encoded = enc_outputs[0].get();
  int64_t encoded_len = *enc_outputs[1]->GetTensorData<int64_t>();

  // Update cache
  encoder_cache_.cache_last_channel = std::move(enc_outputs[2]);
  encoder_cache_.cache_last_time = std::move(enc_outputs[3]);
  encoder_cache_.cache_last_channel_len = std::move(enc_outputs[4]);

  // Run RNNT decoder
  std::string chunk_text = RunRNNTDecoder(encoded, encoded_len);
  full_transcript_ += chunk_text;

  return chunk_text;
}

std::string StreamingASR::RunRNNTDecoder(OrtValue* encoder_output, int64_t encoded_len) {
  auto& allocator = model_.allocator_cpu_;
  std::string result;

  auto enc_info = encoder_output->GetTensorTypeAndShapeInfo();
  auto enc_shape = enc_info->GetShape();
  int64_t hidden_dim = enc_shape[1];
  int64_t time_steps = std::min(enc_shape[2], encoded_len);
  const float* enc_data = encoder_output->GetTensorData<float>();

  auto run_options = OrtRunOptions::Create();

  for (int64_t t = 0; t < time_steps; ++t) {
    // Extract single encoder frame: [1, hidden_dim, 1]
    auto frame_shape = std::array<int64_t, 3>{1, hidden_dim, 1};
    auto encoder_frame = OrtValue::CreateTensor(allocator, frame_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    float* frame_data = encoder_frame->GetTensorMutableData<float>();
    for (int64_t d = 0; d < hidden_dim; ++d) {
      frame_data[d] = enc_data[d * enc_shape[2] + t];
    }

    constexpr int kMaxSymbolsPerStep = 10;
    for (int sym = 0; sym < kMaxSymbolsPerStep; ++sym) {
      // Prepare decoder inputs
      auto targets_shape = std::array<int64_t, 2>{1, 1};
      auto targets = OrtValue::CreateTensor(allocator, targets_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
      *targets->GetTensorMutableData<int32_t>() = decoder_state_.last_token;

      auto tgt_len_shape = std::array<int64_t, 1>{1};
      auto target_length = OrtValue::CreateTensor(allocator, tgt_len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
      *target_length->GetTensorMutableData<int32_t>() = 1;

      const char* dec_input_names[] = {
          "encoder_outputs", "targets", "target_length",
          "input_states_1", "input_states_2"};
      OrtValue* dec_inputs[] = {
          encoder_frame.get(), targets.get(), target_length.get(),
          decoder_state_.state_1.get(), decoder_state_.state_2.get()};

      const char* dec_output_names[] = {
          "outputs", "output_states_1", "output_states_2"};

      auto dec_outputs = decoder_session_->Run(
          run_options.get(),
          dec_input_names, dec_inputs, 5,
          dec_output_names, 3);

      // Find argmax
      const float* logits_data = dec_outputs[0]->GetTensorData<float>();
      auto logits_shape = dec_outputs[0]->GetTensorTypeAndShapeInfo()->GetShape();
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

      // Blank => next time step
      if (best_token == cache_config_.blank_id || best_token >= cache_config_.vocab_size) {
        break;
      }

      // Emit token & update state
      decoder_state_.last_token = best_token;
      decoder_state_.state_1 = std::move(dec_outputs[1]);
      decoder_state_.state_2 = std::move(dec_outputs[2]);

      if (best_token < static_cast<int>(vocab_.size())) {
        std::string token_str = vocab_[best_token];
        // Replace sentencepiece space marker "▁" with space
        size_t pos = 0;
        while ((pos = token_str.find("\xe2\x96\x81", pos)) != std::string::npos) {
          token_str.replace(pos, 3, " ");
          pos += 1;
        }
        result += token_str;
      }
    }
  }

  return result;
}

std::unique_ptr<StreamingASR> CreateStreamingASR(Model& model) {
  return std::make_unique<StreamingASR>(model);
}

}  // namespace Generators
