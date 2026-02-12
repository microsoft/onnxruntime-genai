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

// ─── Mel spectrogram utilities (Slaney scale, matching librosa/NeMo) ────────

// Slaney mel scale: linear below 1000 Hz, logarithmic above
static constexpr float kMinLogHz = 1000.0f;
static constexpr float kMinLogMel = 15.0f;              // 1000 / (200/3)
static constexpr float kLinScale = 200.0f / 3.0f;       // Hz per mel (linear region)
static constexpr float kLogStep = 0.06875177742094912f;  // log(6.4) / 27

static float HzToMel(float hz) {
  if (hz < kMinLogHz) return hz / kLinScale;
  return kMinLogMel + std::log(hz / kMinLogHz) / kLogStep;
}
static float MelToHz(float mel) {
  if (mel < kMinLogMel) return mel * kLinScale;
  return kMinLogHz * std::exp((mel - kMinLogMel) * kLogStep);
}

void StreamingASR::InitMelFilterbank() {
  int num_bins = kFFTSize / 2 + 1;
  float mel_low = HzToMel(0.0f);
  float mel_high = HzToMel(static_cast<float>(kSampleRate) / 2.0f);

  // Compute mel center frequencies in Hz (num_mels + 2 points)
  std::vector<float> mel_f(kNumMels + 2);
  for (int i = 0; i < kNumMels + 2; ++i) {
    float mel = mel_low + (mel_high - mel_low) * i / (kNumMels + 1);
    mel_f[i] = MelToHz(mel);
  }

  // Differences between consecutive mel center frequencies (Hz)
  std::vector<float> fdiff(kNumMels + 1);
  for (int i = 0; i < kNumMels + 1; ++i) {
    fdiff[i] = mel_f[i + 1] - mel_f[i];
  }

  // FFT bin center frequencies in Hz
  std::vector<float> fft_freqs(num_bins);
  for (int k = 0; k < num_bins; ++k) {
    fft_freqs[k] = static_cast<float>(k) * kSampleRate / kFFTSize;
  }

  // Build triangular filterbank with Slaney normalization (matches librosa exactly)
  mel_filters_.resize(kNumMels, std::vector<float>(num_bins, 0.0f));
  for (int m = 0; m < kNumMels; ++m) {
    for (int k = 0; k < num_bins; ++k) {
      float lower = (fft_freqs[k] - mel_f[m]) / (fdiff[m] + 1e-10f);
      float upper = (mel_f[m + 2] - fft_freqs[k]) / (fdiff[m + 1] + 1e-10f);
      mel_filters_[m][k] = std::max(0.0f, std::min(lower, upper));
    }
    // Slaney area normalization: 2 / bandwidth
    float enorm = 2.0f / (mel_f[m + 2] - mel_f[m] + 1e-10f);
    for (int k = 0; k < num_bins; ++k) {
      mel_filters_[m][k] *= enorm;
    }
  }

  hann_window_.resize(kWinLength);
  for (int i = 0; i < kWinLength; ++i) {
    hann_window_[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / kWinLength));
  }
}

std::pair<std::vector<float>, int> StreamingASR::ComputeLogMel(const float* audio, size_t num_samples) {
  // Center-padded STFT: prepend overlap from previous chunk (or zeros for first chunk)
  // This ensures we get exactly 56 mel frames for 8960-sample chunks,
  // matching NeMo's center=True STFT behavior.
  int pad = kFFTSize / 2;  // 256 samples
  std::vector<float> padded(pad + num_samples);
  std::memcpy(padded.data(), audio_overlap_.data(), pad * sizeof(float));
  std::memcpy(padded.data() + pad, audio, num_samples * sizeof(float));

  // Update overlap buffer with tail of current chunk for next call
  if (num_samples >= static_cast<size_t>(pad)) {
    audio_overlap_.assign(audio + num_samples - pad, audio + num_samples);
  } else {
    size_t keep = pad - num_samples;
    std::vector<float> new_overlap(pad, 0.0f);
    std::memcpy(new_overlap.data(), audio_overlap_.data() + num_samples, keep * sizeof(float));
    std::memcpy(new_overlap.data() + keep, audio, num_samples * sizeof(float));
    audio_overlap_ = std::move(new_overlap);
  }

  if (static_cast<int>(padded.size()) < kWinLength) {
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
      mel_spec[m * num_frames + t] = std::log(std::max(val, 5.96046448e-08f));  // 2^-24 (NeMo default)
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
      // Format: "token_text index" (space-separated) or "token_text\tindex" (tab-separated)
      auto sep_pos = line.rfind(' ');
      if (sep_pos == std::string::npos) sep_pos = line.rfind('\t');
      if (sep_pos != std::string::npos) {
        vocab_.push_back(line.substr(0, sep_pos));
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
  decoder_session_ = nemotron_model->session_decoder_.get();
  joiner_session_ = nemotron_model->session_joiner_.get();
  cache_config_ = nemotron_model->cache_config_;

  // Initialize mel filterbank
  InitMelFilterbank();

  // Initialize audio overlap buffer for center-padded STFT
  audio_overlap_.assign(kFFTSize / 2, 0.0f);

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
  audio_overlap_.assign(kFFTSize / 2, 0.0f);
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
      "audio_signal", "length",
      "cache_last_channel", "cache_last_time", "cache_last_channel_len"};
  OrtValue* enc_inputs[] = {
      processed_signal.get(), signal_length.get(),
      encoder_cache_.cache_last_channel.get(),
      encoder_cache_.cache_last_time.get(),
      encoder_cache_.cache_last_channel_len.get()};

  // Encoder outputs — let ORT allocate
  const char* enc_output_names[] = {
      "outputs", "encoded_lengths",
      "cache_last_channel_next", "cache_last_time_next", "cache_last_channel_next_len"};

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
      // ── Step 1: Run decoder (prediction network) ──
      auto targets_shape = std::array<int64_t, 2>{1, 1};
      auto targets = OrtValue::CreateTensor(allocator, targets_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
      *targets->GetTensorMutableData<int32_t>() = decoder_state_.last_token;

      auto tgt_len_shape = std::array<int64_t, 1>{1};
      auto target_length = OrtValue::CreateTensor(allocator, tgt_len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
      *target_length->GetTensorMutableData<int32_t>() = 1;

      const char* dec_input_names[] = {
          "targets", "target_length",
          "states.1", "onnx::Slice_3"};
      OrtValue* dec_inputs[] = {
          targets.get(), target_length.get(),
          decoder_state_.state_1.get(), decoder_state_.state_2.get()};

      const char* dec_output_names[] = {
          "outputs", "prednet_lengths", "states", "162"};

      auto dec_outputs = decoder_session_->Run(
          run_options.get(),
          dec_input_names, dec_inputs, 4,
          dec_output_names, 4);

      // dec_outputs[0] = decoder hidden [1, 640, 1]
      // dec_outputs[1] = prednet_lengths [1]
      // dec_outputs[2] = new states h [2, ?, 640]
      // dec_outputs[3] = new states c [2, ?, 640]

      // ── Step 2: Run joiner (joint network) ──
      const char* join_input_names[] = {
          "encoder_outputs", "decoder_outputs"};
      OrtValue* join_inputs[] = {
          encoder_frame.get(), dec_outputs[0].get()};

      const char* join_output_names[] = {"outputs"};

      auto join_outputs = joiner_session_->Run(
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

      // Blank => next time step
      if (best_token == cache_config_.blank_id || best_token >= cache_config_.vocab_size) {
        break;
      }

      // Emit token & update state
      decoder_state_.last_token = best_token;
      decoder_state_.state_1 = std::move(dec_outputs[2]);
      decoder_state_.state_2 = std::move(dec_outputs[3]);

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
