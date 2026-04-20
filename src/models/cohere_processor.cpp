// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "cohere_processor.h"
#include "nemo_mel_spectrogram.h"

#include <cstring>
#include <cmath>
#include <numeric>

namespace Generators {

// Mirror of ort-extensions internal RawAudiosObject layout to access raw audio bytes.
struct RawAudiosAccess : ort_extensions::OrtxObjectImpl {
  std::unique_ptr<ort_extensions::AudioRawData[]> audios_;
  size_t num_audios_{};
};

// Decode 16-bit PCM WAV file bytes to float32 samples.
static std::vector<float> DecodeWav16(const std::byte* data, size_t size) {
  if (size < 44) return {};
  auto u8 = reinterpret_cast<const uint8_t*>(data);
  if (std::memcmp(u8, "RIFF", 4) != 0 || std::memcmp(u8 + 8, "WAVE", 4) != 0)
    return {};
  size_t pos = 12;
  while (pos + 8 <= size) {
    uint32_t chunk_size = u8[pos + 4] | (u8[pos + 5] << 8) | (u8[pos + 6] << 16) | (u8[pos + 7] << 24);
    if (std::memcmp(u8 + pos, "data", 4) == 0) {
      size_t data_start = pos + 8;
      size_t data_bytes = std::min(static_cast<size_t>(chunk_size), size - data_start);
      size_t num_samples = data_bytes / 2;
      std::vector<float> samples(num_samples);
      auto pcm16 = reinterpret_cast<const int16_t*>(u8 + data_start);
      for (size_t i = 0; i < num_samples; ++i)
        samples[i] = static_cast<float>(pcm16[i]) / 32768.0f;
      return samples;
    }
    pos += 8 + chunk_size;
    if (chunk_size & 1) pos++;
  }
  return {};
}

// Apply per-feature normalization (Cohere-style): for each mel bin,
// compute mean and std across time, then normalize.
// mel is [num_mels, num_frames] row-major.
static void PerFeatureNormalize(float* mel, int num_mels, int num_frames) {
  constexpr float eps = 1e-5f;
  for (int m = 0; m < num_mels; ++m) {
    float* row = mel + m * num_frames;
    // Mean
    float sum = 0.0f;
    for (int t = 0; t < num_frames; ++t) sum += row[t];
    float mean = sum / num_frames;
    // Std
    float var_sum = 0.0f;
    for (int t = 0; t < num_frames; ++t) {
      float d = row[t] - mean;
      var_sum += d * d;
    }
    float std_val = std::sqrt(var_sum / (num_frames - 1)) + eps;
    // Normalize
    for (int t = 0; t < num_frames; ++t)
      row[t] = (row[t] - mean) / std_val;
  }
}

CohereProcessor::CohereProcessor(Config& config, const SessionInfo& session_info)
    : audio_features_type_{session_info.GetInputDataType(config.model.encoder.inputs.audio_features)} {
  config.AddMapping(std::string(Config::Defaults::AudioFeaturesName), config.model.encoder.inputs.audio_features);
  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.decoder.inputs.input_ids);
}

std::unique_ptr<NamedTensors> CohereProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  const auto* audios = payload.audios;
  if (!audios || !audios->audios_)
    throw std::runtime_error("No audios provided to process.");

  // Decode WAV to float PCM
  auto* raw = static_cast<RawAudiosAccess*>(audios->audios_.get());
  if (raw->num_audios_ == 0)
    throw std::runtime_error("No audio files loaded.");
  auto& wav_bytes = raw->audios_[0];
  auto samples = DecodeWav16(wav_bytes.data(), wav_bytes.size());
  if (samples.empty())
    throw std::runtime_error("Failed to decode WAV. Ensure 16-bit PCM WAV.");

  // Compute Cohere-compatible log-mel spectrogram via NeMo mel code
  nemo_mel::NemoMelConfig mel_cfg{};
  mel_cfg.num_mels = 128;
  mel_cfg.fft_size = 512;
  mel_cfg.hop_length = 160;
  mel_cfg.win_length = 400;
  mel_cfg.sample_rate = 16000;
  mel_cfg.preemph = 0.97f;
  mel_cfg.log_eps = 5.960464477539063e-08f;  // Cohere's log_zero_guard_value

  int num_frames = 0;
  auto mel_data = nemo_mel::NemoComputeLogMelBatch(
      samples.data(), samples.size(), mel_cfg, num_frames);

  // Apply per-feature normalization (Cohere-specific)
  PerFeatureNormalize(mel_data.data(), mel_cfg.num_mels, num_frames);

  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  // Create mel tensor [1, num_mels, num_frames]
  auto mel_shape = std::array<int64_t, 3>{1, mel_cfg.num_mels, num_frames};
  auto mel_tensor = OrtValue::CreateTensor<float>(allocator, std::span<int64_t>(mel_shape.data(), 3));
  std::memcpy(mel_tensor->GetTensorMutableData<float>(), mel_data.data(),
              mel_data.size() * sizeof(float));
  named_tensors->emplace(std::string(Config::Defaults::AudioFeaturesName),
                         std::make_shared<Tensor>(std::move(mel_tensor)));

  // Create mel_length tensor [1]
  auto ml_shape = std::array<int64_t, 1>{1};
  auto ml_tensor = OrtValue::CreateTensor<int64_t>(allocator, std::span<int64_t>(ml_shape.data(), 1));
  ml_tensor->GetTensorMutableData<int64_t>()[0] = num_frames;
  named_tensors->emplace("mel_length", std::make_shared<Tensor>(std::move(ml_tensor)));

  // Encode prompt tokens
  std::shared_ptr<Tensor> input_ids;
  if (!payload.prompt.empty()) {
    const char* prompt_cstr = payload.prompt.c_str();
    input_ids = tokenizer.EncodeBatch(std::span<const char*>(&prompt_cstr, 1));
  } else {
    input_ids = tokenizer.EncodeBatch(payload.prompts);
  }
  named_tensors->emplace(std::string(Config::Defaults::InputIdsName), input_ids);

  return named_tensors;
}

}  // namespace Generators
