// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "cohere_processor.h"

#include <cstring>

namespace Generators {

// Mirror of ort-extensions internal RawAudiosObject layout to access raw audio bytes.
// AudioRawData = std::vector<std::byte> (defined in speech_extractor.h).
struct RawAudiosAccess : ort_extensions::OrtxObjectImpl {
  std::unique_ptr<ort_extensions::AudioRawData[]> audios_;
  size_t num_audios_{};
};

// Decode 16-bit PCM WAV file bytes to float32 samples.
// Returns empty vector on failure.
static std::vector<float> DecodeWav16(const std::byte* data, size_t size) {
  if (size < 44) return {};

  auto u8 = reinterpret_cast<const uint8_t*>(data);

  // Verify RIFF/WAVE header
  if (std::memcmp(u8, "RIFF", 4) != 0 || std::memcmp(u8 + 8, "WAVE", 4) != 0)
    return {};

  // Find "data" chunk
  size_t pos = 12;
  while (pos + 8 <= size) {
    uint32_t chunk_size = u8[pos + 4] | (u8[pos + 5] << 8) | (u8[pos + 6] << 16) | (u8[pos + 7] << 24);
    if (std::memcmp(u8 + pos, "data", 4) == 0) {
      size_t data_start = pos + 8;
      size_t data_bytes = std::min(static_cast<size_t>(chunk_size), size - data_start);
      size_t num_samples = data_bytes / 2;

      std::vector<float> samples(num_samples);
      auto pcm16 = reinterpret_cast<const int16_t*>(u8 + data_start);
      for (size_t i = 0; i < num_samples; ++i) {
        samples[i] = static_cast<float>(pcm16[i]) / 32768.0f;
      }
      return samples;
    }
    pos += 8 + chunk_size;
    if (chunk_size & 1) pos++;  // Chunks are 2-byte aligned
  }
  return {};
}

CohereProcessor::CohereProcessor(Config& config, const SessionInfo& session_info)
    : audio_features_type_{session_info.GetInputDataType(config.model.encoder.inputs.audio_features)} {
  config.AddMapping(std::string(Config::Defaults::AudioFeaturesName), config.model.encoder.inputs.audio_features);
  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.decoder.inputs.input_ids);
}

std::unique_ptr<NamedTensors> CohereProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  const auto* audios = payload.audios;
  if (!audios || !audios->audios_) {
    throw std::runtime_error("No audios provided to process.");
  }

  // Access raw audio bytes from the Audios object
  auto* raw = static_cast<RawAudiosAccess*>(audios->audios_.get());
  if (raw->num_audios_ == 0) {
    throw std::runtime_error("No audio files loaded.");
  }

  // Decode WAV to float PCM (batch_size=1 for now)
  auto& wav_bytes = raw->audios_[0];
  auto samples = DecodeWav16(wav_bytes.data(), wav_bytes.size());
  if (samples.empty()) {
    throw std::runtime_error("Failed to decode WAV file. Ensure it is 16-bit PCM WAV format.");
  }

  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  // Create audio tensor [1, num_samples]
  auto shape = std::array<int64_t, 2>{1, static_cast<int64_t>(samples.size())};
  auto audio_tensor = OrtValue::CreateTensor<float>(allocator, std::span<int64_t>(shape.data(), 2));
  std::copy(samples.begin(), samples.end(), audio_tensor->GetTensorMutableData<float>());

  named_tensors->emplace(std::string(Config::Defaults::AudioFeaturesName),
                         std::make_shared<Tensor>(std::move(audio_tensor)));

  // Encode prompt tokens — handle both single prompt and batch prompts
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
