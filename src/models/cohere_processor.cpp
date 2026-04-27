// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "cohere_processor.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace Generators {

CohereProcessor::CohereProcessor(Config& config, const SessionInfo& session_info)
    : audio_features_type_{session_info.GetInputDataType(config.model.encoder.inputs.audio_features)},
      max_audio_clip_s_{config.model.max_audio_clip_s},
      overlap_chunk_s_{config.model.overlap_chunk_s},
      min_energy_window_samples_{config.model.min_energy_window_samples} {
  auto processor_config = (config.config_path / fs::path(config.model.speech.config_filename)).string();
  processor_ = ort_extensions::OrtxObjectPtr<OrtxFeatureExtractor>(OrtxCreateSpeechFeatureExtractor, processor_config.c_str());

  config.AddMapping(std::string(Config::Defaults::AudioFeaturesName), config.model.encoder.inputs.audio_features);
  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.decoder.inputs.input_ids);
}

// --- WAV decoding: parse raw WAV bytes to mono float32 PCM ---

static std::pair<std::vector<float>, int> ParseWavBytes(const uint8_t* data, size_t size) {
  if (size < 44 || std::memcmp(data, "RIFF", 4) != 0 || std::memcmp(data + 8, "WAVE", 4) != 0)
    throw std::runtime_error("CohereProcessor: not a valid WAV file");

  int16_t num_channels = 0;
  int32_t sample_rate = 0;
  int16_t bits_per_sample = 0;
  const uint8_t* audio_data = nullptr;
  size_t audio_bytes = 0;

  size_t pos = 12;
  while (pos + 8 <= size) {
    uint32_t chunk_sz;
    std::memcpy(&chunk_sz, data + pos + 4, 4);

    if (std::memcmp(data + pos, "fmt ", 4) == 0) {
      std::memcpy(&num_channels, data + pos + 10, 2);
      std::memcpy(&sample_rate, data + pos + 12, 4);
      std::memcpy(&bits_per_sample, data + pos + 22, 2);
    } else if (std::memcmp(data + pos, "data", 4) == 0) {
      audio_data = data + pos + 8;
      audio_bytes = std::min(static_cast<size_t>(chunk_sz), size - pos - 8);
    }
    pos += 8 + chunk_sz;
  }

  if (!audio_data || num_channels == 0 || sample_rate == 0)
    throw std::runtime_error("CohereProcessor: invalid WAV structure");

  size_t num_samples = audio_bytes / (bits_per_sample / 8) / num_channels;
  std::vector<float> samples(num_samples);

  if (bits_per_sample == 16) {
    const int16_t* raw = reinterpret_cast<const int16_t*>(audio_data);
    for (size_t i = 0; i < num_samples; ++i) {
      if (num_channels == 1) {
        samples[i] = raw[i] / 32768.0f;
      } else {
        float sum = 0.0f;
        for (int c = 0; c < num_channels; ++c)
          sum += raw[i * num_channels + c];
        samples[i] = (sum / num_channels) / 32768.0f;
      }
    }
  } else if (bits_per_sample == 32) {
    const int32_t* raw = reinterpret_cast<const int32_t*>(audio_data);
    for (size_t i = 0; i < num_samples; ++i) {
      if (num_channels == 1) {
        samples[i] = raw[i] / 2147483648.0f;
      } else {
        float sum = 0.0f;
        for (int c = 0; c < num_channels; ++c)
          sum += raw[i * num_channels + c] / 2147483648.0f;
        samples[i] = sum / num_channels;
      }
    }
  } else {
    throw std::runtime_error("CohereProcessor: unsupported bits_per_sample=" + std::to_string(bits_per_sample));
  }

  return {std::move(samples), sample_rate};
}

std::vector<std::pair<size_t, size_t>> CohereProcessor::SplitWaveformIntoChunks(
    const float* samples, size_t num_samples, int sample_rate) const {
  size_t chunk_size = std::max(size_t(1), static_cast<size_t>(max_audio_clip_s_ * sample_rate));
  size_t boundary_ctx = std::max(size_t(1), static_cast<size_t>(overlap_chunk_s_ * sample_rate));
  size_t min_window = static_cast<size_t>(min_energy_window_samples_);

  std::vector<std::pair<size_t, size_t>> chunks;

  if (num_samples <= chunk_size) {
    chunks.push_back({0, num_samples});
    return chunks;
  }

  size_t idx = 0;
  while (idx < num_samples) {
    if (idx + chunk_size >= num_samples) {
      chunks.push_back({idx, num_samples});
      break;
    }

    size_t search_start = std::max(idx, idx + chunk_size - boundary_ctx);
    size_t search_end = std::min(idx + chunk_size, num_samples);

    size_t split = idx + chunk_size;  // fallback
    if (search_end > search_start) {
      float min_energy = std::numeric_limits<float>::infinity();
      size_t quietest = search_start;

      size_t upper = (search_end - search_start > min_window) ? search_end - search_start - min_window : 0;
      for (size_t i = 0; i <= upper; i += min_window) {
        size_t w_start = search_start + i;
        size_t w_end = std::min(w_start + min_window, search_end);
        float energy = 0.0f;
        for (size_t j = w_start; j < w_end; ++j) {
          energy += samples[j] * samples[j];
        }
        energy = std::sqrt(energy / (w_end - w_start));
        if (energy < min_energy) {
          min_energy = energy;
          quietest = w_start;
        }
      }
      split = quietest;
    }

    split = std::max(idx + 1, std::min(split, num_samples));
    chunks.push_back({idx, split});
    idx = split;
  }

  return chunks;
}

std::vector<uint8_t> CohereProcessor::SamplesToWavBytes(const float* samples, size_t num_samples, int sample_rate) const {
  // WAV header: 44 bytes + data
  size_t data_size = num_samples * 2;  // 16-bit PCM
  size_t file_size = 44 + data_size;
  std::vector<uint8_t> wav(file_size);
  uint8_t* p = wav.data();

  // RIFF header
  std::memcpy(p, "RIFF", 4); p += 4;
  uint32_t chunk_size = static_cast<uint32_t>(file_size - 8);
  std::memcpy(p, &chunk_size, 4); p += 4;
  std::memcpy(p, "WAVE", 4); p += 4;

  // fmt subchunk
  std::memcpy(p, "fmt ", 4); p += 4;
  uint32_t fmt_size = 16; std::memcpy(p, &fmt_size, 4); p += 4;
  uint16_t audio_format = 1; std::memcpy(p, &audio_format, 2); p += 2;  // PCM
  uint16_t num_channels = 1; std::memcpy(p, &num_channels, 2); p += 2;
  uint32_t sr = static_cast<uint32_t>(sample_rate); std::memcpy(p, &sr, 4); p += 4;
  uint32_t byte_rate = sr * 2; std::memcpy(p, &byte_rate, 4); p += 4;
  uint16_t block_align = 2; std::memcpy(p, &block_align, 2); p += 2;
  uint16_t bits_per_sample = 16; std::memcpy(p, &bits_per_sample, 2); p += 2;

  // data subchunk
  std::memcpy(p, "data", 4); p += 4;
  uint32_t ds = static_cast<uint32_t>(data_size); std::memcpy(p, &ds, 4); p += 4;

  // Convert float32 -> int16
  int16_t* dst = reinterpret_cast<int16_t*>(p);
  for (size_t i = 0; i < num_samples; ++i) {
    float v = std::max(-1.0f, std::min(1.0f, samples[i]));
    dst[i] = static_cast<int16_t>(v * 32767.0f);
  }

  return wav;
}

std::pair<std::unique_ptr<OrtValue>, int64_t> CohereProcessor::ExtractMel(const std::vector<uint8_t>& wav_bytes) const {
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};

  // Create OrtxRawAudios from WAV bytes
  const void* data_ptrs[1] = {wav_bytes.data()};
  int64_t sizes[1] = {static_cast<int64_t>(wav_bytes.size())};
  ort_extensions::OrtxObjectPtr<OrtxRawAudios> chunk_audios;
  CheckResult(OrtxCreateRawAudios(chunk_audios.ToBeAssigned(), data_ptrs, sizes, 1));

  // Run mel extraction
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  CheckResult(OrtxFeatureExtraction(processor_.get(), chunk_audios.get(), result.ToBeAssigned()));

  ort_extensions::OrtxObjectPtr<OrtxTensor> mel;
  CheckResult(OrtxTensorResultGetAt(result.get(), 0, mel.ToBeAssigned()));

  const int64_t* shape_ptr = nullptr;
  size_t num_dims = 0;
  const void* data_ptr = nullptr;
  CheckResult(OrtxGetTensorData(mel.get(), &data_ptr, &shape_ptr, &num_dims));

  int64_t num_frames = shape_ptr[2];  // [1, num_mels, num_frames]

  // Copy to OrtValue
  auto ort_tensor = ProcessTensor<float>(mel.get(), allocator);
  return {std::move(ort_tensor), num_frames};
}

std::unique_ptr<NamedTensors> CohereProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  const auto* audios = payload.audios;
  if (!audios || !audios->audios_)
    throw std::runtime_error("No audios provided to process.");

  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  // Try waveform-level chunking if raw bytes are available
  bool have_raw = !audios->raw_bytes_.empty() && !audios->raw_bytes_[0].empty();

  if (have_raw) {
    // Parse WAV to get raw PCM samples
    auto [samples, sample_rate] = ParseWavBytes(audios->raw_bytes_[0].data(), audios->raw_bytes_[0].size());

    // Split waveform at energy boundaries
    auto chunk_ranges = SplitWaveformIntoChunks(samples.data(), samples.size(), sample_rate);

    // Process first chunk
    {
      auto [start, end] = chunk_ranges[0];
      auto wav = SamplesToWavBytes(samples.data() + start, end - start, sample_rate);
      auto [mel_tensor, mel_frames] = ExtractMel(wav);

      named_tensors->emplace(std::string(Config::Defaults::AudioFeaturesName),
                             std::make_shared<Tensor>(std::move(mel_tensor)));

      auto ml_shape = std::array<int64_t, 1>{1};
      auto ml_tensor = OrtValue::CreateTensor<int64_t>(allocator, std::span<int64_t>(ml_shape.data(), 1));
      ml_tensor->GetTensorMutableData<int64_t>()[0] = mel_frames;
      named_tensors->emplace("mel_length", std::make_shared<Tensor>(std::move(ml_tensor)));
    }

    // Process remaining chunks
    for (size_t i = 1; i < chunk_ranges.size(); ++i) {
      auto [start, end] = chunk_ranges[i];
      auto wav = SamplesToWavBytes(samples.data() + start, end - start, sample_rate);
      auto [mel_tensor, mel_frames] = ExtractMel(wav);

      named_tensors->emplace("cohere_chunk_" + std::to_string(i),
                             std::make_shared<Tensor>(std::move(mel_tensor)));

      auto ml_shape = std::array<int64_t, 1>{1};
      auto ml_tensor = OrtValue::CreateTensor<int64_t>(allocator, std::span<int64_t>(ml_shape.data(), 1));
      ml_tensor->GetTensorMutableData<int64_t>()[0] = mel_frames;
      named_tensors->emplace("cohere_chunk_mel_length_" + std::to_string(i),
                             std::make_shared<Tensor>(std::move(ml_tensor)));
    }

    // Store chunk count
    auto count_shape = std::array<int64_t, 1>{1};
    auto count_tensor = OrtValue::CreateTensor<int64_t>(allocator, std::span<int64_t>(count_shape.data(), 1));
    count_tensor->GetTensorMutableData<int64_t>()[0] = static_cast<int64_t>(chunk_ranges.size());
    named_tensors->emplace("cohere_chunk_count", std::make_shared<Tensor>(std::move(count_tensor)));
  } else {
    // Fallback: no raw audio access, run mel on full audio (no chunking)
    ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
    CheckResult(OrtxFeatureExtraction(processor_.get(), audios->audios_.get(), result.ToBeAssigned()));

    ort_extensions::OrtxObjectPtr<OrtxTensor> mel;
    CheckResult(OrtxTensorResultGetAt(result.get(), 0, mel.ToBeAssigned()));

    const int64_t* mel_shape_ptr = nullptr;
    size_t mel_num_dims = 0;
    const void* mel_data_ptr = nullptr;
    CheckResult(OrtxGetTensorData(mel.get(), &mel_data_ptr, &mel_shape_ptr, &mel_num_dims));
    int64_t num_frames = mel_shape_ptr[2];

    named_tensors->emplace(std::string(Config::Defaults::AudioFeaturesName),
                           std::make_shared<Tensor>(ProcessTensor<float>(mel.get(), allocator)));

    auto ml_shape = std::array<int64_t, 1>{1};
    auto ml_tensor = OrtValue::CreateTensor<int64_t>(allocator, std::span<int64_t>(ml_shape.data(), 1));
    ml_tensor->GetTensorMutableData<int64_t>()[0] = num_frames;
    named_tensors->emplace("mel_length", std::make_shared<Tensor>(std::move(ml_tensor)));

    // Single chunk
    auto count_shape = std::array<int64_t, 1>{1};
    auto count_tensor = OrtValue::CreateTensor<int64_t>(allocator, std::span<int64_t>(count_shape.data(), 1));
    count_tensor->GetTensorMutableData<int64_t>()[0] = 1;
    named_tensors->emplace("cohere_chunk_count", std::make_shared<Tensor>(std::move(count_tensor)));
  }

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
