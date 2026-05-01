// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "cohere_processor.h"
#include "speech_features.hpp"

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

// --- Decode audio to PCM via OrtxDecodeAudio ---

static std::pair<const float*, size_t> GetDecodedPCM(
    OrtxRawAudios* raw_audios, size_t index,
    ort_extensions::OrtxObjectPtr<OrtxTensorResult>& decode_result_holder,
    int& out_sample_rate) {
  OrtxTensorResult* decode_result = nullptr;
  CheckResult(OrtxDecodeAudio(raw_audios, index, 0 /* native rate */, &decode_result));
  decode_result_holder.reset(decode_result);

  ort_extensions::OrtxObjectPtr<OrtxTensor> pcm_tensor;
  CheckResult(OrtxTensorResultGetAt(decode_result, 0, pcm_tensor.ToBeAssigned()));

  ort_extensions::OrtxObjectPtr<OrtxTensor> sr_tensor;
  CheckResult(OrtxTensorResultGetAt(decode_result, 1, sr_tensor.ToBeAssigned()));

  const void* pcm_data = nullptr;
  const int64_t* pcm_shape = nullptr;
  size_t pcm_dims = 0;
  CheckResult(OrtxGetTensorData(pcm_tensor.get(), &pcm_data, &pcm_shape, &pcm_dims));

  const void* sr_data = nullptr;
  const int64_t* sr_shape = nullptr;
  size_t sr_dims = 0;
  CheckResult(OrtxGetTensorData(sr_tensor.get(), &sr_data, &sr_shape, &sr_dims));

  out_sample_rate = static_cast<int>(*static_cast<const int64_t*>(sr_data));

  size_t num_samples = 1;
  for (size_t d = 0; d < pcm_dims; ++d) num_samples *= pcm_shape[d];

  return {static_cast<const float*>(pcm_data), num_samples};
}

// --- Waveform splitting with FIXED OVERLAP between chunks ---
//
// The model's training-time chunker uses energy-based, non-overlapping splits.
// That produces seam artifacts at chunk boundaries (e.g. "...UK. Come from..."
// when a sentence is split mid-utterance) because each chunk's decoder runs
// independently and naturally terminates with sentence-end punctuation.
//
// We instead emit overlapping windows of length max_audio_clip_s_ stepping by
// (max_audio_clip_s_ - overlap_chunk_s_). The same audio (and hence the same
// words) appears at the tail of chunk N and the head of chunk N+1; the
// downstream merge step (CohereState::GetJoinedChunkText) does word-level
// dedup so the final transcript is seam-free.
std::vector<std::pair<size_t, size_t>> CohereProcessor::SplitWaveformIntoChunks(
    const float* /*samples*/, size_t num_samples, int sample_rate) const {
  size_t chunk_size = std::max(size_t(1), static_cast<size_t>(max_audio_clip_s_ * sample_rate));
  size_t overlap = static_cast<size_t>(overlap_chunk_s_ * sample_rate);
  if (overlap >= chunk_size) overlap = chunk_size / 2;
  size_t step = chunk_size - overlap;
  if (step == 0) step = chunk_size;

  std::vector<std::pair<size_t, size_t>> chunks;

  if (num_samples <= chunk_size) {
    chunks.push_back({0, num_samples});
    return chunks;
  }

  size_t idx = 0;
  while (idx < num_samples) {
    size_t end = std::min(idx + chunk_size, num_samples);
    chunks.push_back({idx, end});
    if (end == num_samples) break;
    idx += step;
  }
  return chunks;
}

// --- Compute mel + normalize from PCM float32 ---

std::pair<std::unique_ptr<OrtValue>, int64_t> CohereProcessor::ComputeMelFromPCM(
    const float* samples, size_t num_samples) const {
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};

  // NemoLogMel: PCM -> log-mel spectrogram (via onnxruntime-extensions)
  int num_frames = 0;
  auto mel_data = nemo_mel::NemoComputeLogMelBatch(samples, num_samples, mel_cfg_, num_frames);

  int64_t num_mels = mel_cfg_.num_mels;

  // PerFeatureNormalize: per-row mean/std normalization
  // Matches ort_extensions::PerFeatureNormalize with feature_first=1, eps=norm_eps_
  if (num_frames > 1) {
    for (int64_t f = 0; f < num_mels; ++f) {
      float* row = mel_data.data() + f * num_frames;
      float sum = 0.0f;
      for (int t = 0; t < num_frames; ++t) sum += row[t];
      float mean = sum / num_frames;

      float var_sum = 0.0f;
      for (int t = 0; t < num_frames; ++t) {
        float d = row[t] - mean;
        var_sum += d * d;
      }
      float std_val = std::sqrt(var_sum / (num_frames - 1)) + norm_eps_;

      for (int t = 0; t < num_frames; ++t)
        row[t] = (row[t] - mean) / std_val;
    }
  } else if (num_frames == 1) {
    std::fill(mel_data.begin(), mel_data.end(), 0.0f);
  }

  // Create OrtValue [1, num_mels, num_frames]
  auto shape = std::array<int64_t, 3>{1, num_mels, num_frames};
  auto tensor = OrtValue::CreateTensor<float>(allocator, std::span<int64_t>(shape.data(), 3));
  std::memcpy(tensor->GetTensorMutableData<float>(), mel_data.data(), mel_data.size() * sizeof(float));

  return {std::move(tensor), static_cast<int64_t>(num_frames)};
}

// --- Main Process ---

std::unique_ptr<NamedTensors> CohereProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  const auto* audios = payload.audios;
  if (!audios || !audios->audios_)
    throw std::runtime_error("No audios provided to process.");

  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  // Decode audio to PCM
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> decode_result;
  int sample_rate = 0;
  auto [pcm_data, num_samples] = GetDecodedPCM(audios->audios_.get(), 0, decode_result, sample_rate);

  // Split waveform at energy boundaries
  auto chunk_ranges = SplitWaveformIntoChunks(pcm_data, num_samples, sample_rate);

  // Compute mel for first chunk
  {
    auto [start, end] = chunk_ranges[0];
    auto [mel_tensor, mel_frames] = ComputeMelFromPCM(pcm_data + start, end - start);

    named_tensors->emplace(std::string(Config::Defaults::AudioFeaturesName),
                           std::make_shared<Tensor>(std::move(mel_tensor)));

    auto ml_shape = std::array<int64_t, 1>{1};
    auto ml_tensor = OrtValue::CreateTensor<int64_t>(allocator, std::span<int64_t>(ml_shape.data(), 1));
    ml_tensor->GetTensorMutableData<int64_t>()[0] = mel_frames;
    named_tensors->emplace("mel_length", std::make_shared<Tensor>(std::move(ml_tensor)));
  }

  // Compute mel for remaining chunks
  for (size_t i = 1; i < chunk_ranges.size(); ++i) {
    auto [start, end] = chunk_ranges[i];
    auto [mel_tensor, mel_frames] = ComputeMelFromPCM(pcm_data + start, end - start);

    named_tensors->emplace("cohere_chunk_" + std::to_string(i),
                           std::make_shared<Tensor>(std::move(mel_tensor)));

    auto ml_shape = std::array<int64_t, 1>{1};
    auto ml_tensor = OrtValue::CreateTensor<int64_t>(allocator, std::span<int64_t>(ml_shape.data(), 1));
    ml_tensor->GetTensorMutableData<int64_t>()[0] = mel_frames;
    named_tensors->emplace("cohere_chunk_mel_length_" + std::to_string(i),
                           std::make_shared<Tensor>(std::move(ml_tensor)));
  }

  // Store chunk count
  {
    auto count_shape = std::array<int64_t, 1>{1};
    auto count_tensor = OrtValue::CreateTensor<int64_t>(allocator, std::span<int64_t>(count_shape.data(), 1));
    count_tensor->GetTensorMutableData<int64_t>()[0] = static_cast<int64_t>(chunk_ranges.size());
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
