// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "cohere_processor.h"
#include "speech_features.hpp"
#include "c_api_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace Generators {

CohereProcessor::CohereProcessor(Config& config, const SessionInfo& session_info)
    : audio_features_type_{session_info.GetInputDataType(config.model.encoder.inputs.audio_features)},
      max_audio_clip_s_{config.model.max_audio_clip_s},
      overlap_chunk_s_{config.model.overlap_chunk_s} {
  mel_cfg_.num_mels    = config.model.num_mels;
  mel_cfg_.fft_size    = config.model.fft_size;
  mel_cfg_.hop_length  = config.model.hop_length;
  mel_cfg_.win_length  = config.model.win_length;
  mel_cfg_.sample_rate = config.model.sample_rate;
  mel_cfg_.preemph     = config.model.preemph;
  mel_cfg_.log_eps     = config.model.log_eps;
  norm_eps_ = config.model.norm_eps;

  config.AddMapping(std::string(Config::Defaults::AudioFeaturesName), config.model.encoder.inputs.audio_features);
  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.decoder.inputs.input_ids);
}

// Decode audio to PCM via OrtxDecodeAudio in extensions
static std::pair<const float*, size_t> GetDecodedPCM(
    OrtxRawAudios* raw_audios, size_t index, int target_sample_rate,
    ort_extensions::OrtxObjectPtr<OrtxTensorResult>& decode_result_holder,
    int& out_sample_rate) {
  OrtxTensorResult* decode_result = nullptr;
  // Pass target_sample_rate so ortx resamples on decode.
  CheckResult(OrtxDecodeAudio(raw_audios, index, target_sample_rate, &decode_result));
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

// Waveform splitting with ENERGY-BASED SILENCE BOUNDARIES (no overlap)
//
// Mirrors CohereAsrFeatureExtractor._split_audio_chunks_energy from the HF
// transformers reference implementation. Each chunk is at most
// max_audio_clip_s_ seconds; the cut point is snapped to the quietest
// `min_energy_window_samples`-sized window inside the last
// `overlap_chunk_s_` seconds of the chunk. Chunks do NOT overlap, which
// matches the model's training-time chunker (whole utterances per chunk),
// so the per-chunk decoded text can be plain space-joined without dedup.
static size_t FindEnergySplitPoint(const float* samples, size_t start, size_t end,
                                   size_t min_window) {
  if (end <= start || end - start <= min_window) {
    return (start + end) / 2;
  }
  const size_t upper = (end - start) - min_window;
  size_t quietest = start;
  float min_energy = std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < upper; i += min_window) {
    double sum_sq = 0.0;
    const float* w = samples + start + i;
    for (size_t k = 0; k < min_window; ++k) {
      sum_sq += static_cast<double>(w[k]) * w[k];
    }
    const float energy = static_cast<float>(std::sqrt(sum_sq / min_window));
    if (energy < min_energy) {
      min_energy = energy;
      quietest = start + i;
    }
  }
  return quietest;
}

std::vector<std::pair<size_t, size_t>> CohereProcessor::SplitWaveformIntoChunks(
    const float* samples, size_t num_samples, int sample_rate) const {
  const size_t chunk_size = std::max<size_t>(
      1, static_cast<size_t>(std::round(max_audio_clip_s_ * sample_rate)));
  const size_t boundary_ctx = std::max<size_t>(
      1, static_cast<size_t>(std::round(overlap_chunk_s_ * sample_rate)));
  const size_t min_window = static_cast<size_t>(sample_rate) / 10;  // 100 ms

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
    const size_t search_start = (idx + chunk_size > boundary_ctx)
                                    ? (idx + chunk_size - boundary_ctx)
                                    : idx;
    const size_t search_end = std::min(idx + chunk_size, num_samples);
    size_t split = (search_end > search_start)
                       ? FindEnergySplitPoint(samples, search_start, search_end, min_window)
                       : (idx + chunk_size);
    split = std::max(idx + 1, std::min(split, num_samples));
    chunks.push_back({idx, split});
    idx = split;
  }
  return chunks;
}

// Compute mel + normalize from PCM float32.
std::pair<std::unique_ptr<OrtValue>, int64_t> CohereProcessor::ComputeMelFromPCM(
    const float* samples, size_t num_samples) const {
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};

  // Step 1: PCM -> log-mel spectrogram (onnxruntime-extensions kernel).
  int num_frames = 0;
  auto mel_data = nemo_mel::NemoComputeLogMelBatch(samples, num_samples, mel_cfg_, num_frames);
  const int64_t num_mels = mel_cfg_.num_mels;

  // Step 2: per-feature normalize with feature-first normalization.
  ort_extensions::PerFeatureNormalize norm_kernel;
  ort_extensions::AttrDict norm_attrs{
      {"eps", static_cast<double>(norm_eps_)},
      {"feature_first", int64_t{1}},
  };
  if (auto status = norm_kernel.Init(norm_attrs); !status.IsOk()) {
    throw std::runtime_error(std::string("PerFeatureNormalize::Init failed: ") + status.Message());
  }

  std::vector<int64_t> mel_shape{num_mels, static_cast<int64_t>(num_frames)};
  ortc::Tensor<float> norm_in(mel_shape, mel_data.data());
  ortc::Tensor<float> norm_out(&ort_extensions::CppAllocator::Instance());
  if (auto status = norm_kernel.Compute(norm_in, norm_out); !status.IsOk()) {
    throw std::runtime_error(std::string("PerFeatureNormalize::Compute failed: ") + status.Message());
  }

  // Step 3: copy normalized data into an OrtValue [1, num_mels, num_frames].
  auto shape = std::array<int64_t, 3>{1, num_mels, static_cast<int64_t>(num_frames)};
  auto tensor = OrtValue::CreateTensor<float>(allocator, std::span<int64_t>(shape.data(), 3));
  std::memcpy(tensor->GetTensorMutableData<float>(),
              norm_out.Data(),
              static_cast<size_t>(num_mels) * num_frames * sizeof(float));

  return {std::move(tensor), static_cast<int64_t>(num_frames)};
}

// --- Main Process ---

std::unique_ptr<NamedTensors> CohereProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  const auto* audios = payload.audios;
  if (!audios || !audios->audios_)
    throw std::runtime_error("No audios provided to process.");

  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  // Decode audio to PCM, resampled to the model's expected sample rate.
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> decode_result;
  int sample_rate = 0;
  auto [pcm_data, num_samples] = GetDecodedPCM(audios->audios_.get(), 0, mel_cfg_.sample_rate, decode_result, sample_rate);

  // Split waveform at energy-based silence boundaries (no overlap).
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
