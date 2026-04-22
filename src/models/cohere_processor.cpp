// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "cohere_processor.h"
#include "nemo_mel_spectrogram.h"

#include <cstring>
#include <numeric>

namespace Generators {

CohereProcessor::CohereProcessor(Config& config, const SessionInfo& session_info)
    : audio_features_type_{session_info.GetInputDataType(config.model.encoder.inputs.audio_features)} {
  config.AddMapping(std::string(Config::Defaults::AudioFeaturesName), config.model.encoder.inputs.audio_features);
  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.decoder.inputs.input_ids);

  // Read mel spectrogram config from genai_config.json
  mel_cfg_.num_mels = config.model.num_mels;
  mel_cfg_.fft_size = config.model.fft_size;
  mel_cfg_.hop_length = config.model.hop_length;
  mel_cfg_.win_length = config.model.win_length;
  mel_cfg_.sample_rate = config.model.sample_rate;
  mel_cfg_.preemph = config.model.preemph;
  mel_cfg_.log_eps = config.model.log_eps;
}

std::unique_ptr<NamedTensors> CohereProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  const auto* audios = payload.audios;
  if (!audios || !audios->audios_)
    throw std::runtime_error("No audios provided to process.");

  // Decode audio using ort-extensions (handles WAV/MP3/FLAC, resampling, stereo→mono)
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> decode_result;
  CheckResult(OrtxDecodeAudio(audios->audios_.get(), 0, /*target_sample_rate=*/16000, decode_result.ToBeAssigned()));

  ort_extensions::OrtxObjectPtr<OrtxTensor> pcm_tensor;
  CheckResult(OrtxTensorResultGetAt(decode_result.get(), 0, pcm_tensor.ToBeAssigned()));

  // Get PCM data pointer and length
  const float* pcm_data = nullptr;
  const int64_t* pcm_shape = nullptr;
  size_t pcm_num_dims = 0;
  CheckResult(OrtxGetTensorData(pcm_tensor.get(), reinterpret_cast<const void**>(&pcm_data), &pcm_shape, &pcm_num_dims));
  size_t num_samples = static_cast<size_t>(pcm_shape[1]);  // shape is [1, num_samples]

  // Compute log-mel spectrogram via NeMo mel code (params from genai_config.json)
  int num_frames = 0;
  auto mel_data = nemo_mel::NemoComputeLogMelBatch(
      pcm_data, num_samples, mel_cfg_, num_frames);

  // Apply per-feature normalization (via ort-extensions)
  CheckResult(OrtxPerFeatureNormalize(mel_data.data(), mel_cfg_.num_mels, num_frames, 1e-5f));

  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  // Create mel tensor [1, num_mels, num_frames]
  auto mel_shape = std::array<int64_t, 3>{1, mel_cfg_.num_mels, num_frames};
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
