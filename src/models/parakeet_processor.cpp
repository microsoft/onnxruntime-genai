// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "parakeet_processor.h"
#include "runner.hpp"
#include "c_api_utils.hpp"
#include "nemo_speech_features.hpp"
#include "nemo_mel_spectrogram.h"

namespace Generators {

ParakeetTdtProcessor::ParakeetTdtProcessor(Config& config, const SessionInfo& /*session_info*/)
    : config_{config} {}

std::unique_ptr<NamedTensors> ParakeetTdtProcessor::Process(const Tokenizer& /*tokenizer*/,
                                                            const Payload& payload) const {
  const auto& m = config_.model;
  const auto* audios = payload.audios;
  if (!audios || !audios->audios_) {
    throw std::runtime_error("ParakeetTdtProcessor::Process: no audio provided.");
  }
  if (audios->num_audios_ != 1) {
    throw std::runtime_error(
        "ParakeetTdtProcessor currently supports a single audio clip per call. Got: " +
        std::to_string(audios->num_audios_));
  }

  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  // 1. Decode audio file to float32 mono PCM at the model's sample rate with downmixing.
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> decoded;
  CheckResult(OrtxDecodeAudio(audios->audios_.get(), 0,
                              static_cast<int64_t>(m.sample_rate),
                              /*stereo_to_mono=*/1,
                              decoded.ToBeAssigned()));

  OrtxTensor* pcm_tensor = nullptr;
  CheckResult(OrtxTensorResultGetAt(decoded.get(), 0, &pcm_tensor));

  const float* pcm_data{};
  const int64_t* pcm_shape{};
  size_t pcm_dims = 0;
  CheckResult(OrtxGetTensorData(pcm_tensor, reinterpret_cast<const void**>(&pcm_data),
                                &pcm_shape, &pcm_dims));

  int64_t num_samples = 0;
  if (pcm_dims == 1) {
    num_samples = pcm_shape[0];
  } else if (pcm_dims == 2) {
    num_samples = pcm_shape[1];
  } else {
    throw std::runtime_error("Unexpected PCM tensor rank: " + std::to_string(pcm_dims));
  }

  // 2. Full-utterance mel calculation.
  nemo_mel::NemoMelConfig mel_cfg{};
  mel_cfg.num_mels = m.num_mels;
  mel_cfg.fft_size = m.fft_size;
  mel_cfg.hop_length = m.hop_length;
  mel_cfg.win_length = m.win_length;
  mel_cfg.sample_rate = m.sample_rate;
  mel_cfg.preemph = m.preemph;
  mel_cfg.log_eps = m.log_eps;

  nemo_mel::NemoStreamingMelExtractor mel_extractor(mel_cfg);
  auto [mel_data, num_frames] = mel_extractor.Process(pcm_data, static_cast<size_t>(num_samples));
  if (num_frames <= 0) {
    throw std::runtime_error("ParakeetTdtProcessor: audio is too short to produce mel frames.");
  }

  // 3. Per-feature mean/std normalization (NeMo `normalize_batch`).
  ort_extensions::PerFeatureNormalize norm_kernel;
  ort_extensions::AttrDict norm_attrs{
      {"eps", static_cast<double>(m.norm_eps)},
      {"feature_first", int64_t{1}},
  };
  if (auto status = norm_kernel.Init(norm_attrs); !status.IsOk()) {
    throw std::runtime_error(std::string("PerFeatureNormalize::Init failed: ") + status.Message());
  }

  std::vector<int64_t> mel_shape{m.num_mels, static_cast<int64_t>(num_frames)};
  ortc::Tensor<float> norm_in(mel_shape, mel_data.data());
  ortc::Tensor<float> norm_out(&ort_extensions::CppAllocator::Instance());
  if (auto status = norm_kernel.Compute(norm_in, norm_out); !status.IsOk()) {
    throw std::runtime_error(std::string("PerFeatureNormalize::Compute failed: ") + status.Message());
  }

  // 4. Package the normalized mel as the model input tensor.
  auto mel_value = OrtValue::CreateTensor<float>(
      allocator, std::vector<int64_t>{1, m.num_mels, static_cast<int64_t>(num_frames)});
  std::memcpy(mel_value->GetTensorMutableData<float>(), norm_out.Data(),
              static_cast<size_t>(m.num_mels) * num_frames * sizeof(float));
  named_tensors->emplace(std::string(Config::Defaults::AudioFeaturesName),
                         std::make_shared<Tensor>(std::move(mel_value)));

  return named_tensors;
}

}  // namespace Generators
