// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "parakeet_processor.h"

namespace Generators {

ParakeetProcessor::ParakeetProcessor(Config& config, const SessionInfo& /*session_info*/) {
  sample_rate_ = config.model.sample_rate;
  decoder_start_token_id_ = static_cast<int32_t>(config.model.decoder_start_token_id);
}

std::unique_ptr<NamedTensors> ParakeetProcessor::Process(const Tokenizer& /*tokenizer*/,
                                                          const Payload& payload) const {
  const auto* audios = payload.audios;
  if (!audios || !audios->audios_) {
    throw std::runtime_error("ParakeetProcessor::Process: no audio provided.");
  }
  if (audios->num_audios_ != 1) {
    throw std::runtime_error(
        "ParakeetProcessor currently supports a single audio clip per call. Got: " +
        std::to_string(audios->num_audios_));
  }

  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  // ── Decode the audio file(s) to float32 mono PCM at the model's rate ──
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> decoded;
  CheckResult(OrtxDecodeAudio(audios->audios_.get(), 0,
                              static_cast<int64_t>(sample_rate_),
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
    // [channels, samples] — stereo_to_mono should have collapsed this to 1.
    num_samples = pcm_shape[1];
  } else {
    throw std::runtime_error("Unexpected PCM tensor rank: " + std::to_string(pcm_dims));
  }

  auto pcm_value = OrtValue::CreateTensor<float>(allocator,
                                                  std::vector<int64_t>{1, num_samples});
  std::memcpy(pcm_value->GetTensorMutableData<float>(), pcm_data,
              static_cast<size_t>(num_samples) * sizeof(float));
  named_tensors->emplace("audio_pcm", std::make_shared<Tensor>(std::move(pcm_value)));

  // ── Insert a single placeholder input id so the Generator has a sequence
  //    to anchor on. The user is expected to slice tokens[1:] when decoding.
  auto ids_value = OrtValue::CreateTensor<int32_t>(allocator,
                                                    std::vector<int64_t>{1, 1});
  *ids_value->GetTensorMutableData<int32_t>() = decoder_start_token_id_;
  named_tensors->emplace(std::string(Config::Defaults::InputIdsName),
                         std::make_shared<Tensor>(std::move(ids_value)));

  return named_tensors;
}

}  // namespace Generators
