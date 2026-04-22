// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "cohere_processor.h"

#include <cstring>

namespace Generators {

CohereProcessor::CohereProcessor(Config& config, const SessionInfo& session_info)
    : audio_features_type_{session_info.GetInputDataType(config.model.encoder.inputs.audio_features)} {
  auto processor_config = (config.config_path / fs::path(config.model.speech.config_filename)).string();
  processor_ = ort_extensions::OrtxObjectPtr<OrtxFeatureExtractor>(OrtxCreateSpeechFeatureExtractor, processor_config.c_str());

  config.AddMapping(std::string(Config::Defaults::AudioFeaturesName), config.model.encoder.inputs.audio_features);
  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.decoder.inputs.input_ids);
}

std::unique_ptr<NamedTensors> CohereProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  const auto* audios = payload.audios;
  if (!audios || !audios->audios_)
    throw std::runtime_error("No audios provided to process.");

  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  // Run the full pipeline: AudioDecoder → NemoLogMel → PerFeatureNormalize
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  CheckResult(OrtxFeatureExtraction(processor_.get(), audios->audios_.get(), result.ToBeAssigned()));

  ort_extensions::OrtxObjectPtr<OrtxTensor> mel;
  CheckResult(OrtxTensorResultGetAt(result.get(), 0, mel.ToBeAssigned()));

  // Get mel shape to extract num_frames for mel_length tensor
  const int64_t* mel_shape_ptr = nullptr;
  size_t mel_num_dims = 0;
  const void* mel_data_ptr = nullptr;
  CheckResult(OrtxGetTensorData(mel.get(), &mel_data_ptr, &mel_shape_ptr, &mel_num_dims));
  int64_t num_frames = mel_shape_ptr[2];  // shape is [1, num_mels, num_frames]

  // Create audio features tensor
  named_tensors->emplace(std::string(Config::Defaults::AudioFeaturesName),
                         std::make_shared<Tensor>(ProcessTensor<float>(mel.get(), allocator)));

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
