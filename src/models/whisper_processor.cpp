// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"

namespace Generators {

WhisperProcessor::WhisperProcessor(Config& config, const SessionInfo& session_info)
    : input_features_type_{session_info.GetInputDataType(config.model.encoder_decoder_init.inputs.input_features)} {
  auto processor_config = (config.config_path / fs::path(config.model.speech.config_filename)).string();
  processor_ = ort_extensions::OrtxObjectPtr<OrtxFeatureExtractor>(OrtxCreateSpeechFeatureExtractor, processor_config.c_str());

  config.AddMapping(std::string(Config::Defaults::InputFeaturesName), config.model.encoder_decoder_init.inputs.input_features);
}

std::unique_ptr<NamedTensors> WhisperProcessor::Process([[maybe_unused]] const Tokenizer& tokenizer, const Payload& payload) const {
  const auto* audios = payload.audios;
  if (!audios || !audios->audios_) {
    throw std::runtime_error("No audios provided to process.");
  }

  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  CheckResult(OrtxSpeechLogMel(processor_.get(), audios->audios_.get(), result.ToBeAssigned()));

  ort_extensions::OrtxObjectPtr<OrtxTensor> mel;
  CheckResult(OrtxTensorResultGetAt(result.get(), 0, mel.ToBeAssigned()));

  if (input_features_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    named_tensors->emplace(std::string(Config::Defaults::InputFeaturesName),
                           std::make_shared<Tensor>(ProcessTensor<float>(mel.get(), allocator)));
  } else {
    named_tensors->emplace(std::string(Config::Defaults::InputFeaturesName),
                           std::make_shared<Tensor>(ProcessTensor<Ort::Float16_t>(mel.get(), allocator)));
  }

  return named_tensors;
}

}  // namespace Generators
