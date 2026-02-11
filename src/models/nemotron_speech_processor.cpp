// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "nemotron_speech_processor.h"
#include <filesystem>

namespace Generators {

NemotronSpeechProcessor::NemotronSpeechProcessor(Config& config, const SessionInfo& session_info)
    : audio_features_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT} {
  // For Nemotron, the encoder input is "processed_signal" (log-mel spectrogram)
  std::string encoder_input = config.model.encoder.inputs.audio_features;
  if (encoder_input.empty()) {
    encoder_input = "processed_signal";
  }
  config.AddMapping(std::string(Config::Defaults::AudioFeaturesName), encoder_input);

  // Load the speech feature extractor config (same as Whisper)
  auto processor_config = (config.config_path / fs::path(config.model.speech.config_filename)).string();
  if (std::filesystem::exists(processor_config)) {
    processor_ = ort_extensions::OrtxObjectPtr<OrtxFeatureExtractor>(
        OrtxCreateSpeechFeatureExtractor, processor_config.c_str());
  }
  // If no config exists, the processor_ stays null and feature extraction
  // will be handled by NemotronSpeechState::RunEncoder via ORT Extensions.
}

std::unique_ptr<NamedTensors> NemotronSpeechProcessor::Process(const Tokenizer& tokenizer,
                                                                const Payload& payload) const {
  const auto* audios = payload.audios;
  if (!audios || !audios->audios_) {
    throw std::runtime_error("No audios provided to process.");
  }

  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  if (processor_) {
    // Use ORT extensions feature extractor (same pattern as WhisperProcessor)
    ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
    CheckResult(OrtxFeatureExtraction(processor_.get(), audios->audios_.get(), result.ToBeAssigned()));

    ort_extensions::OrtxObjectPtr<OrtxTensor> mel;
    CheckResult(OrtxTensorResultGetAt(result.get(), 0, mel.ToBeAssigned()));

    named_tensors->emplace(std::string(Config::Defaults::AudioFeaturesName),
                           std::make_shared<Tensor>(ProcessTensor<float>(mel.get(), allocator)));
  } else {
    // No feature extractor available — pass raw audio as-is.
    // The StreamingASR class will handle mel extraction internally.
    // Create a placeholder tensor so the pipeline doesn't fail.
    auto shape = std::array<int64_t, 2>{1, 1};
    auto placeholder = OrtValue::CreateTensor(allocator, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    *placeholder->GetTensorMutableData<float>() = 0.0f;
    named_tensors->emplace(std::string(Config::Defaults::AudioFeaturesName),
                           std::make_shared<Tensor>(std::move(placeholder)));
  }

  return named_tensors;
}

}  // namespace Generators
