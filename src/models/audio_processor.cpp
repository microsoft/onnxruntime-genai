// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"

#include <regex>

namespace Generators {

namespace {

std::unique_ptr<OrtValue> ProcessMel(ort_extensions::OrtxObjectPtr<OrtxTensor>& mel,
                                     ONNXTensorElementDataType expected_type, Ort::Allocator& allocator) {
  if (!(expected_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || expected_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)) {
    throw std::runtime_error("Expected input_features to be of type float or float16. Actual: " + std::to_string(expected_type));
  }

  const float* mel_data{};
  const int64_t* shape{};
  size_t num_dims;
  CheckResult(OrtxGetTensorDataFloat(mel.get(), &mel_data, &shape, &num_dims));
  std::span<const int64_t> shape_span(shape, num_dims);
  auto input_features_value = expected_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                                  ? OrtValue::CreateTensor<float>(allocator, shape_span)
                                  : OrtValue::CreateTensor<Ort::Float16_t>(allocator, shape_span);
  if (expected_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    std::copy(mel_data, mel_data + input_features_value->GetTensorTypeAndShapeInfo()->GetElementCount(),
              input_features_value->GetTensorMutableData<float>());
  } else {
    auto input_features_fp32 = OrtValue::CreateTensor<float>(
        allocator.GetInfo(),
        std::span<float>(const_cast<float*>(mel_data), input_features_value->GetTensorTypeAndShapeInfo()->GetElementCount()),
        shape_span);
    ConvertFp32ToFp16(allocator, *input_features_fp32, input_features_value, DeviceType::CPU, nullptr);
  }

  return input_features_value;
}

}  // namespace

std::unique_ptr<Audios> LoadAudios(const std::span<const char* const>& audio_paths) {
  for (const char* audio_path : audio_paths) {
    if (!fs::path(audio_path).exists()) {
      throw std::runtime_error("Audio path does not exist: " + std::string(audio_path));
    }
  }
  ort_extensions::OrtxObjectPtr<OrtxRawAudios> audios;
  CheckResult(OrtxLoadAudios(ort_extensions::ptr(audios), audio_paths.data(), audio_paths.size()));

  return std::make_unique<Audios>(std::move(audios), audio_paths.size());
}

AudioProcessor::AudioProcessor(Config& config, const SessionInfo& session_info)
    : input_features_type_{session_info.GetInputDataType(config.model.encoder_decoder_init.inputs.input_features)} {
  const std::string default_processor_file_name = "audio_processor_config.json";
  auto processor_config = (config.config_path / fs::path(default_processor_file_name)).string();
  processor_ = ort_extensions::OrtxObjectPtr<OrtxFeatureExtractor>(OrtxCreateSpeechFeatureExtractor, processor_config.c_str());

  config.AddMapping(std::string(Config::Defaults::InputFeaturesName), config.model.encoder_decoder_init.inputs.input_features);
}

std::unique_ptr<NamedTensors> AudioProcessor::Process(const Tokenizer& tokenizer, const Audios* audios,
                                                      const std::string& language, const std::string& task,
                                                      int32_t no_timestamps) const {
  if (!audios || !audios->audios_) {
    throw std::runtime_error("No audios provided to process.");
  }

  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  CheckResult(OrtxSpeechLogMel(processor_.get(), audios->audios_.get(), ort_extensions::ptr(result)));

  ort_extensions::OrtxObjectPtr<OrtxTensor> mel;
  CheckResult(OrtxTensorResultGetAt(result.get(), 0, ort_extensions::ptr(mel)));

  named_tensors->emplace(std::string(Config::Defaults::InputFeaturesName),
                         std::make_shared<Tensor>(ProcessMel(mel, input_features_type_, allocator)));

  constexpr auto start_of_transcript = "<|startoftranscript|>";
  const int32_t start_of_transcript_token_id = tokenizer.TokenToTokenId(start_of_transcript);
  const auto prompt_token_ids = tokenizer.GetDecoderPromptIds(audios->num_audios_, language, task, no_timestamps);

  const std::array<int64_t, 2> shape{static_cast<int64_t>(audios->num_audios_),
                                     static_cast<int64_t>(1U + prompt_token_ids.size())};
  auto decoder_input_ids = OrtValue::CreateTensor<int32_t>(allocator, shape);
  for (int64_t i = 0; i < audios->num_audios_; ++i) {
    auto decoder_input_ids_span = std::span<int32_t>(decoder_input_ids->GetTensorMutableData<int32_t>() + i * shape[1],
                                                     shape[1]);
    decoder_input_ids_span[0] = start_of_transcript_token_id;
    std::copy(prompt_token_ids.begin(), prompt_token_ids.end(), decoder_input_ids_span.data() + 1);
  }

  named_tensors->emplace(std::string(Config::Defaults::InputIdsName),
                         std::make_shared<Tensor>(std::move(decoder_input_ids)));

  return named_tensors;
}

}  // namespace Generators
