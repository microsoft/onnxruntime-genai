// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"

#include <regex>

namespace Generators {

namespace {

std::tuple<std::unique_ptr<OrtValue>, std::unique_ptr<OrtValue>>
ProcessImageAudioPrompt(const Generators::Tokenizer& tokenizer, const std::string& prompt,
                        OrtxTensor* num_img_tokens, OrtxTensor* audio_sizes,
                        Ort::Allocator& allocator) {
  const int64_t* num_img_tokens_data{};
  int64_t num_images{};
  if (num_img_tokens) {
    const int64_t* num_img_tokens_shape{};
    size_t num_img_tokens_num_dims;
    CheckResult(OrtxGetTensorData(num_img_tokens, reinterpret_cast<const void**>(&num_img_tokens_data),
                                  &num_img_tokens_shape, &num_img_tokens_num_dims));
    num_images = std::accumulate(num_img_tokens_shape,
                                 num_img_tokens_shape + num_img_tokens_num_dims,
                                 1LL, std::multiplies<int64_t>());
  }

  const float* audio_sizes_data{};
  int64_t num_audios{};
  if (audio_sizes) {
    const int64_t* audio_sizes_shape{};
    size_t audio_sizes_num_dims;
    CheckResult(OrtxGetTensorData(audio_sizes, reinterpret_cast<const void**>(&audio_sizes_data),
                                  &audio_sizes_shape, &audio_sizes_num_dims));
    num_audios = std::accumulate(audio_sizes_shape,
                                 audio_sizes_shape + audio_sizes_num_dims,
                                 1LL, std::multiplies<int64_t>());
  }

  std::unique_ptr<OrtValue> audio_projection_mode_value = OrtValue::CreateTensor<int64_t>(allocator, std::vector<int64_t>({1}));
  if (num_images == 0 && num_audios == 0) {
    audio_projection_mode_value->GetTensorMutableData<int64_t>()[0] = 0;  // Language
  } else if (num_audios == 0) {
    audio_projection_mode_value->GetTensorMutableData<int64_t>()[0] = 1;  // Vision, language
  } else if (num_images == 0) {
    audio_projection_mode_value->GetTensorMutableData<int64_t>()[0] = 2;  // Speech, language
  } else {
    audio_projection_mode_value->GetTensorMutableData<int64_t>()[0] = 3;  // Vision, speech, language
  }

  const std::regex image_pattern("<\\|image_\\d+\\|>");
  const std::regex audio_pattern("<\\|audio_\\d+\\|>");
  std::string processed_prompt = std::regex_replace(prompt, image_pattern, "<|endoftext10|>");
  processed_prompt = std::regex_replace(processed_prompt, audio_pattern, "<|endoftext11|>");

  const std::vector<int32_t> input_ids = tokenizer.Encode(processed_prompt.c_str());
  std::vector<int32_t> processed_input_ids;

  constexpr int32_t image_special_token_id = 200010;
  constexpr int32_t audio_special_token_id = 200011;
  size_t image_idx{0U}, audio_idx{0U};
  for (const auto token : input_ids) {
    if (token == image_special_token_id) {
      if (static_cast<int64_t>(image_idx) >= num_images) {
        throw std::runtime_error("Number of image tokens exceeds the number of images. Please fix the prompt.");
      }

      for (int64_t j = 0; j < num_img_tokens_data[image_idx]; ++j) {
        processed_input_ids.push_back(token);
      }
      image_idx++;
    } else if (token == audio_special_token_id) {
      if (static_cast<int64_t>(audio_idx) >= num_audios) {
        throw std::runtime_error("Number of audio tokens exceeds the number of audios. Please fix the prompt.");
      }

      for (int64_t j = 0; j < static_cast<int64_t>(audio_sizes_data[audio_idx] + 0.5f); ++j) {
        processed_input_ids.push_back(token);
      }
      audio_idx++;
    } else {
      processed_input_ids.push_back(token);
    }
  }

  if (static_cast<int64_t>(image_idx) != num_images) {
    throw std::runtime_error("Number of image tokens does not match the number of images. Please fix the prompt.");
  }

  if (static_cast<int64_t>(audio_idx) != num_audios) {
    throw std::runtime_error("Number of audio tokens does not match the number of audios. Please fix the prompt.");
  }

  // input_ids is created. Pack it into an allocated OrtValue to avoid managing the memory.
  const std::vector<int64_t> shape{1, static_cast<int64_t>(processed_input_ids.size())};
  auto input_ids_value = OrtValue::CreateTensor<int32_t>(allocator, shape);
  std::copy(processed_input_ids.begin(), processed_input_ids.end(), input_ids_value->GetTensorMutableData<int32_t>());
  return std::tuple<std::unique_ptr<OrtValue>, std::unique_ptr<OrtValue>>(std::move(input_ids_value), std::move(audio_projection_mode_value));
}

}  // namespace

PhiMultiModalProcessor::PhiMultiModalProcessor(Config& config, const SessionInfo& session_info)
    : pixel_values_type_{session_info.GetInputDataType(config.model.vision.inputs.pixel_values)},
      attention_mask_type_{session_info.GetInputDataType(config.model.vision.inputs.attention_mask)},
      audio_features_type_{session_info.GetInputDataType(config.model.speech.inputs.audio_embeds)},
      audio_sizes_type_{session_info.GetInputDataType(config.model.speech.inputs.audio_sizes)} {
  const auto image_processor_config = (config.config_path / fs::path(config.model.vision.config_filename)).string();
  CheckResult(OrtxCreateProcessor(image_processor_.ToBeAssigned(), image_processor_config.c_str()));

  const auto audio_processor_config = (config.config_path / fs::path(config.model.speech.config_filename)).string();
  CheckResult(OrtxCreateSpeechFeatureExtractor(audio_processor_.ToBeAssigned(), audio_processor_config.c_str()));

  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.embedding.inputs.input_ids);

  config.AddMapping(std::string(Config::Defaults::PixelValuesName), config.model.vision.inputs.pixel_values);
  config.AddMapping(std::string(Config::Defaults::AttentionMaskName), config.model.vision.inputs.attention_mask);
  config.AddMapping(std::string(Config::Defaults::ImageSizesName), config.model.vision.inputs.image_sizes);

  config.AddMapping(std::string(Config::Defaults::AudioEmbedsName), config.model.speech.inputs.audio_embeds);
  config.AddMapping(std::string(Config::Defaults::AudioAttentionMaskName), config.model.speech.inputs.attention_mask);
  config.AddMapping(std::string(Config::Defaults::AudioSizesName), config.model.speech.inputs.audio_sizes);
}

std::unique_ptr<NamedTensors> PhiMultiModalProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> image_result;
  OrtxTensor *pixel_values{}, *image_sizes{}, *image_attention_mask{}, *num_img_tokens{};
  if (payload.images) {
    CheckResult(OrtxImagePreProcess(image_processor_.get(), payload.images->images_.get(), image_result.ToBeAssigned()));

    CheckResult(OrtxTensorResultGetAt(image_result.get(), 0, &pixel_values));
    CheckResult(OrtxTensorResultGetAt(image_result.get(), 1, &image_sizes));
    CheckResult(OrtxTensorResultGetAt(image_result.get(), 2, &image_attention_mask));
    CheckResult(OrtxTensorResultGetAt(image_result.get(), 3, &num_img_tokens));
  }

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> audio_result;
  OrtxTensor *audio_embeds{}, *audio_attention_mask{}, *audio_sizes{};
  if (payload.audios) {
    CheckResult(OrtxFeatureExtraction(audio_processor_.get(), payload.audios->audios_.get(), audio_result.ToBeAssigned()));

    CheckResult(OrtxTensorResultGetAt(audio_result.get(), 0, &audio_embeds));
    CheckResult(OrtxTensorResultGetAt(audio_result.get(), 1, &audio_attention_mask));
    CheckResult(OrtxTensorResultGetAt(audio_result.get(), 2, &audio_sizes));
  }

  auto [input_ids, audio_projection_mode] = ProcessImageAudioPrompt(tokenizer, payload.prompt, num_img_tokens, audio_sizes, allocator);
  named_tensors->emplace(Config::Defaults::InputIdsName, std::make_shared<Tensor>(std::move(input_ids)));
  named_tensors->emplace(Config::Defaults::AudioProjectionModeName, std::make_shared<Tensor>(std::move(audio_projection_mode)));

  if (payload.images) {
    if (pixel_values_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                             std::make_shared<Tensor>(ProcessTensor<float>(pixel_values, allocator)));
    } else if (pixel_values_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
      named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                             std::make_shared<Tensor>(ProcessTensor<Ort::BFloat16_t>(pixel_values, allocator)));
    } else {
      named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                             std::make_shared<Tensor>(ProcessTensor<Ort::Float16_t>(pixel_values, allocator)));
    }

    named_tensors->emplace(std::string(Config::Defaults::ImageSizesName),
                           std::make_shared<Tensor>(ProcessTensor<int64_t>(image_sizes, allocator)));
    if (attention_mask_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      named_tensors->emplace(std::string(Config::Defaults::ImageAttentionMaskName),
                             std::make_shared<Tensor>(ProcessTensor<float>(image_attention_mask, allocator)));
    } else if (attention_mask_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
      named_tensors->emplace(std::string(Config::Defaults::ImageAttentionMaskName),
                             std::make_shared<Tensor>(ProcessTensor<Ort::BFloat16_t>(image_attention_mask, allocator)));
    } else {
      named_tensors->emplace(std::string(Config::Defaults::ImageAttentionMaskName),
                             std::make_shared<Tensor>(ProcessTensor<Ort::Float16_t>(image_attention_mask, allocator)));
    }

    named_tensors->emplace(Config::Defaults::NumImageTokens,
                           std::make_shared<Tensor>(ProcessTensor<int64_t>(num_img_tokens, allocator)));
  }

  if (payload.audios) {
    if (audio_features_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      named_tensors->emplace(std::string(Config::Defaults::AudioEmbedsName),
                             std::make_shared<Tensor>(ProcessTensor<float>(audio_embeds, allocator)));
    } else if (audio_features_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
      named_tensors->emplace(std::string(Config::Defaults::AudioEmbedsName),
                             std::make_shared<Tensor>(ProcessTensor<Ort::BFloat16_t>(audio_embeds, allocator)));
    } else {
      named_tensors->emplace(std::string(Config::Defaults::AudioEmbedsName),
                             std::make_shared<Tensor>(ProcessTensor<Ort::Float16_t>(audio_embeds, allocator)));
    }

    named_tensors->emplace(std::string(Config::Defaults::AudioAttentionMaskName),
                           std::make_shared<Tensor>(ProcessTensor<bool>(audio_attention_mask, allocator)));

    named_tensors->emplace(std::string(Config::Defaults::AudioSizesName),
                           std::make_shared<Tensor>(ProcessTensor<float, int64_t>(audio_sizes, allocator)));
  }

  return named_tensors;
}

}  // namespace Generators
