// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"

#include <regex>

namespace Generators {

namespace {

std::tuple<std::unique_ptr<OrtValue>, std::unique_ptr<OrtValue>>
ProcessImagePrompt(const Generators::Tokenizer& tokenizer, const std::string& prompt,
                   OrtxTensor* num_img_tokens, Ort::Allocator& allocator) {
  const int64_t *num_img_tokens_data{}, *num_img_tokens_shape{};
  size_t num_img_tokens_num_dims{};
  if (num_img_tokens) {
    CheckResult(OrtxGetTensorData(num_img_tokens, reinterpret_cast<const void**>(&num_img_tokens_data),
                                  &num_img_tokens_shape, &num_img_tokens_num_dims));
  }

  int64_t num_images = 0;
  if (num_img_tokens_data) {
    num_images = num_img_tokens_num_dims > 0 ? num_img_tokens_shape[0] : 0;
  }

  // Qwen3-VL uses <|vision_start|>, <|image_pad|>, and <|vision_end|> tokens
  constexpr char vision_start_token[] = "<|vision_start|>";
  constexpr char image_pad_token[] = "<|image_pad|>";
  constexpr char vision_end_token[] = "<|vision_end|>";

  std::string text = prompt;

  // If prompt is empty and we have images, generate default prompt with vision tokens
  if (text.empty() && num_images > 0) {
    for (int64_t i = 0; i < num_images; ++i) {
      if (i > 0) text += " ";
      text += vision_start_token;
    }
  }

  // Count the number of vision_start tokens in the prompt
  const std::regex vision_start_regex{std::string(vision_start_token)};
  const auto vision_start_begin = std::sregex_iterator(text.begin(), text.end(), vision_start_regex);
  const auto vision_start_end = std::sregex_iterator();
  const auto vision_start_count = std::distance(vision_start_begin, vision_start_end);

  if (num_images != vision_start_count) {
    throw std::runtime_error("Prompt contained " + std::to_string(vision_start_count) +
                           " <|vision_start|> tokens but received " +
                           std::to_string(num_images) + " images.");
  }

  // Replace each <|vision_start|> with <|vision_start|><|image_pad|>...<|vision_end|>
  // where the number of <|image_pad|> tokens is determined by num_img_tokens_data
  if (num_images > 0 && num_img_tokens_data) {
    size_t pos = 0;
    for (int64_t i = 0; i < num_images; ++i) {
      pos = text.find(vision_start_token, pos);
      if (pos == std::string::npos) {
        break;
      }

      // Build the image token sequence: <|vision_start|> + N * <|image_pad|> + <|vision_end|>
      std::string image_sequence = std::string(vision_start_token);
      for (int64_t j = 0; j < num_img_tokens_data[i]; ++j) {
        image_sequence += image_pad_token;
      }
      image_sequence += vision_end_token;

      // Replace the <|vision_start|> token with the full sequence
      text.replace(pos, strlen(vision_start_token), image_sequence);
      pos += image_sequence.length();
    }
  }

  // Tokenize the expanded text
  const std::vector<int32_t> input_ids = tokenizer.Encode(text.c_str());

  // Create input_ids tensor
  std::unique_ptr<OrtValue> input_ids_value = OrtValue::CreateTensor<int32_t>(
      allocator, std::vector<int64_t>{1, static_cast<int64_t>(input_ids.size())});
  std::copy(input_ids.begin(), input_ids.end(), input_ids_value->GetTensorMutableData<int32_t>());

  // Create num_img_tokens tensor for output
  std::unique_ptr<OrtValue> num_img_tokens_value = nullptr;
  if (num_images > 0 && num_img_tokens_data) {
    num_img_tokens_value = OrtValue::CreateTensor<int64_t>(allocator, std::vector<int64_t>{num_images});
    std::copy(num_img_tokens_data, num_img_tokens_data + num_images,
              num_img_tokens_value->GetTensorMutableData<int64_t>());
  }

  return {std::move(input_ids_value), std::move(num_img_tokens_value)};
}

}  // namespace

Qwen3ImageProcessor::Qwen3ImageProcessor(Config& config, const SessionInfo& session_info)
    : pixel_values_type_{session_info.GetInputDataType(config.model.vision.inputs.pixel_values)} {
  const auto processor_config = (config.config_path / fs::path(config.model.vision.config_filename)).string();
  CheckResult(OrtxCreateProcessor(processor_.ToBeAssigned(), processor_config.c_str()));

  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.embedding.inputs.input_ids);
  config.AddMapping(std::string(Config::Defaults::PixelValuesName), config.model.vision.inputs.pixel_values);
  config.AddMapping(std::string(Config::Defaults::ImageSizesName), config.model.vision.inputs.image_sizes);
}

std::unique_ptr<NamedTensors> Qwen3ImageProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  std::string prompt = std::string(payload.prompt);
  const Images* images = payload.images;
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  if (!images) {
    auto [input_ids, num_img_tokens] = ProcessImagePrompt(tokenizer, prompt, nullptr, allocator);
    named_tensors->emplace(Config::Defaults::InputIdsName, std::make_shared<Tensor>(std::move(input_ids)));
    return named_tensors;
  }

  // Process images using ort_extensions
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  CheckResult(OrtxImagePreProcess(processor_.get(), images->images_.get(), result.ToBeAssigned()));

  OrtxTensor* pixel_values = nullptr;
  CheckResult(OrtxTensorResultGetAt(result.get(), 0, &pixel_values));

  OrtxTensor* image_sizes = nullptr;
  CheckResult(OrtxTensorResultGetAt(result.get(), 1, &image_sizes));

  OrtxTensor* num_img_tokens = nullptr;
  CheckResult(OrtxTensorResultGetAt(result.get(), 2, &num_img_tokens));

  // Process the prompt with image tokens
  auto [input_ids, num_img_tokens_output] = ProcessImagePrompt(tokenizer, prompt, num_img_tokens, allocator);

  named_tensors->emplace(std::string(Config::Defaults::InputIdsName),
                         std::make_shared<Tensor>(std::move(input_ids)));

  // Add pixel_values with appropriate type
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

  if (num_img_tokens_output) {
    named_tensors->emplace(Config::Defaults::NumImageTokens,
                           std::make_shared<Tensor>(std::move(num_img_tokens_output)));
  }

  return named_tensors;
}

}  // namespace Generators
