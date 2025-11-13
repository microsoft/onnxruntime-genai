// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"

#include <regex>

namespace Generators {

namespace {

std::tuple<std::unique_ptr<OrtValue>, std::unique_ptr<OrtValue>>
ProcessImagePrompt(const Generators::Tokenizer& tokenizer, const std::string& prompt,
                   OrtxTensor* pixel_values, OrtxTensor* image_grid_thw, Ort::Allocator& allocator) {
  constexpr char vision_start_token[] = "<|vision_start|>";
  constexpr char vision_end_token[] = "<|vision_end|>";
  constexpr char image_pad_token[] = "<|image_pad|>";

  int64_t num_images = 0;
  int64_t total_image_tokens = 0;
  
  if (pixel_values && image_grid_thw) {
    const float* pixel_values_data{};
    const int64_t* pixel_values_shape{};
    size_t pixel_values_num_dims;
    CheckResult(OrtxGetTensorData(pixel_values, reinterpret_cast<const void**>(&pixel_values_data),
                                  &pixel_values_shape, &pixel_values_num_dims));
    
    const int64_t* image_grid_thw_data{};
    const int64_t* image_grid_thw_shape{};
    size_t image_grid_thw_num_dims;
    CheckResult(OrtxGetTensorData(image_grid_thw, reinterpret_cast<const void**>(&image_grid_thw_data),
                                  &image_grid_thw_shape, &image_grid_thw_num_dims));
    
    num_images = image_grid_thw_shape[0];
    
    // Calculate total image tokens based on grid dimensions
    // For each image: (temporal * height * width) / (merge_size^2)
    constexpr int64_t merge_size = 2;
    for (int64_t i = 0; i < num_images; ++i) {
      int64_t t = image_grid_thw_data[i * 3 + 0];
      int64_t h = image_grid_thw_data[i * 3 + 1];
      int64_t w = image_grid_thw_data[i * 3 + 2];
      total_image_tokens += (t * h * w) / (merge_size * merge_size);
    }
  }

  // Generate input_ids with vision tokens
  std::string text = prompt;
  
  // If prompt is empty, add vision markers for each image
  if (text.empty()) {
    for (int64_t i = 0; i < num_images; ++i) {
      text += std::string(vision_start_token) + " " + std::string(vision_end_token);
      if (i < num_images - 1) {
        text += " ";
      }
    }
  }

  // Count the number of vision_start tokens and make sure it matches the number of images
  const std::regex vision_start_regex{std::string(vision_start_token)};
  const auto vision_start_begin = std::sregex_iterator(text.begin(), text.end(), vision_start_regex);
  const auto vision_start_end = std::sregex_iterator();
  const auto vision_start_tokens = std::distance(vision_start_begin, vision_start_end);
  
  if (num_images != vision_start_tokens) {
    throw std::runtime_error("Prompt contained " + std::to_string(vision_start_tokens) + 
                           " vision_start tokens but received " + std::to_string(num_images) + " images.");
  }

  // For Qwen2-VL, we need to replace vision markers with image_pad tokens
  // The number of image_pad tokens for each image depends on the image dimensions
  if (num_images > 0 && image_grid_thw) {
    const int64_t* image_grid_thw_data{};
    const int64_t* image_grid_thw_shape{};
    size_t image_grid_thw_num_dims;
    CheckResult(OrtxGetTensorData(image_grid_thw, reinterpret_cast<const void**>(&image_grid_thw_data),
                                  &image_grid_thw_shape, &image_grid_thw_num_dims));
    
    constexpr int64_t merge_size = 2;
    std::string modified_text;
    size_t last_pos = 0;
    size_t image_idx = 0;
    
    std::smatch match;
    std::string temp_text = text;
    while (std::regex_search(temp_text, match, vision_start_regex)) {
      // Add text before the vision_start token
      modified_text += text.substr(last_pos, match.position() - (last_pos - (text.size() - temp_text.size())));
      
      // Calculate number of image_pad tokens for this image
      int64_t t = image_grid_thw_data[image_idx * 3 + 0];
      int64_t h = image_grid_thw_data[image_idx * 3 + 1];
      int64_t w = image_grid_thw_data[image_idx * 3 + 2];
      int64_t num_pads = (t * h * w) / (merge_size * merge_size);
      
      // Add vision_start, image_pad tokens, and vision_end
      modified_text += vision_start_token;
      for (int64_t i = 0; i < num_pads; ++i) {
        modified_text += image_pad_token;
      }
      modified_text += vision_end_token;
      
      last_pos = match.position() + match.length() + (text.size() - temp_text.size());
      
      // Find and skip vision_end token
      size_t vision_end_pos = text.find(vision_end_token, last_pos);
      if (vision_end_pos != std::string::npos) {
        last_pos = vision_end_pos + strlen(vision_end_token);
      }
      
      temp_text = match.suffix();
      image_idx++;
    }
    modified_text += text.substr(last_pos);
    text = modified_text;
  }

  const std::vector<int32_t> input_ids = tokenizer.Encode(text.c_str());

  std::unique_ptr<OrtValue> input_ids_value = OrtValue::CreateTensor<int32_t>(
      allocator, std::vector<int64_t>{1, static_cast<int64_t>(input_ids.size())});
  std::copy(input_ids.begin(), input_ids.end(), input_ids_value->GetTensorMutableData<int32_t>());

  std::unique_ptr<OrtValue> num_img_tokens = OrtValue::CreateTensor<int64_t>(
      allocator, std::vector<int64_t>{1});
  num_img_tokens->GetTensorMutableData<int64_t>()[0] = total_image_tokens;

  return {std::move(input_ids_value), std::move(num_img_tokens)};
}

}  // namespace

QwenImageProcessor::QwenImageProcessor(Config& config, const SessionInfo& session_info)
    : pixel_values_type_{session_info.GetInputDataType(config.model.vision.inputs.pixel_values)} {
  const auto processor_config = (config.config_path / fs::path(config.model.vision.config_filename)).string();
  CheckResult(OrtxCreateProcessor(processor_.ToBeAssigned(), processor_config.c_str()));

  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.embedding.inputs.input_ids);
  config.AddMapping(std::string(Config::Defaults::PixelValuesName), config.model.vision.inputs.pixel_values);
}

std::unique_ptr<NamedTensors> QwenImageProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  std::string prompt = std::string(payload.prompt);
  const Images* images = payload.images;
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  if (!images) {
    [[maybe_unused]] auto [input_ids, num_img_tokens] = ProcessImagePrompt(tokenizer, prompt, nullptr, nullptr, allocator);
    named_tensors->emplace(Config::Defaults::InputIdsName, std::make_shared<Tensor>(std::move(input_ids)));
    return named_tensors;
  }

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  CheckResult(OrtxImagePreProcess(processor_.get(), images->images_.get(), result.ToBeAssigned()));

  OrtxTensor* pixel_values = nullptr;
  CheckResult(OrtxTensorResultGetAt(result.get(), 0, &pixel_values));

  OrtxTensor* image_grid_thw = nullptr;
  CheckResult(OrtxTensorResultGetAt(result.get(), 1, &image_grid_thw));

  auto [input_ids, num_img_tokens] = ProcessImagePrompt(tokenizer, prompt, pixel_values, image_grid_thw, allocator);
  named_tensors->emplace(std::string(Config::Defaults::InputIdsName), std::make_shared<Tensor>(std::move(input_ids)));

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

  // Add image_grid_thw tensor
  named_tensors->emplace("image_grid_thw",
                         std::make_shared<Tensor>(ProcessTensor<int64_t>(image_grid_thw, allocator)));

  named_tensors->emplace(std::string(Config::Defaults::NumImageTokens), std::make_shared<Tensor>(std::move(num_img_tokens)));

  return named_tensors;
}

}  // namespace Generators
