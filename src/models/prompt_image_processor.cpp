// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"

#include <regex>

namespace Generators {

namespace {

std::unique_ptr<OrtValue> ProcessImagePrompt(const Generators::Tokenizer& tokenizer, const std::string& prompt,
                                             ortc::Tensor<int64_t>* num_image_tokens, Ort::Allocator& allocator) {
  const size_t num_images = num_image_tokens ? num_image_tokens->NumberOfElement() : 0U;
  auto* num_image_tokens_data = num_image_tokens ? num_image_tokens->Data() : nullptr;

  const std::regex pattern("<\\|image_\\d+\\|>");
  const std::vector<std::string> prompt_chunks(
      std::sregex_token_iterator(prompt.begin(), prompt.end(), pattern, -1),
      std::sregex_token_iterator());

  std::vector<std::vector<int32_t>> input_ids_chunks(prompt_chunks.size());
  for (size_t i = 0; i < prompt_chunks.size(); ++i) {
    input_ids_chunks[i] = tokenizer.Encode(prompt_chunks[i].c_str());
  }

  const std::vector<std::string> image_tags(
      std::sregex_token_iterator(prompt.begin(), prompt.end(), pattern),
      std::sregex_token_iterator());

  std::vector<int32_t> image_ids(image_tags.size());
  constexpr size_t image_id_position_begin = 8;  // <|image_ : Character at idx 8 is the beginning of the image_id
  for (size_t i = 0; i < image_tags.size(); ++i) {
    const size_t image_id_position_end = image_tags[i].size() - 2;  // |> : Character at idx size() - 2 is '|' which marks the end of the image_id
    image_ids[i] = std::stoi(image_tags[i].substr(image_id_position_begin,
                                                  image_id_position_end - image_id_position_begin));
  }

  if (std::set<int32_t>(image_ids.begin(), image_ids.end()).size() != num_images) {
    throw std::runtime_error("Number of unique image tags does not match the number of images.");
  }

  std::vector<int32_t> input_ids;
  for (size_t i = 0; i < input_ids_chunks.size(); ++i) {
    input_ids.insert(input_ids.end(), input_ids_chunks[i].begin(), input_ids_chunks[i].end());
    if (i < image_ids.size()) {
      if (image_ids[i] < 1 || image_ids[i] > static_cast<int32_t>(num_images)) {
        std::string error_message = "Encountered unexpected value of image_id in the prompt. Expected a value <= " +
                                    std::to_string(num_images) + ". Actual value: " + std::to_string(image_ids[i]);
        throw std::runtime_error(error_message);
      }
      for (size_t j = 0; j < num_image_tokens_data[image_ids[i] - 1]; ++j) {
        input_ids.push_back(-image_ids[i]);
      }
    }
  }

  const std::vector<int64_t> shape{1, static_cast<int64_t>(input_ids.size())};
  auto input_ids_value = OrtValue::CreateTensor<int32_t>(allocator, shape);
  std::copy(input_ids.begin(), input_ids.end(), input_ids_value->GetTensorMutableData<int32_t>());
  return input_ids_value;
}

std::unique_ptr<OrtValue> ProcessPixelValues(ortc::Tensor<float>* pixel_values, ONNXTensorElementDataType expected_type,
                                             Ort::Allocator& allocator) {
  if (!(expected_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || expected_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)) {
    throw std::runtime_error("Expected pixel_values to be of type float or float16. Actual: " + expected_type);
  }
  auto pixel_values_value = expected_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                                ? OrtValue::CreateTensor<float>(allocator, pixel_values->Shape())
                                : OrtValue::CreateTensor<Ort::Float16_t>(allocator, pixel_values->Shape());
  if (expected_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    std::copy(pixel_values->Data(), pixel_values->Data() + pixel_values->NumberOfElement(),
              pixel_values_value->GetTensorMutableData<float>());
  } else {
    auto pixel_values_fp32 = OrtValue::CreateTensor<float>(
        allocator.GetInfo(),
        std::span<float>(const_cast<float*>(pixel_values->Data()), pixel_values->NumberOfElement()),
        pixel_values->Shape());
    ConvertFp32ToFp16(allocator, *pixel_values_fp32, pixel_values_value, DeviceType::CPU, nullptr);
  }

  return pixel_values_value;
}

std::unique_ptr<OrtValue> ProcessImageSizes(ortc::Tensor<int64_t>* image_sizes, Ort::Allocator& allocator) {
  auto image_sizes_value = OrtValue::CreateTensor<int64_t>(allocator, image_sizes->Shape());
  std::copy(image_sizes->Data(), image_sizes->Data() + image_sizes->NumberOfElement(),
            image_sizes_value->GetTensorMutableData<int64_t>());
  return image_sizes_value;
}

}  // namespace

std::unique_ptr<Images> LoadImage(const char* image_path) {
  if (!fs::exists(image_path)) {
    throw std::runtime_error("Image path does not exist: " + std::string(image_path));
  }
  auto [images, num_images] = ort_extensions::LoadRawImages({image_path});
  return std::make_unique<Images>(std::move(images), num_images);
}

ImageProcessor::ImageProcessor(const Config& config, const SessionInfo& session_info)
    : input_ids_name_{config.model.vision.inputs.input_ids},
      pixel_values_name_{config.model.vision.inputs.pixel_values},
      pixel_values_type_{session_info.GetInputDataType(pixel_values_name_)},
      image_sizes_name_{config.model.vision.inputs.image_sizes} {
  auto processor_config = (config.config_path / config.model.vision.image_processor.processor_config).u8string();
  CheckResult(OrtxCreateProcessor(processor_.Address(), processor_config.c_str()));
}

std::unique_ptr<NamedTensors> ImageProcessor::Process(const Tokenizer& tokenizer, const std::string& prompt, const Images& images) {
  ort_extensions::ImageProcessor* processor = static_cast<ort_extensions::ImageProcessor*>(processor_.p_);

  ortc::Tensor<float>* pixel_values;
  ortc::Tensor<int64_t>* image_sizes;
  ortc::Tensor<int64_t>* num_img_tokens;
  auto [status, result] = processor->PreProcess(ort_extensions::span(images.images_.get(), images.num_images_),
                                                &pixel_values, &image_sizes, &num_img_tokens);
  if (!status.IsOk()) {
    throw std::runtime_error(status.ToString());
  }

  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();
  named_tensors->emplace(input_ids_name_,
                         std::make_shared<Tensor>(ProcessImagePrompt(tokenizer, prompt, num_img_tokens, allocator)));
  named_tensors->emplace(pixel_values_name_,
                         std::make_shared<Tensor>(ProcessPixelValues(pixel_values, pixel_values_type_, allocator)));
  named_tensors->emplace(image_sizes_name_,
                         std::make_shared<Tensor>(ProcessImageSizes(image_sizes, allocator)));

  processor->ClearOutputs(&result);

  return named_tensors;
}

}  // namespace Generators
