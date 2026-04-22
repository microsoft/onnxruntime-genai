// Copyright (C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// Portions of this file consist of AI generated content.
// Licensed under the MIT License. See License.txt in the project root for
// license information.

#include "../generators.h"
#include "model.h"
#include "videochat_flash_processor.h"
#include <regex>

namespace Generators {

namespace {

std::unique_ptr<OrtValue> ConvertPixelValues(const OrtValue& float_tensor,
                                             ONNXTensorElementDataType target_type,
                                             Ort::Allocator& allocator) {
  auto shape = float_tensor.GetTensorTypeAndShapeInfo()->GetShape();
  size_t count = float_tensor.GetTensorTypeAndShapeInfo()->GetElementCount();

  if (target_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    auto result = OrtValue::CreateTensor<float>(allocator, shape);
    std::copy(float_tensor.GetTensorData<float>(),
              float_tensor.GetTensorData<float>() + count,
              result->GetTensorMutableData<float>());
    return result;
  }

  std::unique_ptr<OrtValue> result;
  if (target_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
    result = OrtValue::CreateTensor<Ort::BFloat16_t>(allocator, shape);
  } else if (target_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    result = OrtValue::CreateTensor<Ort::Float16_t>(allocator, shape);
  } else {
    throw std::runtime_error("Unsupported target type for pixel values conversion");
  }

  auto* cpu_device = GetDeviceInterface(DeviceType::CPU);
  void* input_data = const_cast<void*>(static_cast<const void*>(float_tensor.GetTensorData<float>()));
  void* output_data = result->GetTensorMutableRawData();
  cpu_device->Cast(input_data, output_data, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, target_type, count);
  return result;
}

// Build input_ids from prompt, inserting fixed_tokens_per_image <|image_pad|> tokens per image.
std::tuple<std::unique_ptr<OrtValue>, std::unique_ptr<OrtValue>>
BuildPromptTokens(const Tokenizer& tokenizer, const std::string& prompt,
                  int64_t num_images, int64_t tokens_per_image,
                  Ort::Allocator& allocator) {
  constexpr char vision_start_token[] = "<|vision_start|>";
  constexpr char vision_end_token[] = "<|vision_end|>";
  constexpr char image_pad_token[] = "<|image_pad|>";

  std::string text = prompt;
  int64_t total_image_tokens = num_images * tokens_per_image;

  // Verify prompt has the right number of vision_start markers
  const std::regex vision_start_regex{R"(<\|vision_start\|>)"};
  auto begin = std::sregex_iterator(text.begin(), text.end(), vision_start_regex);
  auto end = std::sregex_iterator();
  int64_t marker_count = std::distance(begin, end);

  if (num_images > 0 && marker_count != num_images) {
    throw std::runtime_error("Prompt contained " + std::to_string(marker_count) +
                             " vision_start tokens but received " + std::to_string(num_images) + " images.");
  }

  // Replace each <|vision_start|>...<|vision_end|> block with the correct pad count
  if (num_images > 0) {
    std::string modified;
    size_t last_pos = 0;
    std::string temp = text;
    std::smatch match;

    while (std::regex_search(temp, match, vision_start_regex)) {
      size_t abs_pos = match.position() + (text.size() - temp.size());
      modified += text.substr(last_pos, abs_pos - last_pos);

      modified += vision_start_token;
      for (int64_t i = 0; i < tokens_per_image; ++i)
        modified += image_pad_token;
      modified += vision_end_token;

      last_pos = abs_pos + match.length();
      size_t ve_pos = text.find(vision_end_token, last_pos);
      if (ve_pos != std::string::npos)
        last_pos = ve_pos + strlen(vision_end_token);

      temp = match.suffix();
    }
    modified += text.substr(last_pos);
    text = modified;
  }

  const std::vector<int32_t> input_ids = tokenizer.Encode(text.c_str());

  auto input_ids_value = OrtValue::CreateTensor<int32_t>(
      allocator, std::vector<int64_t>{1, static_cast<int64_t>(input_ids.size())});
  std::copy(input_ids.begin(), input_ids.end(), input_ids_value->GetTensorMutableData<int32_t>());

  auto num_img_tokens = OrtValue::CreateTensor<int64_t>(allocator, std::vector<int64_t>{1});
  num_img_tokens->GetTensorMutableData<int64_t>()[0] = total_image_tokens;

  return {std::move(input_ids_value), std::move(num_img_tokens)};
}

}  // namespace

VideoChatFlashProcessor::VideoChatFlashProcessor(Config& config, const SessionInfo& session_info)
    : pixel_values_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT},
      num_visual_tokens_{config.model.vision.num_visual_tokens} {
  if (num_visual_tokens_ <= 0)
    throw std::runtime_error("videochat_flash_qwen requires vision.num_visual_tokens > 0 in genai_config.json");

  const auto processor_config = (config.config_path / fs::path(config.model.vision.config_filename)).string();
  CheckResult(OrtxCreateProcessor(processor_.ToBeAssigned(), processor_config.c_str()));

  try {
    pixel_values_type_ = session_info.GetInputDataType(config.model.vision.inputs.pixel_values);
  } catch (...) {
  }

  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.embedding.inputs.input_ids);
  config.AddMapping(std::string(Config::Defaults::PixelValuesName), config.model.vision.inputs.pixel_values);
}

std::unique_ptr<NamedTensors> VideoChatFlashProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  std::string prompt = std::string(payload.prompt);
  const Images* images = payload.images;
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  // Text-only: no image processing needed
  if (!images || images->num_images_ == 0) {
    auto [input_ids, num_img_tokens] = BuildPromptTokens(tokenizer, prompt, 0, 0, allocator);
    named_tensors->emplace(std::string(Config::Defaults::InputIdsName),
                           std::make_shared<Tensor>(std::move(input_ids)));
    named_tensors->emplace(std::string(Config::Defaults::NumImageTokens),
                           std::make_shared<Tensor>(std::move(num_img_tokens)));
    return named_tensors;
  }

  // Run ORT Extensions image preprocessing (Decode → Resize → Rescale → Normalize)
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  CheckResult(OrtxImagePreProcess(processor_.get(), images->images_.get(), result.ToBeAssigned()));

  OrtxTensor* pixel_values = nullptr;
  CheckResult(OrtxTensorResultGetAt(result.get(), 0, &pixel_values));

  const float* pv_data{};
  const int64_t* pv_shape{};
  size_t pv_ndims;
  CheckResult(OrtxGetTensorData(pixel_values, reinterpret_cast<const void**>(&pv_data),
                                &pv_shape, &pv_ndims));

  // Determine layout from ORT Extensions output (HWC format)
  int64_t num_imgs, height, width, channels;
  if (pv_ndims == 3) {
    num_imgs = 1;
    height = pv_shape[0];
    width = pv_shape[1];
    channels = pv_shape[2];
  } else if (pv_ndims == 4) {
    num_imgs = pv_shape[0];
    height = pv_shape[1];
    width = pv_shape[2];
    channels = pv_shape[3];
  } else {
    throw std::runtime_error("VideoChatFlashProcessor: unexpected pixel_values rank " +
                             std::to_string(pv_ndims) + " (expected 3 or 4)");
  }

  // Transpose HWC → CHW and reshape to [1, num_frames, C, H, W]
  std::vector<int64_t> target_shape = {1, num_imgs, channels, height, width};
  auto float_tensor = OrtValue::CreateTensor<float>(allocator, target_shape);
  float* dst = float_tensor->GetTensorMutableData<float>();

  for (int64_t n = 0; n < num_imgs; ++n) {
    const float* src_img = pv_data + n * height * width * channels;
    float* dst_img = dst + n * channels * height * width;
    for (int64_t c = 0; c < channels; ++c) {
      for (int64_t h = 0; h < height; ++h) {
        for (int64_t w = 0; w < width; ++w) {
          dst_img[c * height * width + h * width + w] = src_img[h * width * channels + w * channels + c];
        }
      }
    }
  }

  auto converted_pv = ConvertPixelValues(*float_tensor, pixel_values_type_, allocator);
  named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                         std::make_shared<Tensor>(std::move(converted_pv)));

  // Tokenize prompt with fixed visual token padding
  auto [input_ids, num_img_tokens] = BuildPromptTokens(
      tokenizer, prompt, static_cast<int64_t>(images->num_images_),
      num_visual_tokens_, allocator);
  named_tensors->emplace(std::string(Config::Defaults::InputIdsName),
                         std::make_shared<Tensor>(std::move(input_ids)));
  named_tensors->emplace(std::string(Config::Defaults::NumImageTokens),
                         std::make_shared<Tensor>(std::move(num_img_tokens)));

  // Emit image_grid_thw for GetImageFeatureBatchSize to determine num_images.
  // The pixel_values name is remapped (e.g. "pixel_values" → "images"), so the
  // rank-based lookup in GetImageFeatureBatchSize won't match; it falls through
  // to image_grid_thw whose name is not remapped.
  auto grid_thw = OrtValue::CreateTensor<int64_t>(allocator, std::vector<int64_t>{num_imgs, 3});
  auto* grid_ptr = grid_thw->GetTensorMutableData<int64_t>();
  for (int64_t i = 0; i < num_imgs; ++i) {
    grid_ptr[i * 3 + 0] = 1;
    grid_ptr[i * 3 + 1] = height;
    grid_ptr[i * 3 + 2] = width;
  }
  named_tensors->emplace("image_grid_thw",
                         std::make_shared<Tensor>(std::move(grid_thw)));

  return named_tensors;
}

}  // namespace Generators
