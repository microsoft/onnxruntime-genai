// Copyright (C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// Portions of this file consist of AI generated content.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License. See License.txt in the project root for
// license information.

#include "../generators.h"
#include "model.h"
#include "validate_config_path.h"
#include "videochat_flash_processor.h"
#include <regex>

namespace Generators {

namespace {

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

  ValidateConfigPath(config.model.vision.config_filename, "vision config_filename");
  const auto processor_config = (config.config_path / fs::path(config.model.vision.config_filename)).string();
  CheckResult(OrtxCreateProcessor(processor_.ToBeAssigned(), processor_config.c_str()));

  try {
    pixel_values_type_ = session_info.GetInputDataType(config.model.vision.inputs.pixel_values);
  } catch (...) {
    // pixel_values input may be absent when only the language decoder session is loaded;
    // the default-initialized pixel_values_type_ (FLOAT) is used in that case.
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

  ort_extensions::OrtxObjectPtr<OrtxTensor> pixel_values_owner;
  CheckResult(OrtxTensorResultGetAt(result.get(), 0, pixel_values_owner.ToBeAssigned()));
  OrtxTensor* pixel_values = pixel_values_owner.get();

  const float* pv_data{};
  const int64_t* pv_shape{};
  size_t pv_ndims;
  CheckResult(OrtxGetTensorData(pixel_values, reinterpret_cast<const void**>(&pv_data),
                                &pv_shape, &pv_ndims));

  // Detect whether ORT Extensions output is HWC or CHW.
  // Once processor_config.json includes a Permute3D step, the output will be
  // NCHW and the HWC path below can be removed.
  int64_t num_imgs, channels, height, width;
  bool is_hwc;
  if (pv_ndims == 3) {
    num_imgs = 1;
    // CHW: [C, H, W] vs HWC: [H, W, C] — channel dim is the small one
    is_hwc = (pv_shape[2] < pv_shape[0]);
    if (is_hwc) {
      height = pv_shape[0];
      width = pv_shape[1];
      channels = pv_shape[2];
    } else {
      channels = pv_shape[0];
      height = pv_shape[1];
      width = pv_shape[2];
    }
  } else if (pv_ndims == 4) {
    num_imgs = pv_shape[0];
    is_hwc = (pv_shape[3] < pv_shape[1]);
    if (is_hwc) {
      height = pv_shape[1];
      width = pv_shape[2];
      channels = pv_shape[3];
    } else {
      channels = pv_shape[1];
      height = pv_shape[2];
      width = pv_shape[3];
    }
  } else {
    throw std::runtime_error("VideoChatFlashProcessor: unexpected pixel_values rank " +
                             std::to_string(pv_ndims) + " (expected 3 or 4)");
  }

  // Vision model expects [1, num_frames, C, H, W]
  {
    std::vector<int64_t> target_shape = {1, num_imgs, channels, height, width};
    size_t count = static_cast<size_t>(num_imgs * channels * height * width);

    auto float_tensor = OrtValue::CreateTensor<float>(allocator, target_shape);
    float* dst = float_tensor->GetTensorMutableData<float>();

    if (is_hwc) {
      for (int64_t n = 0; n < num_imgs; ++n) {
        const float* src_img = pv_data + n * height * width * channels;
        float* dst_img = dst + n * channels * height * width;
        for (int64_t c = 0; c < channels; ++c)
          for (int64_t h = 0; h < height; ++h)
            for (int64_t w = 0; w < width; ++w)
              dst_img[c * height * width + h * width + w] = src_img[h * width * channels + w * channels + c];
      }
    } else {
      std::copy(pv_data, pv_data + count, dst);
    }

    std::unique_ptr<OrtValue> pv_ortvalue;
    if (pixel_values_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      pv_ortvalue = std::move(float_tensor);
    } else {
      auto* p_device = GetDeviceInterface(DeviceType::CPU);
      Cast(*float_tensor, pv_ortvalue, *p_device, pixel_values_type_);
    }
    named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                           std::make_shared<Tensor>(std::move(pv_ortvalue)));
  }

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
  named_tensors->emplace(std::string(Config::Defaults::ImageGridThwName),
                         std::make_shared<Tensor>(std::move(grid_thw)));

  return named_tensors;
}

}  // namespace Generators
