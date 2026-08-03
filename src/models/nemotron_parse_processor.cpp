// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "nemotron_parse_processor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace Generators {
namespace {

constexpr std::array<float, 3> kClipMean{0.48145466f, 0.4578275f, 0.40821073f};
constexpr std::array<float, 3> kClipStd{0.26862954f, 0.26130258f, 0.27577711f};
// Request bounding boxes, element classes, and Markdown in the parsed output.
constexpr std::string_view kDefaultTaskPrompt =
    "</s><s><predict_bbox><predict_classes><output_markdown>";

std::unique_ptr<OrtValue> BuildInputIds(const Tokenizer& tokenizer,
                                        std::string_view prompt,
                                        int32_t decoder_start_token_id,
                                        Ort::Allocator& allocator) {
  const auto task_prompt = prompt.empty() ? kDefaultTaskPrompt : prompt;
  auto prompt_ids = tokenizer.Encode(std::string(task_prompt).c_str());
  const int32_t tokenizer_bos = tokenizer.TokenToTokenId("<s>");
  const int32_t tokenizer_eos = tokenizer.TokenToTokenId("</s>");

  // AutoProcessor tokenization uses add_special_tokens=True. OGA tokenizers
  // deliberately disable automatic special tokens, so reproduce the checkpoint's
  // [decoder_start, tokenizer_bos, prompt..., tokenizer_eos] contract here.
  std::vector<int32_t> input_ids;
  input_ids.reserve(prompt_ids.size() + 3);
  input_ids.push_back(decoder_start_token_id);
  input_ids.push_back(tokenizer_bos);
  input_ids.insert(input_ids.end(), prompt_ids.begin(), prompt_ids.end());
  input_ids.push_back(tokenizer_eos);

  const std::array<int64_t, 2> shape{1, static_cast<int64_t>(input_ids.size())};
  auto value = OrtValue::CreateTensor<int32_t>(allocator, shape);
  std::copy(input_ids.begin(), input_ids.end(),
            value->GetTensorMutableData<int32_t>());
  return value;
}

struct DecodedImage {
  const uint8_t* data;
  int64_t height;
  int64_t width;
};

DecodedImage GetDecodedImage(OrtxTensor* tensor) {
  const uint8_t* data{};
  const int64_t* shape{};
  size_t rank{};
  CheckResult(OrtxGetTensorData(tensor, reinterpret_cast<const void**>(&data),
                                &shape, &rank));

  if (rank == 3 && shape[2] == 3) {
    return {data, shape[0], shape[1]};
  }
  if (rank == 4 && shape[0] == 1 && shape[3] == 3) {
    return {data, shape[1], shape[2]};
  }
  throw std::runtime_error(
      "Nemotron Parse decoded image must have shape [H,W,3] or [1,H,W,3]");
}

std::pair<int64_t, int64_t> ResizeShape(int64_t source_height,
                                        int64_t source_width,
                                        int64_t target_height,
                                        int64_t target_width) {
  int64_t resized_height = source_height;
  int64_t resized_width = source_width;
  const double aspect_ratio =
      static_cast<double>(source_width) / static_cast<double>(source_height);

  // Match the checkpoint's LongestMaxSizeHW implementation, including its
  // sequential integer truncation.
  if (source_height > target_height) {
    resized_height = target_height;
    resized_width = static_cast<int64_t>(resized_height * aspect_ratio);
  }
  if (resized_width > target_width) {
    resized_width = target_width;
    resized_height = static_cast<int64_t>(resized_width / aspect_ratio);
  }
  return {std::max<int64_t>(1, resized_height),
          std::max<int64_t>(1, resized_width)};
}

float BilinearSample(const DecodedImage& image, int64_t y, int64_t x,
                     int64_t resized_height, int64_t resized_width,
                     int channel) {
  const double source_y =
      (static_cast<double>(y) + 0.5) * image.height / resized_height - 0.5;
  const double source_x =
      (static_cast<double>(x) + 0.5) * image.width / resized_width - 0.5;
  const int64_t y0 = std::clamp<int64_t>(
      static_cast<int64_t>(std::floor(source_y)), 0, image.height - 1);
  const int64_t x0 = std::clamp<int64_t>(
      static_cast<int64_t>(std::floor(source_x)), 0, image.width - 1);
  const int64_t y1 = std::min(y0 + 1, image.height - 1);
  const int64_t x1 = std::min(x0 + 1, image.width - 1);
  const double wy = std::clamp(source_y, 0.0,
                               static_cast<double>(image.height - 1)) -
                    y0;
  const double wx = std::clamp(source_x, 0.0,
                               static_cast<double>(image.width - 1)) -
                    x0;

  const auto at = [&](int64_t sy, int64_t sx) {
    return static_cast<double>(
        image.data[(sy * image.width + sx) * 3 + channel]);
  };
  const double top = at(y0, x0) * (1.0 - wx) + at(y0, x1) * wx;
  const double bottom = at(y1, x0) * (1.0 - wx) + at(y1, x1) * wx;
  return static_cast<float>(top * (1.0 - wy) + bottom * wy);
}

std::unique_ptr<OrtValue> PreprocessImage(const DecodedImage& image,
                                          int64_t target_height,
                                          int64_t target_width,
                                          ONNXTensorElementDataType output_type,
                                          Ort::Allocator& allocator) {
  const auto [resized_height, resized_width] =
      ResizeShape(image.height, image.width, target_height, target_width);
  const int64_t pad_top = (target_height - resized_height) / 2;
  const int64_t pad_left = (target_width - resized_width) / 2;
  const std::array<int64_t, 4> shape{1, 3, target_height, target_width};
  auto fp32 = OrtValue::CreateTensor<float>(allocator, shape);
  float* output = fp32->GetTensorMutableData<float>();

  for (int channel = 0; channel < 3; ++channel) {
    const float white = (1.0f - kClipMean[channel]) / kClipStd[channel];
    float* channel_output =
        output + channel * target_height * target_width;
    std::fill_n(channel_output, target_height * target_width, white);
    for (int64_t y = 0; y < resized_height; ++y) {
      for (int64_t x = 0; x < resized_width; ++x) {
        const float pixel =
            BilinearSample(image, y, x, resized_height, resized_width, channel) /
            255.0f;
        channel_output[(y + pad_top) * target_width + x + pad_left] =
            (pixel - kClipMean[channel]) / kClipStd[channel];
      }
    }
  }

  if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return fp32;
  }
  if (output_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
      output_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
    throw std::runtime_error(
        "Nemotron Parse pixel_values must be float, float16, or bfloat16");
  }

  std::unique_ptr<OrtValue> converted;
  Cast(*fp32, converted, *GetDeviceInterface(DeviceType::CPU), output_type);
  return converted;
}

}  // namespace

NemotronParseProcessor::NemotronParseProcessor(
    Config& config, const SessionInfo& session_info)
    : pixel_values_type_{session_info.GetInputDataType(
          config.model.vision.inputs.pixel_values)},
      decoder_start_token_id_{config.model.bos_token_id} {
  const auto shape =
      session_info.GetInputShape(config.model.vision.inputs.pixel_values);
  if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3 ||
      shape[2] <= 0 || shape[3] <= 0) {
    throw std::runtime_error(
        "Nemotron Parse native processor requires pixel_values [1,3,H,W]");
  }
  target_height_ = shape[2];
  target_width_ = shape[3];

  const auto processor_config =
      (config.config_path /
       fs::path(config.model.vision.config_filename))
          .string();
  CheckResult(
      OrtxCreateProcessor(processor_.ToBeAssigned(), processor_config.c_str()));

  config.AddMapping(std::string(Config::Defaults::InputIdsName),
                    config.model.decoder.inputs.input_ids);
  config.AddMapping(std::string(Config::Defaults::PixelValuesName),
                    config.model.vision.inputs.pixel_values);
}

std::unique_ptr<NamedTensors> NemotronParseProcessor::Process(
    const Tokenizer& tokenizer, const Payload& payload) const {
  if (!payload.images || payload.images->num_images_ != 1) {
    throw std::runtime_error("Nemotron Parse requires exactly one image");
  }
  if (payload.audios) {
    throw std::runtime_error("Nemotron Parse does not accept audio input");
  }

  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();
  named_tensors->emplace(
      std::string(Config::Defaults::InputIdsName),
      std::make_shared<Tensor>(BuildInputIds(
          tokenizer, payload.prompt, decoder_start_token_id_, allocator)));

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  CheckResult(OrtxImagePreProcess(processor_.get(),
                                  payload.images->images_.get(),
                                  result.ToBeAssigned()));
  ort_extensions::OrtxObjectPtr<OrtxTensor> decoded_owner;
  CheckResult(
      OrtxTensorResultGetAt(result.get(), 0, decoded_owner.ToBeAssigned()));
  auto pixel_values =
      PreprocessImage(GetDecodedImage(decoded_owner.get()), target_height_,
                      target_width_, pixel_values_type_, allocator);
  named_tensors->emplace(
      std::string(Config::Defaults::PixelValuesName),
      std::make_shared<Tensor>(std::move(pixel_values)));
  return named_tensors;
}

}  // namespace Generators
