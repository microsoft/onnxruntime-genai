// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "config.h"
#include "models/model.h"
#include "models/nemotron_parse.h"
#include "models/preprocessing/genai_tokenizer.h"
#include "models/preprocessing/nemotron_parse_processor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace Generators {
namespace {

// Request bounding boxes, element classes, and Markdown in the parsed output.
constexpr std::string_view kDefaultTaskPrompt =
    "</s><s><predict_bbox><predict_classes><output_markdown>";

std::unique_ptr<OrtValue> BuildInputIds(const Tokenizer& tokenizer,
                                        std::string_view prompt,
                                        int32_t decoder_start_token_id,
                                        int64_t required_prompt_length,
                                        int context_length,
                                        Ort::Allocator& allocator) {
  auto input_ids = tokenizer.Encode(std::string(prompt).c_str());
  // The tokenizer handles its own special tokens; decoder-start is model-specific.
  input_ids.insert(input_ids.begin(), decoder_start_token_id);

  ValidateNemotronParsePromptLength(input_ids.size(), required_prompt_length,
                                   context_length);

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
                                          const std::array<float, 3>& image_mean,
                                          const std::array<float, 3>& image_std,
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
    const float white = (1.0f - image_mean[channel]) / image_std[channel];
    float* channel_output =
        output + channel * target_height * target_width;
    std::fill_n(channel_output, target_height * target_width, white);
    for (int64_t y = 0; y < resized_height; ++y) {
      for (int64_t x = 0; x < resized_width; ++x) {
        const float pixel =
            BilinearSample(image, y, x, resized_height, resized_width, channel) /
            255.0f;
        channel_output[(y + pad_top) * target_width + x + pad_left] =
            (pixel - image_mean[channel]) / image_std[channel];
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
      decoder_start_token_id_{config.model.bos_token_id},
      context_length_{config.model.context_length} {
  const auto input_ids_shape =
      session_info.GetInputShape(config.model.decoder.inputs.input_ids);
  if (input_ids_shape.size() != 2) {
    throw std::runtime_error(
        "Nemotron Parse decoder input_ids must have rank 2");
  }
  required_prompt_length_ = input_ids_shape[1];
  const auto shape =
      session_info.GetInputShape(config.model.vision.inputs.pixel_values);
  if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3 ||
      shape[2] <= 0 || shape[3] <= 0) {
    throw std::runtime_error(
        "Nemotron Parse native processor requires pixel_values [1,3,H,W]");
  }
  target_height_ = shape[2];
  target_width_ = shape[3];

  const auto processor_path = config.config_path / fs::path(config.model.vision.config_filename);
  const auto vision_config = LoadVisionProcessingConfig(processor_path);
  const auto check_dimension = [&](std::optional<int64_t> dimension, int64_t expected, const char* name) {
    if (dimension && *dimension != expected)
      throw std::runtime_error("Invalid Nemotron Parse vision processing config '" + processor_path.string() +
                               "': " + name + " must match the encoder pixel_values shape; re-export the model");
  };
  check_dimension(vision_config.image_height, target_height_, "image_height");
  check_dimension(vision_config.image_width, target_width_, "image_width");
  image_mean_ = vision_config.image_mean;
  image_std_ = vision_config.image_std;
  const auto processor_config = processor_path.string();
  CheckResult(
      OrtxCreateProcessor(processor_.ToBeAssigned(), processor_config.c_str()));

  config.AddMapping(std::string(Config::Defaults::InputIdsName),
                    config.model.decoder.inputs.input_ids);
  config.AddMapping(std::string(Config::Defaults::PixelValuesName),
                    config.model.vision.inputs.pixel_values);
}

void NemotronParseProcessor::ConfigureTokenizer(Tokenizer& tokenizer) const {
  const char* keys[] = {"add_special_tokens"};
  const char* values[] = {"true"};
  tokenizer.UpdateOptions(keys, values, 1);
}

std::unique_ptr<NamedTensors> NemotronParseProcessor::Process(
    const Tokenizer& tokenizer, const Payload& payload) const {
  if (!payload.images || payload.images->num_images_ != 1) {
    throw std::runtime_error("Nemotron Parse requires exactly one image");
  }
  if (payload.audios) {
    throw std::runtime_error("Nemotron Parse does not accept audio input");
  }
  if (payload.prompts.size() > 1)
    throw std::runtime_error("Nemotron Parse does not support multiple prompts");
  if (!payload.prompts.empty() && payload.prompts[0] == nullptr)
    throw std::runtime_error("Nemotron Parse prompt list must not contain a null prompt");
  const std::string_view prompt = payload.prompts.empty()
                                      ? std::string_view{payload.prompt}
                                      : std::string_view{payload.prompts[0]};

  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();
  named_tensors->emplace(
      std::string(Config::Defaults::InputIdsName),
      std::make_shared<Tensor>(BuildInputIds(
          tokenizer, prompt.empty() ? kDefaultTaskPrompt : prompt, decoder_start_token_id_,
          required_prompt_length_, context_length_, allocator)));

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  CheckResult(OrtxImagePreProcess(processor_.get(),
                                  payload.images->images_.get(),
                                  result.ToBeAssigned()));
  ort_extensions::OrtxObjectPtr<OrtxTensor> decoded_owner;
  CheckResult(
      OrtxTensorResultGetAt(result.get(), 0, decoded_owner.ToBeAssigned()));
  auto pixel_values =
      PreprocessImage(GetDecodedImage(decoded_owner.get()), target_height_,
                      target_width_, image_mean_, image_std_, pixel_values_type_, allocator);
  named_tensors->emplace(
      std::string(Config::Defaults::PixelValuesName),
      std::make_shared<Tensor>(std::move(pixel_values)));
  return named_tensors;
}

}  // namespace Generators
