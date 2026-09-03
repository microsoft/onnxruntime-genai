// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "gemma4_vision_state.h"

#include <array>
#include <numeric>

namespace Generators {

namespace {

struct OrtValuePointerRestore {
  OrtValue*& slot;
  OrtValue* value;

  ~OrtValuePointerRestore() { slot = value; }
};

}  // namespace

void Gemma4VisionState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs,
                                       const int64_t num_images,
                                       const int64_t num_image_tokens) {
  image_token_counts_.clear();
  for (const auto& input : extra_inputs) {
    if (input.name == Config::Defaults::NumImageTokens && input.tensor->ort_tensor_) {
      const auto info = input.tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
      const auto count = info->GetElementCount();
      const int64_t* data = input.tensor->ort_tensor_->GetTensorData<int64_t>();
      image_token_counts_.assign(data, data + count);
      break;
    }
  }
  VisionState::SetExtraInputs(extra_inputs, num_images, num_image_tokens);

  const std::string& pixel_values_name = model_.config_->model.vision.inputs.pixel_values;
  const std::string& position_ids_name = model_.config_->model.vision.inputs.pixel_position_ids;
  pixel_values_index_ = SIZE_MAX;
  position_ids_index_ = SIZE_MAX;
  for (size_t i = 0; i < input_names_.size(); ++i) {
    if (input_names_[i] == pixel_values_name) pixel_values_index_ = i;
    if (input_names_[i] == position_ids_name) position_ids_index_ = i;
  }
}

DeviceSpan<float> Gemma4VisionState::Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                                         DeviceSpan<int32_t> next_indices) {
  if (model_.config_->model.vision.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.vision.run_options.value());
  }

  if (num_images_ <= 1) {
    State::Run(*model_.vision_session_);
    return {};
  }

  if (pixel_values_index_ == SIZE_MAX || position_ids_index_ == SIZE_MAX) {
    throw std::runtime_error(
        "Gemma 4 multi-image vision requires pixel_values and pixel_position_ids inputs");
  }
  if (image_token_counts_.size() != static_cast<size_t>(num_images_)) {
    throw std::runtime_error("Gemma 4 multi-image vision requires one num_image_tokens value per image");
  }

  OrtValue* pixel_values = inputs_[pixel_values_index_];
  OrtValue* position_ids = inputs_[position_ids_index_];
  OrtValue* image_features = outputs_[0];
  const OrtValuePointerRestore restore_pixel_values{inputs_[pixel_values_index_], pixel_values};
  const OrtValuePointerRestore restore_position_ids{inputs_[position_ids_index_], position_ids};
  const OrtValuePointerRestore restore_image_features{outputs_[0], image_features};
  const auto pixel_info = pixel_values->GetTensorTypeAndShapeInfo();
  const auto position_info = position_ids->GetTensorTypeAndShapeInfo();
  const auto feature_info = image_features->GetTensorTypeAndShapeInfo();
  const auto pixel_shape = pixel_info->GetShape();
  const auto position_shape = position_info->GetShape();
  const auto feature_shape = feature_info->GetShape();
  if (pixel_shape.size() != 3 || position_shape.size() != 3 ||
      pixel_shape[0] != num_images_ || position_shape[0] != num_images_ ||
      pixel_shape[1] != position_shape[1]) {
    throw std::runtime_error("Gemma 4 multi-image vision inputs must have matching [num_images, num_patches, ...] shapes");
  }
  if (feature_shape.size() != 2) {
    throw std::runtime_error("Gemma 4 image_features must have rank 2 [num_image_tokens, hidden_size]");
  }

  auto element_size = [](ONNXTensorElementDataType type) -> size_t {
    switch (type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return 4;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        return 2;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return 8;
      default:
        throw std::runtime_error("Unsupported tensor element type in Gemma 4 multi-image vision loop");
    }
  };

  const int64_t num_patches = pixel_shape[1];
  const int64_t patch_dim = pixel_shape[2];
  const int64_t position_dim = position_shape[2];
  const int64_t hidden_size = feature_shape[1];
  const auto pixel_type = pixel_info->GetElementType();
  const auto position_type = position_info->GetElementType();
  const auto feature_type = feature_info->GetElementType();
  const size_t pixel_bytes = element_size(pixel_type);
  const size_t position_bytes = element_size(position_type);
  const size_t feature_bytes = element_size(feature_type);
  auto* pixel_data = static_cast<uint8_t*>(pixel_values->GetTensorMutableRawData());
  auto* position_data = static_cast<uint8_t*>(position_ids->GetTensorMutableRawData());
  auto* feature_data = static_cast<uint8_t*>(image_features->GetTensorMutableRawData());
  const auto& pixel_memory = pixel_values->GetTensorMemoryInfo();
  const auto& position_memory = position_ids->GetTensorMemoryInfo();
  const auto& feature_memory = image_features->GetTensorMemoryInfo();

  const int64_t total_image_tokens =
      std::accumulate(image_token_counts_.begin(), image_token_counts_.end(), 0LL);
  if (total_image_tokens != feature_shape[0]) {
    throw std::runtime_error("Gemma 4 image token counts total " + std::to_string(total_image_tokens) +
                             " but image_features has space for " + std::to_string(feature_shape[0]) + " tokens");
  }

  int64_t feature_offset = 0;
  for (int64_t image = 0; image < num_images_; ++image) {
    const int64_t image_tokens = image_token_counts_[static_cast<size_t>(image)];
    if (image_tokens <= 0) {
      throw std::runtime_error("Gemma 4 image token counts must be positive");
    }
    const std::array<int64_t, 3> image_pixel_shape{1, num_patches, patch_dim};
    const std::array<int64_t, 3> image_position_shape{1, num_patches, position_dim};
    const std::array<int64_t, 2> image_feature_shape{image_tokens, hidden_size};

    auto image_pixel_values = OrtValue::CreateTensor(
        pixel_memory, pixel_data + static_cast<size_t>(image * num_patches * patch_dim) * pixel_bytes,
        static_cast<size_t>(num_patches * patch_dim) * pixel_bytes, image_pixel_shape, pixel_type);
    auto image_position_ids = OrtValue::CreateTensor(
        position_memory, position_data + static_cast<size_t>(image * num_patches * position_dim) * position_bytes,
        static_cast<size_t>(num_patches * position_dim) * position_bytes, image_position_shape, position_type);
    auto image_feature_values = OrtValue::CreateTensor(
        feature_memory, feature_data + static_cast<size_t>(feature_offset * hidden_size) * feature_bytes,
        static_cast<size_t>(image_tokens * hidden_size) * feature_bytes, image_feature_shape, feature_type);

    inputs_[pixel_values_index_] = image_pixel_values.get();
    inputs_[position_ids_index_] = image_position_ids.get();
    outputs_[0] = image_feature_values.get();
    State::Run(*model_.vision_session_);
    feature_offset += image_tokens;
  }

  return {};
}

}  // namespace Generators