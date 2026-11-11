// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qwen_vl_state.h"

#include <algorithm>
#include <cstdint>

#include "generator/generators.h"

namespace Generators {

namespace {

void ValidateImageGridThwLayoutAndCount(const std::vector<int64_t>& shape,
                                        size_t elem_count,
                                        int64_t num_images,
                                        const char* tensor_name) {
  if (num_images < 0) {
    throw std::runtime_error(std::string(tensor_name) + " num_images must be non-negative");
  }
  if (shape.size() != 2) {
    throw std::runtime_error(std::string(tensor_name) + " must have rank 2 [num_images, 3]");
  }
  if (shape[0] < 0 || shape[1] < 0) {
    throw std::runtime_error(std::string(tensor_name) + " dimensions must be non-negative");
  }
  if (shape[1] != 3) {
    throw std::runtime_error(std::string(tensor_name) + " second dimension must be 3");
  }

  const size_t shape_image_count = static_cast<size_t>(shape[0]);
  const size_t expected_image_count = static_cast<size_t>(num_images);
  if (shape_image_count < expected_image_count) {
    throw std::runtime_error(std::string(tensor_name) + " shape[0] (" + std::to_string(shape_image_count) +
                             ") is less than required image count (" + std::to_string(expected_image_count) + ")");
  }
  if (elem_count % 3 != 0 || elem_count / 3 < expected_image_count) {
    throw std::runtime_error(std::string(tensor_name) + " element count (" + std::to_string(elem_count) +
                             ") is less than required for " + std::to_string(num_images) +
                             " images (need at least 3 values per image)");
  }
}

}  // namespace

int64_t GetQwenImageCount(const std::vector<ExtraInput>& extra_inputs) {
  for (const auto& input : extra_inputs) {
    if (input.name == Config::Defaults::ImageGridThwName) {
      assert(input.tensor->ort_tensor_);
      const auto info = input.tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
      const auto shape = info->GetShape();
      const int64_t num_images = shape.empty() ? 0 : shape[0];
      ValidateImageGridThwLayoutAndCount(shape, info->GetElementCount(), num_images, "image_grid_thw");
      return num_images;
    }
  }
  return 0;
}

DeviceSpan<float> QwenVisionState::Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                                       DeviceSpan<int32_t> next_indices) {
  if (model_.config_->model.vision.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.vision.run_options.value());
  }

  if (num_images_ <= 1) {
    State::Run(*model_.vision_session_);
    return {};
  }

  const std::string& pixel_values_name = model_.config_->model.vision.inputs.pixel_values;
  const std::string& grid_name = model_.config_->model.vision.inputs.image_grid_thw;
  size_t pixel_values_index = SIZE_MAX;
  size_t grid_index = SIZE_MAX;
  for (size_t i = 0; i < input_names_.size(); ++i) {
    if (input_names_[i] == pixel_values_name) pixel_values_index = i;
    if (input_names_[i] == grid_name) grid_index = i;
  }
  if (pixel_values_index == SIZE_MAX || grid_index == SIZE_MAX) {
    State::Run(*model_.vision_session_);
    return {};
  }

  OrtValue* grid = inputs_[grid_index];
  const int64_t* grid_data = grid->GetTensorData<int64_t>();
  const auto grid_info = grid->GetTensorTypeAndShapeInfo();
  ValidateImageGridThwLayoutAndCount(
      grid_info->GetShape(), grid_info->GetElementCount(), num_images_, "image_grid_thw");

  bool model_supports_batch = false;
  const auto session_input_names = model_.vision_session_->GetInputNames();
  for (size_t i = 0; i < session_input_names.size(); ++i) {
    if (session_input_names[i] == grid_name) {
      const auto shape = model_.vision_session_->GetInputTypeInfo(i)->GetTensorTypeAndShapeInfo().GetShape();
      model_supports_batch = !shape.empty() && shape[0] <= 0;
      break;
    }
  }

  bool uniform_grid = true;
  for (int64_t image = 1; image < num_images_; ++image) {
    if (grid_data[image * 3] != grid_data[0] ||
        grid_data[image * 3 + 1] != grid_data[1] ||
        grid_data[image * 3 + 2] != grid_data[2]) {
      uniform_grid = false;
      break;
    }
  }
  if (model_supports_batch && uniform_grid) {
    State::Run(*model_.vision_session_);
    return {};
  }

  OrtValue* pixel_values = inputs_[pixel_values_index];
  OrtValue* image_features = outputs_[0];
  const auto pixel_info = pixel_values->GetTensorTypeAndShapeInfo();
  const auto feature_info = image_features->GetTensorTypeAndShapeInfo();
  const auto pixel_shape = pixel_info->GetShape();
  const auto feature_shape = feature_info->GetShape();
  const auto pixel_type = pixel_info->GetElementType();
  const auto feature_type = feature_info->GetElementType();
  const int64_t patch_dim = pixel_shape[1];
  const int64_t hidden_size = feature_shape[1];
  const size_t pixel_element_size = Ort::SizeOf(pixel_type);
  const size_t feature_element_size = Ort::SizeOf(feature_type);
  auto* pixel_data = static_cast<uint8_t*>(pixel_values->GetTensorMutableRawData());
  auto* feature_data = static_cast<uint8_t*>(image_features->GetTensorMutableRawData());
  const int64_t merge_size = model_.config_->model.vision.spatial_merge_size;
  const int64_t merge_square = merge_size * merge_size;
  const int64_t total_patches = pixel_shape[0];
  const int64_t total_features = feature_shape[0];

  int64_t total_grid_tokens = 0;
  int64_t total_hw = 0;
  int64_t max_grid_tokens = 0;
  bool all_temporal_dims_one = true;
  for (int64_t image = 0; image < num_images_; ++image) {
    const int64_t grid_tokens =
        grid_data[image * 3] * grid_data[image * 3 + 1] * grid_data[image * 3 + 2];
    total_grid_tokens += grid_tokens;
    total_hw += grid_data[image * 3 + 1] * grid_data[image * 3 + 2];
    max_grid_tokens = std::max(max_grid_tokens, grid_tokens);
    all_temporal_dims_one = all_temporal_dims_one && grid_data[image * 3] == 1;
  }
  const QwenPatchLayout patch_layout = ResolveQwenPatchLayout(
      total_patches, total_grid_tokens, total_hw, max_grid_tokens, num_images_, all_temporal_dims_one);

  const int64_t expected_total_features = total_grid_tokens / merge_square;
  if (total_features < expected_total_features) {
    throw std::runtime_error("pre-allocated image_features dim 0 (" + std::to_string(total_features) +
                             ") is smaller than expected (" + std::to_string(expected_total_features) +
                             ") for " + std::to_string(num_images_) + " images");
  }

  auto cpu_memory = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  int64_t patch_offset = 0;
  int64_t feature_offset = 0;
  for (int64_t image = 0; image < num_images_; ++image) {
    const int64_t temporal = grid_data[image * 3];
    const int64_t height = grid_data[image * 3 + 1];
    const int64_t width = grid_data[image * 3 + 2];
    const int64_t grid_tokens = temporal * height * width;
    const int64_t num_patches = patch_layout.ImagePatchCount(grid_tokens, height, width);
    const int64_t num_features = grid_tokens / merge_square;
    const int64_t image_patch_offset = patch_layout.ImagePatchOffset(image, patch_offset);

    if (grid_tokens % merge_square != 0) {
      throw std::runtime_error("grid tokens (" + std::to_string(grid_tokens) +
                               ") is not divisible by spatial_merge_size^2 (" +
                               std::to_string(merge_square) + ") for image " + std::to_string(image));
    }
    if (image_patch_offset + num_patches > total_patches) {
      throw std::runtime_error("patch_offset (" + std::to_string(image_patch_offset) + ") + num_patches (" +
                               std::to_string(num_patches) + ") exceeds pixel_values dim 0 (" +
                               std::to_string(total_patches) + ")");
    }
    if (feature_offset + num_features > total_features) {
      throw std::runtime_error("feat_offset (" + std::to_string(feature_offset) + ") + num_feats (" +
                               std::to_string(num_features) + ") exceeds image_features dim 0 (" +
                               std::to_string(total_features) + ")");
    }

    const std::array<int64_t, 2> image_pixel_shape{num_patches, patch_dim};
    const std::array<int64_t, 2> image_grid_shape{1, 3};
    const std::array<int64_t, 2> image_feature_shape{num_features, hidden_size};
    auto image_pixel_values = OrtValue::CreateTensor(
        *cpu_memory, pixel_data + static_cast<size_t>(image_patch_offset * patch_dim) * pixel_element_size,
        static_cast<size_t>(num_patches * patch_dim) * pixel_element_size, image_pixel_shape, pixel_type);
    auto image_grid = OrtValue::CreateTensor(
        *cpu_memory, const_cast<int64_t*>(grid_data + image * 3), 3 * sizeof(int64_t),
        image_grid_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    auto image_feature_values = OrtValue::CreateTensor(
        *cpu_memory, feature_data + static_cast<size_t>(feature_offset * hidden_size) * feature_element_size,
        static_cast<size_t>(num_features * hidden_size) * feature_element_size,
        image_feature_shape, feature_type);

    inputs_[pixel_values_index] = image_pixel_values.get();
    inputs_[grid_index] = image_grid.get();
    outputs_[0] = image_feature_values.get();
    State::Run(*model_.vision_session_);

    patch_offset += num_patches;
    feature_offset += num_features;
  }

  inputs_[pixel_values_index] = pixel_values;
  inputs_[grid_index] = grid;
  outputs_[0] = image_features;
  return {};
}

}  // namespace Generators