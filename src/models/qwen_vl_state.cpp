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
      const auto shape = input.tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape();
      const int64_t num_images = shape.empty() ? 0 : shape[0];
      const size_t elem_count = input.tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetElementCount();
      ValidateImageGridThwLayoutAndCount(shape, elem_count, num_images, "image_grid_thw");
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

  const std::string& pv_name = model_.config_->model.vision.inputs.pixel_values;
  const std::string& grid_name = model_.config_->model.vision.inputs.image_grid_thw;

  size_t pv_idx = SIZE_MAX;
  size_t grid_idx = SIZE_MAX;
  for (size_t i = 0; i < input_names_.size(); ++i) {
    if (input_names_[i] == pv_name) {
      pv_idx = i;
    }
    if (input_names_[i] == grid_name) {
      grid_idx = i;
    }
  }

  if (pv_idx == SIZE_MAX || grid_idx == SIZE_MAX) {
    State::Run(*model_.vision_session_);
    return {};
  }

  OrtValue* grid_full = inputs_[grid_idx];
  const int64_t* grid_data = grid_full->GetTensorData<int64_t>();

  const auto grid_shape = grid_full->GetTensorTypeAndShapeInfo()->GetShape();
  const size_t grid_elem_count = grid_full->GetTensorTypeAndShapeInfo()->GetElementCount();
  ValidateImageGridThwLayoutAndCount(grid_shape, grid_elem_count, num_images_, "image_grid_thw");

  bool model_supports_batch = false;
  {
    auto session_input_names = model_.vision_session_->GetInputNames();
    for (size_t si = 0; si < session_input_names.size(); ++si) {
      if (session_input_names[si] == grid_name) {
        auto grid_input_info = model_.vision_session_->GetInputTypeInfo(si);
        auto grid_expected_shape = grid_input_info->GetTensorTypeAndShapeInfo().GetShape();
        if (!grid_expected_shape.empty() && grid_expected_shape[0] <= 0) {
          model_supports_batch = true;
        }
        break;
      }
    }
  }

  bool uniform_grid = true;
  if (num_images_ > 1) {
    int64_t t0 = grid_data[0], h0 = grid_data[1], w0 = grid_data[2];
    for (int64_t img = 1; img < num_images_; ++img) {
      if (grid_data[img * 3] != t0 || grid_data[img * 3 + 1] != h0 || grid_data[img * 3 + 2] != w0) {
        uniform_grid = false;
        break;
      }
    }
  }

  if (model_supports_batch && uniform_grid) {
    State::Run(*model_.vision_session_);
    return {};
  }

  OrtValue* pv_full = inputs_[pv_idx];
  OrtValue* feat_full = outputs_[0];

  auto pv_info = pv_full->GetTensorTypeAndShapeInfo();
  auto feat_info = feat_full->GetTensorTypeAndShapeInfo();
  auto pv_shape = pv_info->GetShape();
  auto feat_shape = feat_info->GetShape();
  auto pv_type = pv_info->GetElementType();
  auto feat_type = feat_info->GetElementType();
  int64_t patch_dim = pv_shape[1];
  int64_t hidden_size = feat_shape[1];

  auto element_size = [](ONNXTensorElementDataType type) -> size_t {
    switch (type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return 4;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        return 2;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return 8;
      default:
        throw std::runtime_error("Unsupported pixel_values element type in multi-image vision loop");
    }
  };
  size_t pv_element_size = element_size(pv_type);
  size_t feat_element_size = element_size(feat_type);

  void* pv_raw = pv_full->GetTensorMutableRawData();
  void* feat_raw = feat_full->GetTensorMutableRawData();
  int64_t spatial_merge_size = model_.config_->model.vision.spatial_merge_size;

  auto cpu_mem = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  int64_t total_patches = pv_shape[0];
  int64_t total_feats = feat_shape[0];
  int64_t merge_sq = spatial_merge_size * spatial_merge_size;

  int64_t total_grid_tokens = 0;
  int64_t total_hw = 0;
  int64_t max_grid_tokens = 0;
  for (int64_t img = 0; img < num_images_; ++img) {
    int64_t grid_tokens = grid_data[img * 3] * grid_data[img * 3 + 1] * grid_data[img * 3 + 2];
    total_grid_tokens += grid_tokens;
    total_hw += grid_data[img * 3 + 1] * grid_data[img * 3 + 2];
    max_grid_tokens = std::max(max_grid_tokens, grid_tokens);
  }
  const QwenPatchLayout patch_layout =
      ResolveQwenPatchLayout(total_patches, total_grid_tokens, total_hw, max_grid_tokens, num_images_);

  int64_t expected_total_feats = total_grid_tokens / merge_sq;
  if (total_feats < expected_total_feats)
    throw std::runtime_error("pre-allocated image_features dim 0 (" + std::to_string(total_feats) +
                             ") is smaller than expected (" + std::to_string(expected_total_feats) +
                             ") for " + std::to_string(num_images_) + " images");

  int64_t patch_offset = 0;
  int64_t feat_offset = 0;
  for (int64_t img = 0; img < num_images_; ++img) {
    int64_t t = grid_data[img * 3];
    int64_t h = grid_data[img * 3 + 1];
    int64_t w = grid_data[img * 3 + 2];
    int64_t grid_tokens = t * h * w;
    int64_t num_patches = patch_layout.ImagePatchCount(grid_tokens, h, w);
    int64_t num_feats = grid_tokens / merge_sq;
    int64_t image_patch_offset = patch_layout.ImagePatchOffset(img, patch_offset);

    if (grid_tokens % merge_sq != 0)
      throw std::runtime_error("grid tokens (" + std::to_string(grid_tokens) +
                               ") is not divisible by spatial_merge_size^2 (" +
                               std::to_string(merge_sq) + ") for image " + std::to_string(img));
    if (image_patch_offset + num_patches > total_patches)
      throw std::runtime_error("patch_offset (" + std::to_string(image_patch_offset) + ") + num_patches (" +
                               std::to_string(num_patches) + ") exceeds pixel_values dim 0 (" +
                               std::to_string(total_patches) + ")");
    if (feat_offset + num_feats > total_feats)
      throw std::runtime_error("feat_offset (" + std::to_string(feat_offset) + ") + num_feats (" +
                               std::to_string(num_feats) + ") exceeds image_features dim 0 (" +
                               std::to_string(total_feats) + ")");

    std::vector<int64_t> sub_pv_shape = {num_patches, patch_dim};
    std::vector<int64_t> sub_grid_shape = {1LL, 3LL};
    std::vector<int64_t> sub_feat_shape = {num_feats, hidden_size};

    auto sub_pv = OrtValue::CreateTensor(
        *cpu_mem,
        static_cast<uint8_t*>(pv_raw) + static_cast<size_t>(image_patch_offset * patch_dim) * pv_element_size,
        static_cast<size_t>(num_patches * patch_dim) * pv_element_size,
        std::span<const int64_t>(sub_pv_shape), pv_type);

    auto sub_grid = OrtValue::CreateTensor(
        *cpu_mem,
        const_cast<void*>(static_cast<const void*>(grid_data + img * 3)),
        3 * sizeof(int64_t),
        std::span<const int64_t>(sub_grid_shape),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

    auto sub_feat = OrtValue::CreateTensor(
        *cpu_mem,
        static_cast<uint8_t*>(feat_raw) + static_cast<size_t>(feat_offset * hidden_size) * feat_element_size,
        static_cast<size_t>(num_feats * hidden_size) * feat_element_size,
        std::span<const int64_t>(sub_feat_shape), feat_type);

    inputs_[pv_idx] = sub_pv.get();
    inputs_[grid_idx] = sub_grid.get();
    outputs_[0] = sub_feat.get();

    State::Run(*model_.vision_session_);

    patch_offset += num_patches;
    feat_offset += num_feats;
  }

  inputs_[pv_idx] = pv_full;
  inputs_[grid_idx] = grid_full;
  outputs_[0] = feat_full;

  return {};
}

}  // namespace Generators
