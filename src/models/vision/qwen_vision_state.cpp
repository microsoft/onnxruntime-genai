// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "models/vision/qwen_vision_state.h"
#include "models/multi_modal.h"

#include <cstring>

namespace Generators {

DeviceSpan<float> QwenVisionState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (model_.config_->model.vision.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.vision.run_options.value());
  }

  // Single image (or no image data): run the ONNX session directly.
  if (num_images_ <= 1) {
    State::Run(*model_.vision_session_);
    return {};
  }

  // Multi-image: vision.onnx is exported for exactly one image at a time.
  //
  // Dynamo unrolls Python for-loops at export time, so an N-image dummy
  // input would produce a graph that only works for that exact N.  To
  // support a variable number of images we export with N=1 and iterate
  // here in C++, slicing out per-image views of pixel_values and
  // image_grid_thw and writing each result directly into the correct
  // offset of the pre-allocated image_features output buffer.
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
    // Couldn't find expected inputs – fall back to single Run.
    State::Run(*model_.vision_session_);
    return {};
  }

  OrtValue* grid_full = inputs_[grid_idx];
  const int64_t* grid_data = grid_full->GetTensorData<int64_t>();

  const auto grid_shape = grid_full->GetTensorTypeAndShapeInfo()->GetShape();
  const size_t grid_elem_count = grid_full->GetTensorTypeAndShapeInfo()->GetElementCount();
  ValidateImageGridThwLayoutAndCount(grid_shape, grid_elem_count, num_images_, "image_grid_thw");

  // Check if the ONNX model accepts dynamic num_images.
  // A non-positive dim-0 (0 or -1) in the model's input shape = dynamic/symbolic.
  bool model_supports_batch = false;
  {
    auto session_input_names = model_.vision_session_->GetInputNames();
    for (size_t si = 0; si < session_input_names.size(); ++si) {
      if (session_input_names[si] == grid_name) {
        auto grid_input_info = model_.vision_session_->GetInputTypeInfo(si);
        auto grid_expected_shape = grid_input_info->GetTensorTypeAndShapeInfo().GetShape();
        if (!grid_expected_shape.empty() && grid_expected_shape[0] <= 0) {
          model_supports_batch = true;  // dim-0 is symbolic — accepts any N
        }
        break;
      }
    }
  }

  // Check if all images share the same (t, h, w) grid.
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

  // --- Batched single-call path (like HuggingFace) ---
  if (model_supports_batch && uniform_grid) {
    // The model has dynamic image_grid_thw dim-0 and all images share the
    // same grid.  Pass all N images' pixel_values and the full [N, 3]
    // grid_thw in one call — the ONNX graph was vectorized to handle this.
    State::Run(*model_.vision_session_);
    return {};
  }

  // --- Per-image loop path (fallback for different-sized images or static models) ---
  OrtValue* pv_full = inputs_[pv_idx];
  OrtValue* feat_full = outputs_[0];  // pre-allocated image_features output

  // Shapes: pixel_values[total_patches, patch_dim], image_features[total_logical_patches, hidden_size]
  auto pv_info = pv_full->GetTensorTypeAndShapeInfo();
  auto feat_info = feat_full->GetTensorTypeAndShapeInfo();
  auto pv_shape = pv_info->GetShape();
  auto feat_shape = feat_info->GetShape();
  auto pv_type = pv_info->GetElementType();
  auto feat_type = feat_info->GetElementType();
  int64_t patch_dim = pv_shape[1];
  int64_t hidden_size = feat_shape[1];

  // Map ONNX element type to byte size.
  auto element_size = [](ONNXTensorElementDataType type) -> size_t {
    switch (type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return 4;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return 2;
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

  // Detect temporal padding: processor may produce more rows than sum(t*h*w)
  int64_t total_grid_tokens = 0;
  int64_t total_hw = 0;
  for (int64_t img = 0; img < num_images_; ++img) {
    total_grid_tokens += grid_data[img * 3] * grid_data[img * 3 + 1] * grid_data[img * 3 + 2];
    total_hw += grid_data[img * 3 + 1] * grid_data[img * 3 + 2];
  }
  bool temporal_padded = (total_patches != total_grid_tokens && total_hw > 0 && total_patches % total_hw == 0);
  int64_t hw_multiplier = temporal_padded ? (total_patches / total_hw) : 0;

  // Validate that the pre-allocated output buffer is large enough for all images
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
    int64_t num_patches = temporal_padded ? (hw_multiplier * h * w) : grid_tokens;
    int64_t num_feats = grid_tokens / merge_sq;

    if (grid_tokens % merge_sq != 0)
      throw std::runtime_error("grid tokens (" + std::to_string(grid_tokens) +
                               ") is not divisible by spatial_merge_size^2 (" +
                               std::to_string(merge_sq) + ") for image " + std::to_string(img));
    if (patch_offset + num_patches > total_patches)
      throw std::runtime_error("patch_offset (" + std::to_string(patch_offset) + ") + num_patches (" +
                               std::to_string(num_patches) + ") exceeds pixel_values dim 0 (" +
                               std::to_string(total_patches) + ")");
    if (feat_offset + num_feats > total_feats)
      throw std::runtime_error("feat_offset (" + std::to_string(feat_offset) + ") + num_feats (" +
                               std::to_string(num_feats) + ") exceeds image_features dim 0 (" +
                               std::to_string(total_feats) + ")");

    // Create non-owning sub-tensors (zero-copy views into the original buffers).
    std::vector<int64_t> sub_pv_shape = {num_patches, patch_dim};
    std::vector<int64_t> sub_grid_shape = {1LL, 3LL};  // vision.onnx expects [1, 3] per image
    std::vector<int64_t> sub_feat_shape = {num_feats, hidden_size};

    auto sub_pv = OrtValue::CreateTensor(
        *cpu_mem,
        static_cast<uint8_t*>(pv_raw) + static_cast<size_t>(patch_offset * patch_dim) * pv_element_size,
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

    // Temporarily point the State's inputs/output to the per-image slices,
    // run the session, then advance offsets.
    inputs_[pv_idx] = sub_pv.get();
    inputs_[grid_idx] = sub_grid.get();
    outputs_[0] = sub_feat.get();

    State::Run(*model_.vision_session_);

    patch_offset += num_patches;
    feat_offset += num_feats;
  }

  // Restore original pointers so the State remains valid after this call.
  inputs_[pv_idx] = pv_full;
  inputs_[grid_idx] = grid_full;
  outputs_[0] = feat_full;

  return {};
}

}  // namespace Generators
