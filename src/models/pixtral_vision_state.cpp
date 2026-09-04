// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "multi_modal.h"

#include <cstring>

namespace Generators {

void PixtralVisionState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs,
                                        const int64_t num_images,
                                        const int64_t num_image_tokens) {
  image_heights_.clear();
  image_widths_.clear();
  for (const auto& input : extra_inputs) {
    if (input.name == Config::Defaults::ImageSizesName && input.tensor->ort_tensor_) {
      auto shape = input.tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape();
      if (shape.size() != 2 || shape[1] != 2)
        throw std::runtime_error(
            "PixtralVisionState: image_sizes must be [N, 2], got [" +
            std::to_string(shape.size() > 0 ? shape[0] : 0) + ", " +
            std::to_string(shape.size() > 1 ? shape[1] : 0) + "]");
      const int64_t* data = input.tensor->ort_tensor_->GetTensorData<int64_t>();
      int64_t n = shape[0];
      for (int64_t i = 0; i < n; ++i) {
        image_heights_.push_back(data[i * 2]);
        image_widths_.push_back(data[i * 2 + 1]);
      }
      break;
    }
  }

  VisionState::SetExtraInputs(extra_inputs, num_images, num_image_tokens);
}

DeviceSpan<float> PixtralVisionState::Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                                          DeviceSpan<int32_t> next_indices) {
  if (model_.config_->model.vision.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.vision.run_options.value());
  }

  if (num_images_ <= 1) {
    State::Run(*model_.vision_session_);
    return {};
  }

  if (image_heights_.empty() || image_widths_.empty()) {
    throw std::runtime_error(
        "PixtralVisionState: multi-image inputs require image_sizes metadata");
  }

  if (static_cast<int64_t>(image_heights_.size()) < num_images_)
    throw std::runtime_error(
        "PixtralVisionState: image_heights_ has " + std::to_string(image_heights_.size()) +
        " entries but num_images_ is " + std::to_string(num_images_));

  if (static_cast<int64_t>(image_widths_.size()) < num_images_)
    throw std::runtime_error(
        "PixtralVisionState: image_widths_ has " + std::to_string(image_widths_.size()) +
        " entries but num_images_ is " + std::to_string(num_images_));

  const std::string& pv_name = model_.config_->model.vision.inputs.pixel_values;

  size_t pv_idx = SIZE_MAX;
  for (size_t i = 0; i < input_names_.size(); ++i) {
    if (input_names_[i] == pv_name) {
      pv_idx = i;
      break;
    }
  }

  if (pv_idx == SIZE_MAX) {
    State::Run(*model_.vision_session_);
    return {};
  }

  OrtValue* pv_full = inputs_[pv_idx];
  OrtValue* feat_full = outputs_[0];

  auto pv_info = pv_full->GetTensorTypeAndShapeInfo();
  auto pv_shape = pv_info->GetShape();
  auto pv_type = pv_info->GetElementType();

  if (pv_shape.size() != 4) {
    throw std::runtime_error(
        "PixtralVisionState: expected 4D pixel_values [N,C,H,W], got " +
        std::to_string(pv_shape.size()) + "D");
  }

  int64_t channels = pv_shape[1];
  int64_t h_max = pv_shape[2];
  int64_t w_max = pv_shape[3];

  auto feat_info = feat_full->GetTensorTypeAndShapeInfo();
  auto feat_shape = feat_info->GetShape();
  auto feat_type = feat_info->GetElementType();
  int64_t hidden_size = feat_shape.back();

  auto element_size = [](ONNXTensorElementDataType type) -> size_t {
    switch (type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return 4;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        return 2;
      default:
        throw std::runtime_error("PixtralVisionState: unsupported element type");
    }
  };
  size_t pv_elem_size = element_size(pv_type);
  size_t feat_elem_size = element_size(feat_type);

  uint8_t* pv_raw = static_cast<uint8_t*>(pv_full->GetTensorMutableRawData());

  int64_t feat_offset = 0;
  size_t image_stride = static_cast<size_t>(channels * h_max * w_max) * pv_elem_size;

  // TODO: Explore batching multiple images through the vision encoder to improve
  // throughput. Currently processes one image at a time due to variable image
  // resolutions producing different patch counts and 2D RoPE position grids.
  // Potential approach: pad to uniform size or restructure the ONNX graph for
  // batched inputs.
  for (int64_t img = 0; img < num_images_; ++img) {
    int64_t h_i = image_heights_[img];
    int64_t w_i = image_widths_[img];

    if (h_i <= 0 || h_i > h_max)
      throw std::runtime_error(
          "PixtralVisionState: image " + std::to_string(img) + " has h_i=" +
          std::to_string(h_i) + " which is out of valid range (0, " +
          std::to_string(h_max) + "]");
    if (w_i <= 0 || w_i > w_max)
      throw std::runtime_error(
          "PixtralVisionState: image " + std::to_string(img) + " has w_i=" +
          std::to_string(w_i) + " which is out of valid range (0, " +
          std::to_string(w_max) + "]");

    std::vector<int64_t> sub_pv_shape = {1, channels, h_i, w_i};
    auto sub_pv = OrtValue::CreateTensor(
        Ort::Allocator::GetWithDefaultOptions(), sub_pv_shape, pv_type);
    uint8_t* sub_pv_data = static_cast<uint8_t*>(sub_pv->GetTensorMutableRawData());

    uint8_t* src_image = pv_raw + img * image_stride;
    size_t dst_offset = 0;
    for (int64_t c = 0; c < channels; ++c) {
      uint8_t* src_channel = src_image + static_cast<size_t>(c * h_max * w_max) * pv_elem_size;
      for (int64_t row = 0; row < h_i; ++row) {
        size_t row_bytes = static_cast<size_t>(w_i) * pv_elem_size;
        std::memcpy(sub_pv_data + dst_offset,
                    src_channel + static_cast<size_t>(row * w_max) * pv_elem_size,
                    row_bytes);
        dst_offset += row_bytes;
      }
    }

    int64_t patch_size = model_.config_->model.vision.patch_size;
    int64_t merge_size = model_.config_->model.vision.spatial_merge_size;
    int64_t num_feats = (h_i / patch_size / merge_size) * (w_i / patch_size / merge_size);

    int64_t total_feats = feat_shape[0];
    if (feat_offset + num_feats > total_feats)
      throw std::runtime_error(
          "PixtralVisionState: feat_offset (" + std::to_string(feat_offset) +
          ") + num_feats (" + std::to_string(num_feats) +
          ") exceeds pre-allocated feature buffer (" + std::to_string(total_feats) + ")");

    std::vector<int64_t> sub_feat_shape = {num_feats, hidden_size};
    auto sub_feat = OrtValue::CreateTensor(
        model_.p_device_->GetAllocator(), sub_feat_shape, feat_type);

    inputs_[pv_idx] = sub_pv.get();
    outputs_[0] = sub_feat.get();

    State::Run(*model_.vision_session_);

    size_t feature_offset_bytes = static_cast<size_t>(feat_offset * hidden_size) * feat_elem_size;
    size_t feature_size_bytes = static_cast<size_t>(num_feats * hidden_size) * feat_elem_size;
    ByteWrapTensor(*model_.p_device_, *feat_full)
        .subspan(feature_offset_bytes, feature_size_bytes)
        .CopyFrom(ByteWrapTensor(*model_.p_device_, *sub_feat));

    feat_offset += num_feats;
  }

  inputs_[pv_idx] = pv_full;
  outputs_[0] = feat_full;

  return {};
}

}  // namespace Generators