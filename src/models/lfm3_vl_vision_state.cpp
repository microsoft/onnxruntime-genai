// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "lfm3_vl_vision_state.h"

#include <algorithm>
#include <array>

namespace Generators {

namespace {

struct OrtValuePointerRestore {
  OrtValue*& slot;
  OrtValue* value;

  ~OrtValuePointerRestore() { slot = value; }
};

}  // namespace

Lfm3VlVisionState::Lfm3VlVisionState(const MultiModalLanguageModel& model, const GeneratorParams& params)
    : VisionState{model, params} {
  // The processor fills one sequence and nothing pads or repeats image features across rows or beams.
  if (params.search.batch_size > 1 || params.search.num_beams > 1) {
    throw std::runtime_error("lfm3_vl supports batch_size 1 and num_beams 1 only; got batch_size " +
                             std::to_string(params.search.batch_size) + " and num_beams " +
                             std::to_string(params.search.num_beams) + ".");
  }
  if (model.config_->model.vision.spatial_merge_size <= 0) {
    throw std::runtime_error("lfm3_vl: model.vision.spatial_merge_size must be positive, got " +
                             std::to_string(model.config_->model.vision.spatial_merge_size) + ".");
  }
  // The shared image count reads image_grid_thw by its default name, so a renamed input would count no images.
  const auto& inputs = model.config_->model.vision.inputs;
  if (inputs.pixel_values != Config::Defaults::PixelValuesName ||
      inputs.image_grid_thw != Config::Defaults::ImageGridThwName) {
    throw std::runtime_error("lfm3_vl: model.vision.inputs must keep the names pixel_values and image_grid_thw, got " +
                             inputs.pixel_values + " and " + inputs.image_grid_thw + ".");
  }
}

void Lfm3VlVisionState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_images,
                                       const int64_t num_image_tokens) {
  grid_thw_.clear();
  int64_t pixel_rows = 0;
  for (const auto& input : extra_inputs) {
    if (input.name == Config::Defaults::ImageGridThwName) {
      if (input.tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        throw std::runtime_error("lfm3_vl: image_grid_thw must be int64.");
      }
      const int64_t* grid = input.tensor->ort_tensor_->GetTensorData<int64_t>();
      grid_thw_.assign(grid, grid + input.tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetElementCount());
    } else if (input.name == Config::Defaults::PixelValuesName) {
      const auto shape = input.tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape();
      if (shape.size() != 2) {
        throw std::runtime_error("lfm3_vl: pixel_values must be [num_patches, row_width], got rank " +
                                 std::to_string(shape.size()) + ".");
      }
      pixel_rows = shape[0];
    }
  }

  // The grid decides how each run slices the rows, so it must account for every row and image token.
  const int64_t merge_size = model_.config_->model.vision.spatial_merge_size;
  int64_t grid_rows = 0;
  int64_t grid_tokens = 0;
  for (size_t i = 0; i < grid_thw_.size(); i += 3) {
    const int64_t t = grid_thw_[i], h = grid_thw_[i + 1], w = grid_thw_[i + 2];
    if (t != 1 || h <= 0 || w <= 0 || h > pixel_rows || w > pixel_rows || h % merge_size != 0 || w % merge_size != 0) {
      throw std::runtime_error("lfm3_vl: image_grid_thw row " + std::to_string(i / 3) + " is [" + std::to_string(t) +
                               ", " + std::to_string(h) + ", " + std::to_string(w) + "]; expected [1, h, w] with h and w " +
                               "positive multiples of " + std::to_string(merge_size) + ".");
    }
    grid_rows += h * w;
    grid_tokens += h * w / (merge_size * merge_size);
  }
  if (grid_rows != pixel_rows || grid_tokens != num_image_tokens) {
    throw std::runtime_error("lfm3_vl: image_grid_thw describes " + std::to_string(grid_rows) + " pixel rows and " +
                             std::to_string(grid_tokens) + " image tokens, but pixel_values has " +
                             std::to_string(pixel_rows) + " rows and num_image_tokens is " +
                             std::to_string(num_image_tokens) + ".");
  }

  VisionState::SetExtraInputs(extra_inputs, num_images, num_image_tokens);
}

DeviceSpan<float> Lfm3VlVisionState::Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                                         DeviceSpan<int32_t> next_indices) {
  if (num_images_ <= 1) {
    return VisionState::Run(current_length, next_tokens, next_indices);
  }
  if (model_.config_->model.vision.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.vision.run_options.value());
  }

  const auto find_input = [this](const std::string& name) {
    const auto it = std::find(input_names_.begin(), input_names_.end(), name);
    if (it == input_names_.end()) {
      throw std::runtime_error("lfm3_vl: the vision model has no bound input named \"" + name + "\".");
    }
    return static_cast<size_t>(it - input_names_.begin());
  };
  const size_t pixel_index = find_input(model_.config_->model.vision.inputs.pixel_values);
  const size_t grid_index = find_input(model_.config_->model.vision.inputs.image_grid_thw);
  OrtValue* pixel_values = inputs_[pixel_index];
  OrtValue* grid = inputs_[grid_index];
  OrtValue* features = outputs_[0];
  // Each run binds one image's tensors, which die with their loop iteration.
  const OrtValuePointerRestore restore_pixel_values{inputs_[pixel_index], pixel_values};
  const OrtValuePointerRestore restore_grid{inputs_[grid_index], grid};
  const OrtValuePointerRestore restore_features{outputs_[0], features};

  const auto pixel_info = pixel_values->GetTensorTypeAndShapeInfo();
  const auto pixel_type = pixel_info->GetElementType();
  const int64_t row_width = pixel_info->GetShape()[1];
  const size_t row_bytes = static_cast<size_t>(row_width) * Ort::SizeOf(pixel_type);
  const auto feature_info = features->GetTensorTypeAndShapeInfo();
  const auto feature_type = feature_info->GetElementType();
  const int64_t hidden_size = feature_info->GetShape().back();
  const size_t token_bytes = static_cast<size_t>(hidden_size) * Ort::SizeOf(feature_type);
  const int64_t merge_square = model_.config_->model.vision.spatial_merge_size * model_.config_->model.vision.spatial_merge_size;

  // The pixel rows and grid are the caller's CPU tensors; the features live on the vision session's device.
  auto cpu_memory = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  auto* pixel_data = static_cast<uint8_t*>(pixel_values->GetTensorMutableRawData());
  auto feature_bytes = ByteWrapTensor(*p_session_device_, *features);
  size_t row_offset = 0;
  size_t token_offset = 0;
  for (int64_t image = 0; image < num_images_; ++image) {
    int64_t* image_grid = grid_thw_.data() + image * 3;
    const int64_t rows = image_grid[1] * image_grid[2];
    const int64_t tokens = rows / merge_square;
    const std::array<int64_t, 2> pixel_shape{rows, row_width};
    const std::array<int64_t, 2> grid_shape{1, 3};
    const std::array<int64_t, 2> feature_shape{tokens, hidden_size};
    auto image_pixels = OrtValue::CreateTensor(*cpu_memory, pixel_data + row_offset * row_bytes,
                                               static_cast<size_t>(rows) * row_bytes, pixel_shape, pixel_type);
    auto image_grid_value = OrtValue::CreateTensor(*cpu_memory, image_grid, 3 * sizeof(int64_t), grid_shape,
                                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    auto image_features = OrtValue::CreateTensor(p_session_device_->GetAllocator(), feature_shape, feature_type);

    inputs_[pixel_index] = image_pixels.get();
    inputs_[grid_index] = image_grid_value.get();
    outputs_[0] = image_features.get();
    State::Run(*model_.vision_session_);

    const size_t image_bytes = static_cast<size_t>(tokens) * token_bytes;
    feature_bytes.subspan(token_offset * token_bytes, image_bytes)
        .CopyFrom(ByteWrapTensor(*p_session_device_, *image_features));
    row_offset += static_cast<size_t>(rows);
    token_offset += static_cast<size_t>(tokens);
  }
  return {};
}

}  // namespace Generators
