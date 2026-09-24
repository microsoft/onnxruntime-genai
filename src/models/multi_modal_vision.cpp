// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "multi_modal_vision.h"
#include "multi_modal.h"
#include "models/model_type.h"
#include "models/vision/qwen_vision_state.h"
#include "models/vision/pixtral_vision_state.h"

#include <numeric>

namespace Generators {

VisionState::VisionState(const MultiModalLanguageModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {}

void VisionState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_images, const int64_t num_image_tokens) {
  num_image_tokens_ = num_image_tokens;
  num_images_ = num_images;

  image_features_ = std::make_unique<MultiModalFeatures>(*this, MultiModalFeatures::Mode::Output,  // Optional model input
                                                         model_.config_->model.vision.outputs.image_features,
                                                         num_images_, num_image_tokens_);
  image_features_->Add();
  extra_inputs_.Add(extra_inputs, model_.vision_session_->GetInputNames());
}

DeviceSpan<float> VisionState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (model_.config_->model.vision.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.vision.run_options.value());
  }

  State::Run(*model_.vision_session_);
  return {};
}

int64_t GetImageFeatureBatchSize(const std::vector<ExtraInput>& extra_inputs) {
  for (size_t i = 0; i < extra_inputs.size(); ++i) {
    if (extra_inputs[i].name == Config::Defaults::PixelValuesName) {
      assert(extra_inputs[i].tensor->ort_tensor_);
      const auto num_dims = extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape().size();
      if (num_dims < 3) {
        // Some processors flatten pixel_values to [total_patches, patch_dim] (rank 2) so that
        // vision.onnx always receives a single-image-shaped input; num_images cannot be inferred
        // from pixel_values alone — fall through to image_grid_thw.
        break;
      }
      // Rank ≥ 3: batch size is the leading dimension (Phi, Gemma, legacy Qwen).
      return extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape().front();
    }
  }

  // Fallback: image_grid_thw has shape [num_images, 3] so its leading dimension directly gives the
  // image count. This tensor is only produced by processors that flatten pixel_values (e.g.
  // Qwen2.5-VL / Qwen3-VL); for models without it (Phi, Gemma) it is absent and we return 0.
  for (size_t i = 0; i < extra_inputs.size(); ++i) {
    if (extra_inputs[i].name == Config::Defaults::ImageGridThwName) {
      assert(extra_inputs[i].tensor->ort_tensor_);
      const auto shape = extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape();
      const int64_t num_images = shape.empty() ? 0 : shape[0];
      const size_t elem_count = extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetElementCount();
      ValidateImageGridThwLayoutAndCount(shape, elem_count, num_images, "image_grid_thw");
      return num_images;
    }
  }

  return 0;
}

int64_t GetNumImageTokens(const std::vector<ExtraInput>& extra_inputs) {
  for (size_t i = 0; i < extra_inputs.size(); ++i) {
    if (extra_inputs[i].name == Config::Defaults::NumImageTokens) {
      assert(extra_inputs[i].tensor->ort_tensor_);
      const int64_t* num_image_tokens_data = extra_inputs[i].tensor->ort_tensor_->GetTensorData<int64_t>();
      return std::accumulate(num_image_tokens_data,
                             num_image_tokens_data + extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetElementCount(),
                             0LL);
    }
  }

  return 0;
}

std::unique_ptr<VisionState> CreateVisionState(const MultiModalLanguageModel& model, const GeneratorParams& params) {
  if (ModelType::IsQwenVLFamily(model.config_->model.type)) {
    return std::make_unique<QwenVisionState>(model, params);
  }
  if (ModelType::IsPixtralFamily(model.config_->model.type)) {
    return std::make_unique<PixtralVisionState>(model, params);
  }
  return std::make_unique<VisionState>(model, params);
}

}  // namespace Generators
