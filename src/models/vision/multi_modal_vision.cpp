// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "models/vision/multi_modal_vision.h"
#include "models/multi_modal.h"
#include "models/model_type.h"
#include "models/vision/gemma4_vision_state.h"
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

int64_t VisionState::GetImageFeatureBatchSize(const std::vector<ExtraInput>& extra_inputs) const {
  for (size_t i = 0; i < extra_inputs.size(); ++i) {
    if (extra_inputs[i].name == Config::Defaults::PixelValuesName) {
      assert(extra_inputs[i].tensor->ort_tensor_);
      const auto num_dims = extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape().size();
      if (num_dims < 3) {
        return 0;
      }
      // Rank ≥ 3: batch size is the leading dimension (Phi, Gemma, legacy Qwen).
      return extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape().front();
    }
  }

  return 0;
}

int64_t VisionState::GetNumImageTokens(const std::vector<ExtraInput>& extra_inputs) const {
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
  if (model.config_->model.type == "gemma4") {
    return std::make_unique<Gemma4VisionState>(model, params);
  }
  if (ModelType::IsQwenVLFamily(model.config_->model.type)) {
    return std::make_unique<QwenVisionState>(model, params);
  }
  if (ModelType::IsPixtralFamily(model.config_->model.type)) {
    return std::make_unique<PixtralVisionState>(model, params);
  }
  return std::make_unique<VisionState>(model, params);
}

}  // namespace Generators
