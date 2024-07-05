// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lora_adapter.h"
#include "model.h"
#include "extra_inputs.h"

namespace Generators {

ExtraInputs::ExtraInputs(const Model& model, State& state) : model_{model}, state_{state} {
  auto& lora_management = model_.GetLoraAdapterManagement();
  const auto total_inputs = state_.params_->extra_inputs.size() + lora_management.GetParamNum();
  extra_input_names_.reserve(total_inputs);
  extra_inputs_.reserve(total_inputs);

  if (state_.GetCapturedGraphInfo()) {
    for (int i = 0; i < state_.params_->extra_inputs.size(); ++i) {
      auto type_and_shape_info = state_.params_->extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
      const auto& input_name = state_.params_->extra_inputs[i].name;

      auto* sb_extra = state_.GetCapturedGraphInfo()->sb_extra_inputs_.at(input_name).get();
      auto ort_value =
          sb_extra->CreateTensorOnStaticBuffer(type_and_shape_info->GetShape(), type_and_shape_info->GetElementType());

      // Copy to value created on top of the StaticBuffer
      CopyToDevice(*state_.params_->extra_inputs[1].tensor->ort_tensor_, *ort_value, model_.device_type_,
                   model_.cuda_stream_);

      extra_input_names_.push_back(input_name);
      extra_inputs_.push_back(std::move(ort_value));
    }
  } else {
    // We don't use graph capture
    for (auto& extra_input : state_.params_->extra_inputs) {
      extra_input_names_.push_back(extra_input.name);
      auto ort_value = DuplicateOrtValue(*extra_input.tensor->ort_tensor_);
      extra_inputs_.push_back(std::move(ort_value));
    }
  }

  // Add Lora Parameters
  lora_management.OutputAdaptersParameters(std::back_inserter(extra_input_names_), std::back_inserter(extra_inputs_));
}

void ExtraInputs::Add() {
  // Add extra user inputs to the state
  for (int i = 0, lim = extra_input_names_.size(); i < lim; ++i) {
    state_.input_names_.push_back(extra_input_names_[i].c_str());
    state_.inputs_.push_back(extra_inputs_[i].get());
  }
}

}  // namespace Generators
