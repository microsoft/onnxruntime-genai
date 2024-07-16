// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lora_adapter.h"
#include "model.h"
#include "extra_inputs.h"

#include <set>

namespace Generators {

ExtraInputs::ExtraInputs(const Model& model, State& state) : model_{model}, state_{state} {
  auto& params = *state_.params_;

  const auto& lora_management = model_.GetLoraAdapterManagement();
  const auto total_inputs = params.extra_inputs.size() + lora_management.GetParamNum();
  extra_input_names_.reserve(total_inputs);
  extra_inputs_.reserve(total_inputs);

  if (state_.GetCapturedGraphInfo()) {
    for (int i = 0; i < params.extra_inputs.size(); ++i) {
      auto type_and_shape_info = params.extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
      const auto& input_name = params.extra_inputs[i].name;

      auto* sb_extra = state_.GetCapturedGraphInfo()->sb_extra_inputs_.at(input_name).get();
      auto ort_value =
          sb_extra->CreateTensorOnStaticBuffer(type_and_shape_info->GetShape(), type_and_shape_info->GetElementType());

      // Copy to value created on top of the StaticBuffer
      CopyToDevice(model_, *params.extra_inputs[1].tensor->ort_tensor_, *ort_value);

      extra_input_names_.push_back(input_name);
      extra_inputs_.push_back(std::move(ort_value));
    }
  } else {
    // We don't use graph capture
    for (auto& extra_input : params.extra_inputs) {
      extra_input_names_.push_back(extra_input.name);
      auto ort_value = DuplicateOrtValue(*extra_input.tensor->ort_tensor_);
      extra_inputs_.push_back(std::move(ort_value));
    }
  }

  // Add Lora Parameters
  const std::set<std::string> adapter_names(params.lora_settings.active_lora_adapters.begin(),
                                            params.lora_settings.active_lora_adapters.end());
  lora_management.OutputAdaptersParameters(adapter_names, std::back_inserter(extra_input_names_),
                                           std::back_inserter(extra_inputs_));
}

void ExtraInputs::Add() {
  // Add extra user inputs to the state
  for (size_t i = 0, lim = extra_input_names_.size(); i < lim; ++i) {
    state_.input_names_.push_back(extra_input_names_[i].c_str());
    state_.inputs_.push_back(extra_inputs_[i].get());
  }
}

}  // namespace Generators
