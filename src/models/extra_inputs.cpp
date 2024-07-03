// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lora_adapter.h"
#include "model.h"
#include "extra_inputs.h"

namespace Generators {

ExtraInputs::ExtraInputs(const Model& model, State& state) : model_{model}, state_{state} {
  // We take extra inputs from LoraAdapters
  auto& lora_management = model_.GetLoraAdapterManagement();
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
