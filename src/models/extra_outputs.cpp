// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "extra_outputs.h"

namespace Generators {

ExtraOutputs::ExtraOutputs(State& state)
    : state_{state} {}

void ExtraOutputs::Add(const std::vector<std::string>& all_output_names) {
  // Add() should be called after all the outputs managed by GenAI are initialized
  all_output_names_ = all_output_names;
  extra_outputs_start_ = state_.output_names_.size();
  for (const auto& output_name : all_output_names_) {
    if (std::none_of(state_.output_names_.begin(), state_.output_names_.end(),
                     [&](const std::string& elem) { return elem == output_name; })) {
      state_.output_names_.push_back(output_name.c_str());
      state_.outputs_.push_back(nullptr);
    }
  }
}

void ExtraOutputs::Update() {
  for (size_t i = extra_outputs_start_; i < state_.output_names_.size(); ++i) {
    state_.outputs_[i] = nullptr;
  }
}

void ExtraOutputs::RegisterOutputs() {
  for (size_t i = extra_outputs_start_; i < state_.output_names_.size(); ++i) {
    state_.outputs_[i] = (output_ortvalues_[state_.output_names_[i]] = std::unique_ptr<OrtValue>(state_.outputs_[i])).get();
  }
}

}  // namespace Generators
