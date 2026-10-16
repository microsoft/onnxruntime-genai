// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "generator/generators.h"
#include "models/model.h"

namespace Generators {

// A completed decoder forward, not token count or position, makes its caches valid.
class CachePhaseInput {
 public:
  CachePhaseInput(State& state, OrtSession& decoder) : state_{state} {
    const auto& name = state_.model_.config_->model.decoder.inputs.has_previous_state;
    if (name.empty()) return;
    SessionInfo info;
    info.Add(decoder);
    if (!info.HasInput(name))
      throw std::runtime_error("has_previous_state decoder input not found: " + name);
    if (info.GetInputDataType(name) != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL ||
        !info.GetInputShape(name).empty())
      throw std::runtime_error("has_previous_state decoder input must be a BOOL scalar []: " + name);
    if (state_.params_->use_graph_capture || state_.params_->search.chunk_size.value_or(0) != 0 ||
        state_.params_->BatchBeamSize() != 1 || state_.params_->speculative.ngram_size > 0)
      throw std::runtime_error("has_previous_state requires batch1/beam1, no graph capture, scheduler chunking or speculative decoding");
    value_ = OrtValue::CreateTensor<bool>(state_.model_.allocator_cpu_, std::span<const int64_t>{});
    *value_->GetTensorMutableData<bool>() = false;
  }

  void Add() {
    if (!value_) return;
    const auto& name = state_.model_.config_->model.decoder.inputs.has_previous_state;
    for (const auto* existing : state_.input_names_)
      if (name == existing)
        throw std::runtime_error("Duplicate managed has_previous_state input: " + name);
    state_.input_names_.push_back(name.c_str());
    state_.inputs_.push_back(value_.get());
  }

  void ValidateExtraInputs(const std::vector<ExtraInput>& inputs) const {
    if (!value_) return;
    const auto& name = state_.model_.config_->model.decoder.inputs.has_previous_state;
    for (const auto& input : inputs)
      if (input.name == name)
        throw std::runtime_error("has_previous_state is runtime-owned; duplicate user input: " + name);
  }

  void Begin(size_t sequence_length) {
    if (!value_) return;
    if (in_flight_)
      throw std::runtime_error("has_previous_state decoder execution failed; create a new Generator");
    // Remains set on an exception: cache writes and search preparation are not transactional.
    in_flight_ = true;
    if (valid_ && sequence_length != 1)
      throw std::runtime_error("has_previous_state cached decoding requires S==1");
    *value_->GetTensorMutableData<bool>() = valid_;
  }

  void Commit() {
    if (!value_) return;
    valid_ = true;
    in_flight_ = false;
  }

 private:
  State& state_;
  std::unique_ptr<OrtValue> value_;
  bool valid_{};
  bool in_flight_{};
};

inline void RejectCachePhaseOperation(const Config& config, const char* operation) {
  if (!config.model.decoder.inputs.has_previous_state.empty())
    throw std::runtime_error(std::string{"has_previous_state does not support "} + operation);
}

}  // namespace Generators
