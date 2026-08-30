// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "hybrid_decoder_io.h"

#include <unordered_set>

namespace Generators {

HybridDecoderIO::HybridDecoderIO(std::shared_ptr<DecoderOnly_Model> model,
                                 ScheduledRequests& scheduled_requests,
                                 std::shared_ptr<CacheManager> cache_manager,
                                 const ExecutionContext& execution_context,
                                 size_t position_planes)
    : DecoderIO(model, scheduled_requests, cache_manager),
      varlen_io_{model, scheduled_requests, cache_manager,
                 &execution_context, nullptr, position_planes} {
  // VarlenDecoderIO owns every allocation behind these borrowed pointers and outlives the vectors.
  input_names_ = varlen_io_.input_names_;
  inputs_ = varlen_io_.inputs_;
  output_names_ = varlen_io_.output_names_;
  outputs_ = varlen_io_.outputs_;
  if (input_names_.size() != inputs_.size() ||
      output_names_.size() != outputs_.size()) {
    throw std::runtime_error(
        "Packed decoder binding names and tensors do not match.");
  }
  BindFixedState(execution_context);
}

void HybridDecoderIO::BindFixedState(
    const ExecutionContext& execution_context) {
  if (execution_context.fixed_state_slots.size() != scheduled_requests_.size() ||
      execution_context.fixed_state_bindings.empty()) {
    throw std::runtime_error(
        "Hybrid execution requires fixed state resources for every scheduled request.");
  }

  for (size_t row = 0; row < scheduled_requests_.size(); ++row) {
    if (execution_context.fixed_state_slots[row].request_id !=
        scheduled_requests_[row].get()) {
      throw std::runtime_error(
          "Hybrid fixed state rows do not match scheduled request order.");
    }
  }

  std::unordered_set<std::string_view> input_names;
  std::unordered_set<std::string_view> output_names;
  input_names.reserve(input_names_.size() +
                      execution_context.fixed_state_bindings.size());
  output_names.reserve(output_names_.size() +
                       execution_context.fixed_state_bindings.size());
  for (size_t index = 0; index < input_names_.size(); ++index) {
    const char* name = input_names_[index];
    if (!name || !inputs_[index] || !input_names.insert(name).second) {
      throw std::runtime_error("Packed decoder inputs contain an invalid or duplicate name.");
    }
  }
  for (size_t index = 0; index < output_names_.size(); ++index) {
    const char* name = output_names_[index];
    if (!name || !outputs_[index] || !output_names.insert(name).second) {
      throw std::runtime_error("Packed decoder outputs contain an invalid or duplicate name.");
    }
  }

  for (const auto& binding : execution_context.fixed_state_bindings) {
    if (!binding.input_name || !binding.output_name || !binding.input || !binding.output ||
        !input_names.insert(binding.input_name).second ||
        !output_names.insert(binding.output_name).second) {
      throw std::runtime_error(
          "Hybrid fixed state contains an invalid or duplicate binding.");
    }
    input_names_.push_back(binding.input_name);
    inputs_.push_back(binding.input);
    output_names_.push_back(binding.output_name);
    outputs_.push_back(binding.output);
  }
}

std::vector<DeviceSpan<float>> HybridDecoderIO::ProcessLogits() {
  return varlen_io_.ProcessLogits();
}

}  // namespace Generators
