// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "hybrid_decoder_io.h"

#include <unordered_set>

namespace Generators {

HybridDecoderIO::HybridDecoderIO(std::shared_ptr<DecoderOnly_Model> model,
                                 ScheduledRequests& scheduled_requests,
                                 std::shared_ptr<CacheManager> cache_manager,
                                 const ExecutionContext& execution_context)
    : DecoderIO(model, scheduled_requests, cache_manager),
      execution_context_{execution_context},
      varlen_io_{model, scheduled_requests, cache_manager, &execution_context} {
  // VarlenDecoderIO owns every allocation behind these borrowed pointers and outlives the vectors.
  input_names_ = varlen_io_.input_names_;
  inputs_ = varlen_io_.inputs_;
  output_names_ = varlen_io_.output_names_;
  outputs_ = varlen_io_.outputs_;
  BindFixedState();
}

void HybridDecoderIO::BindFixedState() {
  if (execution_context_.fixed_state_slots.size() != scheduled_requests_.size() ||
      execution_context_.fixed_state_bindings.empty()) {
    throw std::runtime_error(
        "Hybrid execution requires fixed state resources for every scheduled request.");
  }

  for (size_t row = 0; row < scheduled_requests_.size(); ++row) {
    if (execution_context_.fixed_state_slots[row].request_id !=
        scheduled_requests_[row].get()) {
      throw std::runtime_error(
          "Hybrid fixed state rows do not match scheduled request order.");
    }
  }

  std::unordered_set<std::string_view> input_names;
  std::unordered_set<std::string_view> output_names;
  input_names.reserve(input_names_.size() + execution_context_.fixed_state_bindings.size());
  output_names.reserve(output_names_.size() + execution_context_.fixed_state_bindings.size());
  for (const char* name : input_names_) {
    if (!name || !input_names.insert(name).second) {
      throw std::runtime_error("Packed decoder inputs contain an invalid or duplicate name.");
    }
  }
  for (const char* name : output_names_) {
    if (!name || !output_names.insert(name).second) {
      throw std::runtime_error("Packed decoder outputs contain an invalid or duplicate name.");
    }
  }

  const char* state_update_capture_count_name{};
  OrtValue* state_update_capture_count{};
  const char* state_update_active_name{};
  OrtValue* state_update_active{};
  size_t state_update_capacity{};
  for (const auto& binding : execution_context_.fixed_state_bindings) {
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
    if (binding.state_update_capacity == 0) {
      continue;
    }
    if (!binding.state_update_capture_count_name || !binding.state_update_capture_count) {
      throw std::runtime_error(
          "Hybrid fixed state contains incomplete state_update capture-count metadata.");
    }
            if (bool(binding.state_update_active_name) != bool(binding.state_update_active)) {
          throw std::runtime_error(
              "Hybrid fixed state contains incomplete state_update activity metadata.");
            }
    if (!state_update_capture_count_name) {
      if (!input_names.insert(binding.state_update_capture_count_name).second) {
        throw std::runtime_error(
            "Hybrid state_update capture_count collides with another decoder input.");
      }
      state_update_capture_count_name = binding.state_update_capture_count_name;
      state_update_capture_count = binding.state_update_capture_count;
      state_update_capacity = binding.state_update_capacity;
      input_names_.push_back(state_update_capture_count_name);
      inputs_.push_back(state_update_capture_count);
      if (binding.state_update_active_name) {
        if (!input_names.insert(binding.state_update_active_name).second) {
          throw std::runtime_error(
              "Hybrid state_update activity collides with another decoder input.");
        }
        state_update_active_name = binding.state_update_active_name;
        state_update_active = binding.state_update_active;
        input_names_.push_back(state_update_active_name);
        inputs_.push_back(state_update_active);
      }
    } else if (std::string_view{state_update_capture_count_name} !=
                   binding.state_update_capture_count_name ||
               state_update_capture_count != binding.state_update_capture_count ||
               bool(state_update_active_name) != bool(binding.state_update_active_name) ||
               (state_update_active_name &&
                std::string_view{state_update_active_name} != binding.state_update_active_name) ||
               state_update_active != binding.state_update_active ||
               state_update_capacity != binding.state_update_capacity) {
      throw std::runtime_error(
          "Hybrid fixed state bindings disagree on the shared state_update contract.");
    }

    const bool has_state_update_outputs =
      binding.state_update_value || binding.state_update_capsule;
    if (!has_state_update_outputs) {
      continue;
    }
    using StateUpdateKind = Config::Model::Decoder::StateUpdateKind;
    const bool valid_outputs =
        binding.state_update_kind == StateUpdateKind::CausalConv
            ? binding.state_update_value && !binding.state_update_capsule
            : !binding.state_update_value && binding.state_update_capsule;
    if (!valid_outputs) {
      throw std::runtime_error(
          "Hybrid fixed state contains incomplete state_update output tensors.");
    }
    const auto bind_state_update_output = [&](const char* name, OrtValue* value) {
      if (!value) {
        return;
      }
      if (!name || !output_names.insert(name).second) {
        throw std::runtime_error(
            "Hybrid fixed state contains an invalid or duplicate state_update output.");
      }
      output_names_.push_back(name);
      outputs_.push_back(value);
    };
    bind_state_update_output(binding.state_update_value_name, binding.state_update_value);
    bind_state_update_output(binding.state_update_capsule_name, binding.state_update_capsule);
  }
}

std::vector<DeviceSpan<float>> HybridDecoderIO::ProcessLogits() {
  return varlen_io_.ProcessLogits();
}

}  // namespace Generators
