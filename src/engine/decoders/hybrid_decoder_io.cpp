// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "hybrid_decoder_io.h"

#include <unordered_set>

namespace Generators {

bool HybridDecoderIO::SupportsVariableWidthGraph(const ExecutionContext& context) {
  if (!context.plan || context.plan->requests.empty() ||
      context.fixed_state_binding_key == 0 || context.fixed_state_bindings.empty() ||
      context.fixed_state_slots.size() != context.plan->requests.size())
    return false;

  const auto& shared = context.fixed_state_bindings[0];
  // Compact-update mode keeps GDN's CPU dispatch invariant. Only the exact per-row device
  // counts change on replay; neither counts nor CommitPrefix boundaries are bucketed.
  if (shared.state_update_capacity != 7 ||
      !shared.state_update_capture_count || !shared.state_update_active)
    return false;
  const auto counts = shared.state_update_capture_count->GetTensorTypeAndShapeInfo();
  const auto active = shared.state_update_active->GetTensorTypeAndShapeInfo();
  if (counts->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
      counts->GetShape() != std::vector<int64_t>{static_cast<int64_t>(context.plan->requests.size())} ||
      shared.state_update_capture_count->GetTensorMemoryInfo().GetDeviceType() != OrtMemoryInfoDeviceType_GPU ||
      active->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
      active->GetShape() != std::vector<int64_t>{1} ||
      shared.state_update_active->GetTensorMemoryInfo().GetDeviceType() != OrtMemoryInfoDeviceType_CPU ||
      shared.state_update_active->GetTensorData<int32_t>()[0] != 1)
    return false;

  bool has_drafts = false;
  for (const auto& entry : context.plan->requests) {
    if (entry.is_prefill || entry.unprocessed_token_count == 0 ||
        entry.unprocessed_token_count - 1 != entry.draft_token_count ||
        entry.draft_token_count > shared.state_update_capacity)
      return false;
    has_drafts |= entry.draft_token_count != 0;
  }
  if (!has_drafts)
    return false;

  using Kind = Config::Model::Decoder::StateGroupKind;
  using Update = Config::Model::Decoder::StateUpdateKind;
  for (const auto& binding : context.fixed_state_bindings) {
    const bool conv = binding.kind == Kind::FixedConv &&
                      binding.state_update_kind == Update::CausalConv &&
                      binding.state_update_value && !binding.state_update_capsule;
    const bool recurrent = binding.kind == Kind::FixedRecurrent &&
                           binding.state_update_kind == Update::GatedDeltaNet &&
                           binding.state_update_capsule && !binding.state_update_value;
    if ((!conv && !recurrent) || !binding.input || !binding.output ||
        binding.state_update_capacity != shared.state_update_capacity ||
        binding.state_update_capture_count != shared.state_update_capture_count ||
        binding.state_update_active != shared.state_update_active)
      return false;
  }
  return true;
}

HybridDecoderIO::HybridDecoderIO(std::shared_ptr<DecoderOnly_Model> model,
                                 ScheduledRequests& scheduled_requests,
                                 std::shared_ptr<CacheManager> cache_manager,
                                 const ExecutionContext& execution_context,
                                 VarlenGraphBuffers* graph_buffers,
                                 size_t position_planes,
                                 CpuEmbedding::Workspace* embedding_workspace)
    : DecoderIO(model, scheduled_requests, cache_manager),
      varlen_io_{model, scheduled_requests, cache_manager,
                 &execution_context, graph_buffers, position_planes, embedding_workspace} {
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

  const char* capture_count_name{};
  OrtValue* capture_count{};
  const char* active_name{};
  OrtValue* active{};
  size_t update_capacity{};
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
    if (!capture_count_name) {
      if (!input_names.insert(binding.state_update_capture_count_name).second) {
        throw std::runtime_error(
            "Hybrid state_update capture_count collides with another decoder input.");
      }
      capture_count_name = binding.state_update_capture_count_name;
      capture_count = binding.state_update_capture_count;
      active_name = binding.state_update_active_name;
      active = binding.state_update_active;
      update_capacity = binding.state_update_capacity;
      input_names_.push_back(capture_count_name);
      inputs_.push_back(capture_count);
      if (active_name) {
        if (!input_names.insert(active_name).second) {
          throw std::runtime_error(
              "Hybrid state_update activity collides with another decoder input.");
        }
        input_names_.push_back(active_name);
        inputs_.push_back(active);
      }
    } else if (std::string_view{capture_count_name} != binding.state_update_capture_count_name ||
               capture_count != binding.state_update_capture_count ||
               bool(active_name) != bool(binding.state_update_active_name) ||
               (active_name && std::string_view{active_name} != binding.state_update_active_name) ||
               active != binding.state_update_active ||
               update_capacity != binding.state_update_capacity) {
      throw std::runtime_error(
          "Hybrid fixed state bindings disagree on the shared state_update contract.");
    }

    const bool has_update_outputs =
        binding.state_update_value || binding.state_update_capsule;
    if (!has_update_outputs) {
      continue;
    }
    using StateUpdateKind = Config::Model::Decoder::StateUpdateKind;
    bool valid_outputs = false;
    if (binding.state_update_kind == StateUpdateKind::GatedDeltaNet) {
      valid_outputs = !binding.state_update_value && binding.state_update_capsule;
    } else if (binding.state_update_kind == StateUpdateKind::CausalConv ||
               binding.state_update_kind == StateUpdateKind::Ple ||
               binding.state_update_kind == StateUpdateKind::Indexer) {
      valid_outputs = binding.state_update_value && !binding.state_update_capsule;
    }
    if (!valid_outputs) {
      throw std::runtime_error(
          "Hybrid fixed state contains incomplete state_update output tensors.");
    }
    const auto bind_update_output = [&](const char* name, OrtValue* value) {
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
    bind_update_output(binding.state_update_value_name, binding.state_update_value);
    bind_update_output(binding.state_update_capsule_name, binding.state_update_capsule);
  }
}

std::vector<DeviceSpan<float>> HybridDecoderIO::ProcessLogits() {
  return varlen_io_.ProcessLogits();
}

}  // namespace Generators
