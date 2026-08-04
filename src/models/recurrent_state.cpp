// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "kv_cache.h"  // For ComposeKeyValueName
#include "recurrent_state.h"
#include <algorithm>

namespace Generators {

RecurrentState::RecurrentState(State& state)
    : state_{state} {
  input_name_strings_ = model_.config_->model.decoder.inputs.past_state_names;
  output_name_strings_ = model_.config_->model.decoder.outputs.present_state_names;
  dynamic_states_ = !input_name_strings_.empty();
  if (input_name_strings_.size() != output_name_strings_.size()) {
    throw std::runtime_error("RecurrentState: past_state_names and present_state_names must have the same size.");
  }

  if (!input_name_strings_.empty()) {
    for (const auto& name : input_name_strings_) {
      if (!model_.session_info_.HasInput(name)) {
        throw std::runtime_error("RecurrentState: configured input '" + name + "' was not found in the model.");
      }
    }
  } else {
  const auto& past_key_template = model_.config_->model.decoder.inputs.past_key_names;
  const auto& present_key_template = model_.config_->model.decoder.outputs.present_key_names;

  // Derive recurrent name templates from KV name templates
  auto derive_template = [](const std::string& kv_template, const std::string& suffix) -> std::string {
    auto pos = kv_template.rfind('.');
    if (pos == std::string::npos) return "";
    return kv_template.substr(0, pos + 1) + suffix;
  };

  std::string past_conv_template = derive_template(past_key_template, "conv_state");
  std::string past_recurrent_template = derive_template(past_key_template, "recurrent_state");
  std::string present_conv_template = derive_template(present_key_template, "conv_state");
  std::string present_recurrent_template = derive_template(present_key_template, "recurrent_state");

  if (past_conv_template.empty()) return;

  std::vector<int> layer_indices;
  // Discover recurrent layer indices by scanning all session input names.
  for (const auto& name : model_.session_info_.GetInputNames()) {
    // Try to match against the conv_state template (e.g. "past_key_values.%d.conv_state")
    // Extract the layer index from names that match
    auto prefix = past_conv_template.substr(0, past_conv_template.find('%'));
    auto suffix = past_conv_template.substr(past_conv_template.find('%') + 2);  // skip %d
    if (name.size() > prefix.size() + suffix.size() &&
        name.compare(0, prefix.size(), prefix) == 0 &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      auto idx_str = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
      int idx = std::stoi(idx_str);
      layer_indices.push_back(idx);
    }
  }
  std::sort(layer_indices.begin(), layer_indices.end());

  if (layer_indices.empty()) return;

  if (g_log.enabled)
    Log("info", "RecurrentState: Auto-discovered " + std::to_string(layer_indices.size()) + " recurrent layers (indices: " + [&]() {
                      std::string s;
                      for (size_t i = 0; i < layer_indices.size(); ++i) {
                        if (i) s += ",";
                        s += std::to_string(layer_indices[i]);
                      }
                      return s; }() + ")");

  for (int idx : layer_indices) {
    input_name_strings_.push_back(ComposeKeyValueName(past_conv_template, idx));
    input_name_strings_.push_back(ComposeKeyValueName(past_recurrent_template, idx));
    output_name_strings_.push_back(ComposeKeyValueName(present_conv_template, idx));
    output_name_strings_.push_back(ComposeKeyValueName(present_recurrent_template, idx));
  }
  }

  auto fix_batch_dim = [&](std::vector<int64_t> shape) -> std::vector<int64_t> {
    if (!shape.empty() && shape[0] <= 0) {
      shape[0] = state_.params_->BatchBeamSize();
    }
    return shape;
  };

  auto validate_shape = [](const std::vector<int64_t>& shape, const std::string& name) {
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] <= 0)
        throw std::runtime_error("RecurrentState: " + name + " has unsupported dynamic dim " +
                                 std::to_string(shape[i]) + " at axis " + std::to_string(i));
    }
  };
  types_.reserve(input_name_strings_.size());
  shapes_.reserve(input_name_strings_.size());
  for (const auto& name : input_name_strings_) {
    types_.push_back(model_.session_info_.GetInputDataType(name));
    shapes_.push_back(fix_batch_dim(model_.session_info_.GetInputShape(name)));
    if (dynamic_states_) {
      for (size_t axis = 1; axis < shapes_.back().size(); ++axis) {
        if (shapes_.back()[axis] <= 0) shapes_.back()[axis] = 0;
      }
    } else {
      validate_shape(shapes_.back(), name);
    }
  }

  share_buffers_ = state_.params_->IsPastPresentShareBufferEnabled(model_.config_->model.type);

  if (dynamic_states_) {
    share_buffers_ = false;
    if (state_.params_->search.num_beams != 1) {
      throw std::runtime_error("Beam search is not supported for variable-length recurrent states.");
    }
    if (state_.params_->use_graph_capture) {
      throw std::runtime_error("Graph capture is not supported for variable-length recurrent states.");
    }
  }

  if (state_.params_->use_graph_capture && !share_buffers_) {
    throw std::runtime_error(
        "Graph capture requires past/present buffer sharing for models with recurrent state. "
        "Ensure past_present_share_buffer=true in genai_config.json and num_beams=1 "
        "(beam search disables buffer sharing).");
  }

  if (!share_buffers_) {
    pasts_.resize(input_name_strings_.size());
  }
  presents_.reserve(input_name_strings_.size());

  auto& allocator = model_.p_device_kvcache_->GetAllocator();

  for (size_t i = 0; i < input_name_strings_.size(); ++i) {
    if (!share_buffers_) {
      pasts_[i] = OrtValue::CreateTensor(allocator, shapes_[i], types_[i]);
    }
    if (!dynamic_states_) {
      presents_.push_back(OrtValue::CreateTensor(allocator, shapes_[i], types_[i]));
    }
  }

  if (!share_buffers_) {
    ZeroStates(pasts_);
  }
  ZeroStates(presents_);
}

RecurrentState::~RecurrentState() {
  if (!dynamic_states_ || output_index_ == ~0U) return;
  for (size_t i = 0; i < input_name_strings_.size(); ++i) {
    std::unique_ptr<OrtValue>{state_.outputs_[output_index_ + i]}.reset();
    state_.outputs_[output_index_ + i] = nullptr;
  }
}

void RecurrentState::Add() {
  if (input_name_strings_.empty()) return;

  input_index_ = state_.inputs_.size();
  output_index_ = state_.outputs_.size();

  for (size_t i = 0; i < input_name_strings_.size(); ++i) {
    // Shared buffers: alias input=output for stable addresses (required for graph capture).
    // Separate buffers: use distinct past/present allocations with per-step pointer swap.
    state_.inputs_.push_back(share_buffers_ ? presents_[i].get() : pasts_[i].get());
    state_.input_names_.push_back(input_name_strings_[i].c_str());
    state_.outputs_.push_back(dynamic_states_ ? nullptr : presents_[i].get());
    state_.output_names_.push_back(output_name_strings_[i].c_str());
  }
}

void RecurrentState::Update() {
  if (input_name_strings_.empty() || share_buffers_) return;

  for (size_t i = 0; i < input_name_strings_.size(); ++i) {
    if (dynamic_states_) {
      if (state_.outputs_[output_index_ + i] != nullptr) {
        pasts_[i].reset(state_.outputs_[output_index_ + i]);
        state_.inputs_[input_index_ + i] = pasts_[i].get();
        state_.outputs_[output_index_ + i] = nullptr;
      }
    } else {
      std::swap(pasts_[i], presents_[i]);
      state_.inputs_[input_index_ + i] = pasts_[i].get();
      state_.outputs_[output_index_ + i] = presents_[i].get();
    }
  }
}

void RecurrentState::RewindTo(size_t index) {
  if (input_name_strings_.empty()) return;

  if (index != 0) {
    throw std::runtime_error(
        "RecurrentState::RewindTo(" + std::to_string(index) +
        ") is not supported. Recurrent states cannot be partially rewound.");
  }
  if (share_buffers_) {
    // Shared buffers: zero in place, addresses stay stable.
    ZeroStates(presents_);
  } else {
    if (dynamic_states_) {
      auto& allocator = model_.p_device_kvcache_->GetAllocator();
      for (size_t i = 0; i < pasts_.size(); ++i) {
        std::unique_ptr<OrtValue>{state_.outputs_[output_index_ + i]}.reset();
        pasts_[i] = OrtValue::CreateTensor(allocator, shapes_[i], types_[i]);
      }
    } else {
      ZeroStates(pasts_);
      ZeroStates(presents_);
    }
    for (size_t i = 0; i < input_name_strings_.size(); ++i) {
      state_.inputs_[input_index_ + i] = pasts_[i].get();
      state_.outputs_[output_index_ + i] = dynamic_states_ ? nullptr : presents_[i].get();
    }
  }
}

void RecurrentState::ZeroStates(std::vector<std::unique_ptr<OrtValue>>& states) {
  auto& device = *model_.p_device_kvcache_;
  for (auto& val : states) {
    ByteWrapTensor(device, *val).Zero();
  }
}

std::unique_ptr<RecurrentState> CreateRecurrentState(State& state) {
  auto recurrent_state = std::make_unique<RecurrentState>(state);
  if (recurrent_state->IsEmpty()) {
    return nullptr;
  }
  return recurrent_state;
}

}  // namespace Generators
