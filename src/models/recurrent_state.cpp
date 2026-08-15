// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "kv_cache.h"  // For ComposeKeyValueName
#include "recurrent_state.h"
#include <algorithm>

namespace Generators {

std::string ComposeCacheName(const std::string& template_string, int index) {
  constexpr int32_t CacheNameLength = 64;
  char cache_name[CacheNameLength];
  if (auto length = snprintf(cache_name, std::size(cache_name), template_string.c_str(), index);
      length < 0 || length >= CacheNameLength) {
    throw std::runtime_error("Unable to compose cache name from the provided template " + template_string +
                             ". This could be either due to an encoding error or the name being too long.");
  }
  return std::string(cache_name);
}

RecurrentState::RecurrentState(State& state)
    : state_{state} {
  // Discover recurrent layer indices by scanning all session input names
  const auto& past_recurrent_template = model_.config_->model.decoder.inputs.past_recurrent_names;
  const auto placeholder_pos = past_recurrent_template.find("%d");
  if (placeholder_pos == std::string::npos) return;

  const auto prefix = past_recurrent_template.substr(0, placeholder_pos);
  const auto suffix = past_recurrent_template.substr(placeholder_pos + 2);

  for (const auto& name : model_.session_info_.GetInputNames()) {
    // Try to match against the recurrent state template (e.g. "past.%d.recurrent")
    // Extract the layer index from names that match
    if (name.size() > prefix.size() + suffix.size() &&
        name.compare(0, prefix.size(), prefix) == 0 &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      auto idx_str = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
      int idx = std::stoi(idx_str);
      layer_indices_.push_back(idx);
    }
  }
  std::sort(layer_indices_.begin(), layer_indices_.end());

  if (layer_indices_.empty()) return;

  if (g_log.enabled)
    Log("info", "RecurrentState: Auto-discovered " + std::to_string(layer_indices_.size()) + " recurrent layers (indices: " + [&]() {
                      std::string s;
                      for (size_t i = 0; i < layer_indices_.size(); ++i) {
                        if (i) s += ",";
                        s += std::to_string(layer_indices_[i]);
                      }
                      return s; }() + ")");

  for (int idx : layer_indices_) {
    input_name_strings_.push_back(ComposeCacheName(model_.config_->model.decoder.inputs.past_conv_names, idx));
    input_name_strings_.push_back(ComposeCacheName(model_.config_->model.decoder.inputs.past_recurrent_names, idx));
    output_name_strings_.push_back(ComposeCacheName(model_.config_->model.decoder.outputs.present_conv_names, idx));
    output_name_strings_.push_back(ComposeCacheName(model_.config_->model.decoder.outputs.present_recurrent_names, idx));
  }

  conv_type_ = model_.session_info_.GetInputDataType(input_name_strings_[0]);
  recurrent_type_ = model_.session_info_.GetInputDataType(input_name_strings_[1]);

  auto fix_batch_dim = [&](std::vector<int64_t> shape) -> std::vector<int64_t> {
    if (!shape.empty() && shape[0] <= 0) {
      shape[0] = state_.params_->BatchBeamSize();
    }
    return shape;
  };

  conv_shape_ = fix_batch_dim(model_.session_info_.GetInputShape(input_name_strings_[0]));
  recurrent_shape_ = fix_batch_dim(model_.session_info_.GetInputShape(input_name_strings_[1]));

  // Validate all dims are positive (only batch dim is expected to be dynamic)
  auto validate_shape = [](const std::vector<int64_t>& shape, const std::string& name) {
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] <= 0)
        throw std::runtime_error("RecurrentState: " + name + " has unsupported dynamic dim " +
                                 std::to_string(shape[i]) + " at axis " + std::to_string(i));
    }
  };
  validate_shape(conv_shape_, "conv");
  validate_shape(recurrent_shape_, "recurrent");

  const int num_layers = static_cast<int>(layer_indices_.size());

  share_buffers_ = state_.params_->IsPastPresentShareBufferEnabled(model_.config_->model.type);

  if (state_.params_->use_graph_capture && !share_buffers_) {
    throw std::runtime_error(
        "Graph capture requires past/present buffer sharing for models with recurrent state. "
        "Ensure past_present_share_buffer=true in genai_config.json and num_beams=1 "
        "(beam search disables buffer sharing).");
  }

  if (!share_buffers_) {
    pasts_.resize(num_layers * 2);
  }
  presents_.reserve(num_layers * 2);

  auto& allocator = model_.p_device_kvcache_->GetAllocator();

  for (int i = 0; i < num_layers; ++i) {
    if (!share_buffers_) {
      pasts_[i * 2] = OrtValue::CreateTensor(allocator, conv_shape_, conv_type_);
      pasts_[i * 2 + 1] = OrtValue::CreateTensor(allocator, recurrent_shape_, recurrent_type_);
    }
    presents_.push_back(OrtValue::CreateTensor(allocator, conv_shape_, conv_type_));
    presents_.push_back(OrtValue::CreateTensor(allocator, recurrent_shape_, recurrent_type_));
  }

  if (!share_buffers_) {
    ZeroStates(pasts_);
  }
  ZeroStates(presents_);
}

void RecurrentState::Add() {
  if (layer_indices_.empty()) return;

  input_index_ = state_.inputs_.size();
  output_index_ = state_.outputs_.size();

  const int num_layers = static_cast<int>(layer_indices_.size());
  for (int i = 0; i < num_layers * 2; ++i) {
    // Shared buffers: alias input=output for stable addresses (required for graph capture).
    // Separate buffers: use distinct past/present allocations with per-step pointer swap.
    state_.inputs_.push_back(share_buffers_ ? presents_[i].get() : pasts_[i].get());
    state_.input_names_.push_back(input_name_strings_[i].c_str());
    state_.outputs_.push_back(presents_[i].get());
    state_.output_names_.push_back(output_name_strings_[i].c_str());
  }
}

void RecurrentState::Update() {
  if (layer_indices_.empty() || share_buffers_) return;

  const int num_layers = static_cast<int>(layer_indices_.size());
  for (int i = 0; i < num_layers * 2; ++i) {
    std::swap(pasts_[i], presents_[i]);
    state_.inputs_[input_index_ + i] = pasts_[i].get();
    state_.outputs_[output_index_ + i] = presents_[i].get();
  }
}

void RecurrentState::RewindTo(size_t index) {
  if (layer_indices_.empty()) return;

  if (index != 0) {
    throw std::runtime_error(
        "RecurrentState::RewindTo(" + std::to_string(index) +
        ") is not supported. Recurrent states cannot be partially rewound.");
  }
  if (share_buffers_) {
    // Shared buffers: zero in place, addresses stay stable.
    ZeroStates(presents_);
  } else {
    // Zero and rebind all state buffers.
    ZeroStates(pasts_);
    ZeroStates(presents_);
    const int num_layers = static_cast<int>(layer_indices_.size());
    for (int i = 0; i < num_layers * 2; ++i) {
      state_.inputs_[input_index_ + i] = pasts_[i].get();
      state_.outputs_[output_index_ + i] = presents_[i].get();
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
