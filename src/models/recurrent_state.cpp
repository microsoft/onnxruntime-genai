// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "kv_cache.h"  // For ComposeKeyValueName
#include "recurrent_state.h"
#include <algorithm>
#include <cstring>

namespace Generators {

RecurrentState::RecurrentState(State& state)
    : state_{state} {
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

  // Discover recurrent layer indices by scanning all session input names
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
    input_name_strings_.push_back(ComposeKeyValueName(past_conv_template, idx));
    input_name_strings_.push_back(ComposeKeyValueName(past_recurrent_template, idx));
    output_name_strings_.push_back(ComposeKeyValueName(present_conv_template, idx));
    output_name_strings_.push_back(ComposeKeyValueName(present_recurrent_template, idx));
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
  validate_shape(conv_shape_, "conv_state");
  validate_shape(recurrent_shape_, "recurrent_state");

  size_t conv_elems = 1;
  for (auto d : conv_shape_) conv_elems *= static_cast<size_t>(d);
  size_t recurrent_elems = 1;
  for (auto d : recurrent_shape_) recurrent_elems *= static_cast<size_t>(d);

  conv_bytes_ = conv_elems * Ort::SizeOf(conv_type_);
  recurrent_bytes_ = recurrent_elems * Ort::SizeOf(recurrent_type_);

  // Align offsets to 8 bytes so each tensor's data pointer satisfies its element type alignment
  constexpr size_t kAlignment = 8;
  recurrent_offset_ = (conv_bytes_ + kAlignment - 1) & ~(kAlignment - 1);
  per_layer_stride_ = (recurrent_offset_ + recurrent_bytes_ + kAlignment - 1) & ~(kAlignment - 1);

  const int num_layers = static_cast<int>(layer_indices_.size());
  const size_t total_bytes = per_layer_stride_ * num_layers;

  pasts_.resize(num_layers * 2);
  presents_.reserve(num_layers * 2);

  past_block_ = std::make_unique<uint8_t[]>(total_bytes);
  present_block_ = std::make_unique<uint8_t[]>(total_bytes);
  cpu_mem_info_ = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  for (int i = 0; i < num_layers; ++i) {
    auto* past_base = past_block_.get() + i * per_layer_stride_;
    auto* present_base = present_block_.get() + i * per_layer_stride_;

    pasts_[i * 2] = OrtValue::CreateTensor(
        *cpu_mem_info_, past_base, conv_bytes_, conv_shape_, conv_type_);
    pasts_[i * 2 + 1] = OrtValue::CreateTensor(
        *cpu_mem_info_, past_base + recurrent_offset_, recurrent_bytes_, recurrent_shape_, recurrent_type_);

    presents_.push_back(OrtValue::CreateTensor(
        *cpu_mem_info_, present_base, conv_bytes_, conv_shape_, conv_type_));
    presents_.push_back(OrtValue::CreateTensor(
        *cpu_mem_info_, present_base + recurrent_offset_, recurrent_bytes_, recurrent_shape_, recurrent_type_));
  }
}

void RecurrentState::Add() {
  if (layer_indices_.empty()) return;

  input_index_ = state_.inputs_.size();
  output_index_ = state_.outputs_.size();

  const int num_layers = static_cast<int>(layer_indices_.size());
  for (int i = 0; i < num_layers * 2; ++i) {
    state_.inputs_.push_back(pasts_[i].get());
    state_.input_names_.push_back(input_name_strings_[i].c_str());
    state_.outputs_.push_back(presents_[i].get());
    state_.output_names_.push_back(output_name_strings_[i].c_str());
  }
}

void RecurrentState::Update() {
  if (layer_indices_.empty()) return;

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
    // Recurrent states cannot be partially rewound — they are compressed summaries
    // with no per-position history. Non-zero rewind is a no-op; the state remains unchanged.
    if (g_log.enabled)
      Log("warning", "RecurrentState::RewindTo(" + std::to_string(index) +
                         ") is a no-op. Recurrent states cannot be partially rewound.");
    return;
  }

  const int num_layers = static_cast<int>(layer_indices_.size());

  std::memset(past_block_.get(), 0, per_layer_stride_ * num_layers);
  std::memset(present_block_.get(), 0, per_layer_stride_ * num_layers);

  // Recreate OrtValue views after swap
  for (int i = 0; i < num_layers; ++i) {
    auto* past_base = past_block_.get() + i * per_layer_stride_;
    auto* present_base = present_block_.get() + i * per_layer_stride_;

    pasts_[i * 2] = OrtValue::CreateTensor(
        *cpu_mem_info_, past_base, conv_bytes_, conv_shape_, conv_type_);
    pasts_[i * 2 + 1] = OrtValue::CreateTensor(
        *cpu_mem_info_, past_base + recurrent_offset_, recurrent_bytes_, recurrent_shape_, recurrent_type_);

    presents_[i * 2] = OrtValue::CreateTensor(
        *cpu_mem_info_, present_base, conv_bytes_, conv_shape_, conv_type_);
    presents_[i * 2 + 1] = OrtValue::CreateTensor(
        *cpu_mem_info_, present_base + recurrent_offset_, recurrent_bytes_, recurrent_shape_, recurrent_type_);

    state_.inputs_[input_index_ + i * 2] = pasts_[i * 2].get();
    state_.inputs_[input_index_ + i * 2 + 1] = pasts_[i * 2 + 1].get();
    state_.outputs_[output_index_ + i * 2] = presents_[i * 2].get();
    state_.outputs_[output_index_ + i * 2 + 1] = presents_[i * 2 + 1].get();
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
