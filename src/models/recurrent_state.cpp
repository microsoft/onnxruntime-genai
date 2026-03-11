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
  // Name templates for recurrent state inputs/outputs.
  // These follow the same naming convention as KV cache but with .conv_state and .recurrent_state suffixes.
  const auto& past_key_template = model_.config_->model.decoder.inputs.past_key_names;
  const auto& present_key_template = model_.config_->model.decoder.outputs.present_key_names;

  // Derive recurrent name templates from the KV name templates.
  // e.g. "past_key_values.%d.key" -> "past_key_values.%d.conv_state" / "past_key_values.%d.recurrent_state"
  // e.g. "present.%d.key" -> "present.%d.conv_state" / "present.%d.recurrent_state"
  auto derive_template = [](const std::string& kv_template, const std::string& suffix) -> std::string {
    // Find the last '.' before the key/value part and replace it
    auto pos = kv_template.rfind('.');
    if (pos == std::string::npos) return "";
    return kv_template.substr(0, pos + 1) + suffix;
  };

  std::string past_conv_template = derive_template(past_key_template, "conv_state");
  std::string past_recurrent_template = derive_template(past_key_template, "recurrent_state");
  std::string present_conv_template = derive_template(present_key_template, "conv_state");
  std::string present_recurrent_template = derive_template(present_key_template, "recurrent_state");

  if (past_conv_template.empty()) return;

  // Auto-discover recurrent layer indices by probing session inputs.
  // This is O(256) hash map lookups — negligible, done once at construction.
  for (int i = 0; i < 256; ++i) {
    std::string conv_name = ComposeKeyValueName(past_conv_template, i);
    if (model_.session_info_.HasInput(conv_name)) {
      layer_indices_.push_back(i);
    }
  }

  if (layer_indices_.empty()) return;

  if (g_log.enabled)
    Log("info", "RecurrentState: Auto-discovered " + std::to_string(layer_indices_.size()) +
                    " recurrent layers (indices: " + [&]() {
                      std::string s;
                      for (size_t i = 0; i < layer_indices_.size(); ++i) {
                        if (i) s += ",";
                        s += std::to_string(layer_indices_[i]);
                      }
                      return s;
                    }() + ")");

  // Build name strings for all recurrent layers
  for (int idx : layer_indices_) {
    input_name_strings_.push_back(ComposeKeyValueName(past_conv_template, idx));
    input_name_strings_.push_back(ComposeKeyValueName(past_recurrent_template, idx));
    output_name_strings_.push_back(ComposeKeyValueName(present_conv_template, idx));
    output_name_strings_.push_back(ComposeKeyValueName(present_recurrent_template, idx));
  }

  // Discover data types from session info
  conv_type_ = model_.session_info_.GetInputDataType(input_name_strings_[0]);
  recurrent_type_ = model_.session_info_.GetInputDataType(input_name_strings_[1]);

  // Discover shapes from session info and fix batch dimension
  auto fix_batch_dim = [&](std::vector<int64_t> shape) -> std::vector<int64_t> {
    if (!shape.empty() && shape[0] <= 0) {
      shape[0] = state_.params_->BatchBeamSize();
    }
    return shape;
  };

  conv_shape_ = fix_batch_dim(model_.session_info_.GetInputShape(input_name_strings_[0]));
  recurrent_shape_ = fix_batch_dim(model_.session_info_.GetInputShape(input_name_strings_[1]));

  // Precompute element/byte counts (used for allocation and RewindTo)
  size_t conv_elems = 1;
  for (auto d : conv_shape_) conv_elems *= static_cast<size_t>(d);
  size_t recurrent_elems = 1;
  for (auto d : recurrent_shape_) recurrent_elems *= static_cast<size_t>(d);

  conv_bytes_ = conv_elems * Ort::SizeOf(conv_type_);
  recurrent_bytes_ = recurrent_elems * Ort::SizeOf(recurrent_type_);
  per_layer_bytes_ = conv_bytes_ + recurrent_bytes_;

  const int num_layers = static_cast<int>(layer_indices_.size());
  const size_t per_layer_floats = (per_layer_bytes_ + sizeof(float) - 1) / sizeof(float);  // round up
  const size_t total_floats = per_layer_floats * num_layers;

  pasts_.resize(num_layers * 2);
  presents_.reserve(num_layers * 2);

  // Allocate 2 contiguous blocks instead of 72 individual allocations.
  // make_unique<float[]>(N) is value-initialized to zero by the C++ standard.
  past_block_ = std::make_unique<float[]>(total_floats);
  present_block_ = std::make_unique<float[]>(total_floats);
  cpu_mem_info_ = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  // Create OrtValue views into the contiguous blocks.
  // Layout per block: [conv_0 | recurrent_0 | conv_1 | recurrent_1 | ...]
  for (int i = 0; i < num_layers; ++i) {
    auto* past_base = reinterpret_cast<uint8_t*>(past_block_.get()) + i * per_layer_bytes_;
    auto* present_base = reinterpret_cast<uint8_t*>(present_block_.get()) + i * per_layer_bytes_;

    pasts_[i * 2] = OrtValue::CreateTensor(
        *cpu_mem_info_, past_base, conv_bytes_,
        std::vector<int64_t>(conv_shape_.begin(), conv_shape_.end()), conv_type_);
    pasts_[i * 2 + 1] = OrtValue::CreateTensor(
        *cpu_mem_info_, past_base + conv_bytes_, recurrent_bytes_,
        std::vector<int64_t>(recurrent_shape_.begin(), recurrent_shape_.end()), recurrent_type_);

    presents_.push_back(OrtValue::CreateTensor(
        *cpu_mem_info_, present_base, conv_bytes_,
        std::vector<int64_t>(conv_shape_.begin(), conv_shape_.end()), conv_type_));
    presents_.push_back(OrtValue::CreateTensor(
        *cpu_mem_info_, present_base + conv_bytes_, recurrent_bytes_,
        std::vector<int64_t>(recurrent_shape_.begin(), recurrent_shape_.end()), recurrent_type_));
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

  // Recurrent states have fixed shapes (no sequence dimension that grows).
  // We simply swap past <-> present pointers. Zero memory copy.
  const int num_layers = static_cast<int>(layer_indices_.size());
  for (int i = 0; i < num_layers * 2; ++i) {
    std::swap(pasts_[i], presents_[i]);
    state_.inputs_[input_index_ + i] = pasts_[i].get();
    state_.outputs_[output_index_ + i] = presents_[i].get();
  }
}

void RecurrentState::RewindTo(size_t index) {
  if (layer_indices_.empty()) return;

  // For recurrent states, rewinding means resetting to zeros.
  // Unlike KV cache, there's no sequence position to rewind to — the state
  // is either the current accumulated state or zeros (fresh start).
  if (index == 0) {
    const int num_layers = static_cast<int>(layer_indices_.size());

    // Zero both blocks with a single memset each — O(1) calls instead of O(N).
    std::memset(past_block_.get(), 0, per_layer_bytes_ * num_layers);
    std::memset(present_block_.get(), 0, per_layer_bytes_ * num_layers);

    // Recreate OrtValue views (the memory pointers haven't changed, but
    // after Update() swaps the OrtValues may be pointing to the wrong block).
    for (int i = 0; i < num_layers; ++i) {
      auto* past_base = reinterpret_cast<uint8_t*>(past_block_.get()) + i * per_layer_bytes_;
      auto* present_base = reinterpret_cast<uint8_t*>(present_block_.get()) + i * per_layer_bytes_;

      pasts_[i * 2] = OrtValue::CreateTensor(
          *cpu_mem_info_, past_base, conv_bytes_,
          std::vector<int64_t>(conv_shape_.begin(), conv_shape_.end()), conv_type_);
      pasts_[i * 2 + 1] = OrtValue::CreateTensor(
          *cpu_mem_info_, past_base + conv_bytes_, recurrent_bytes_,
          std::vector<int64_t>(recurrent_shape_.begin(), recurrent_shape_.end()), recurrent_type_);

      presents_[i * 2] = OrtValue::CreateTensor(
          *cpu_mem_info_, present_base, conv_bytes_,
          std::vector<int64_t>(conv_shape_.begin(), conv_shape_.end()), conv_type_);
      presents_[i * 2 + 1] = OrtValue::CreateTensor(
          *cpu_mem_info_, present_base + conv_bytes_, recurrent_bytes_,
          std::vector<int64_t>(recurrent_shape_.begin(), recurrent_shape_.end()), recurrent_type_);

      state_.inputs_[input_index_ + i * 2] = pasts_[i * 2].get();
      state_.inputs_[input_index_ + i * 2 + 1] = pasts_[i * 2 + 1].get();
      state_.outputs_[output_index_ + i * 2] = presents_[i * 2].get();
      state_.outputs_[output_index_ + i * 2 + 1] = presents_[i * 2 + 1].get();
    }
  }
  // For index > 0, recurrent state cannot be partially rewound.
  // We leave it as-is (the KV cache handles its own rewind for attention layers).
}

std::unique_ptr<RecurrentState> CreateRecurrentState(State& state) {
  auto recurrent_state = std::make_unique<RecurrentState>(state);
  if (recurrent_state->IsEmpty()) {
    return nullptr;
  }
  return recurrent_state;
}

}  // namespace Generators
