// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "generator/generators.h"
#include "../model.h"
#include "cross_kv_cache.h"

namespace Generators {

CrossCache::CrossCache(State& state, int sequence_length) {
  const Model& model = state.model_;
  auto& allocator = state.model_.p_device_kvcache_->GetAllocator();
  layer_count_ = model.config_->model.decoder.num_hidden_layers;
  shape_ = std::array<int64_t, 4>{state.params_->BatchBeamSize(), model.config_->model.decoder.num_attention_heads, sequence_length, model.config_->model.decoder.head_size};
  values_.reserve(layer_count_ * 2);

  for (int i = 0; i < layer_count_; ++i) {
    output_name_strings_.emplace_back(ComposeKeyValueName(model.config_->model.encoder.outputs.cross_present_key_names, i));
    output_name_strings_.emplace_back(ComposeKeyValueName(model.config_->model.encoder.outputs.cross_present_value_names, i));

    input_name_strings_.emplace_back(ComposeKeyValueName(model.config_->model.decoder.inputs.cross_past_key_names, i));
    input_name_strings_.emplace_back(ComposeKeyValueName(model.config_->model.decoder.inputs.cross_past_value_names, i));
  }

  // Derive the cross attention KV cache's data type
  type_ = model.session_info_.GetOutputDataType(output_name_strings_[0]);

  for (int i = 0; i < layer_count_; ++i) {
    values_.push_back(OrtValue::CreateTensor(allocator, shape_, type_));
    values_.push_back(OrtValue::CreateTensor(allocator, shape_, type_));
  }
}

void CrossCache::AddOutputs(State& state) {
  for (int i = 0; i < layer_count_ * 2; ++i) {
    state.outputs_.push_back(values_[i].get());
    state.output_names_.push_back(output_name_strings_[i].c_str());
  }
}

void CrossCache::AddInputs(State& state) {
  for (int i = 0; i < layer_count_ * 2; ++i) {
    state.inputs_.push_back(values_[i].get());
    state.input_names_.push_back(input_name_strings_[i].c_str());
  }
}

}  // namespace Generators
