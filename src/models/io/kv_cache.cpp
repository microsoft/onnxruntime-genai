// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "generator/generators.h"
#include "../model.h"
#include "conv_kv_cache.h"
#include "kv_cache.h"
#include "static_kv_cache.h"
#include "windowed_kv_cache.h"
#include <algorithm>

namespace Generators {

CombinedKeyValueCache::CombinedKeyValueCache(State& state)
    : state_{state},
      layer_count_{model_.config_->model.decoder.num_hidden_layers},
      shape_{2, state_.params_->BatchBeamSize(), model_.config_->model.decoder.num_key_value_heads, 0, model_.config_->model.decoder.head_size} {
  pasts_.resize(layer_count_);
  presents_.reserve(layer_count_);

  for (int i = 0; i < layer_count_; ++i) {
    input_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.inputs.past_names, i));
    output_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.present_names, i));
  }

  // Derive the KV data type from the KV input 0
  type_ = model_.session_info_.GetInputDataType(input_name_strings_[0]);

  empty_past_ = OrtValue::CreateTensor(Allocator(), shape_, type_);
  shape_[3] = 0;

  for (int i = 0; i < layer_count_; ++i) {
    presents_.push_back(OrtValue::CreateTensor(Allocator(), shape_, type_));
  }
}

void CombinedKeyValueCache::Add() {
  input_index_ = state_.inputs_.size();
  output_index_ = state_.outputs_.size();

  for (int i = 0; i < layer_count_; i++) {
    state_.inputs_.push_back(empty_past_.get());
    state_.input_names_.push_back(input_name_strings_[i].c_str());
    state_.outputs_.push_back(presents_[i].get());
    state_.output_names_.push_back(output_name_strings_[i].c_str());
  }
}

void CombinedKeyValueCache::Update(DeviceSpan<int32_t> beam_indices, int total_length) {
  assert(state_.params_->search.num_beams == 1 || !beam_indices.empty());  // We require beam_indices if we're a beam search

  if (!is_first_update_) {
    for (int i = 0; i < layer_count_; i++) {
      if (beam_indices.empty()) {
        pasts_[i] = std::move(presents_[i]);
      } else {
        PickPastState(beam_indices, i);
      }
      state_.inputs_[input_index_ + i] = pasts_[i].get();
    }
  }

  shape_[3] = total_length;
  for (int i = 0; i < layer_count_; i++) {
    presents_[i] = OrtValue::CreateTensor(Allocator(), shape_, type_);
    state_.outputs_[output_index_ + i] = presents_[i].get();
  }

  is_first_update_ = false;
}

void CombinedKeyValueCache::RewindTo(size_t index) {
  if (shape_[3] <= static_cast<int>(index)) {
    throw std::runtime_error("Requested length of rewind is greater than the current length.");
  }

  is_first_update_ = true;
  if (index == 0) {
    for (int i = 0; i < layer_count_; i++) {
      pasts_[i] = nullptr;
      state_.inputs_[input_index_ + i] = empty_past_.get();
    }
  } else if (type_ == Ort::TypeToTensorType<float>) {
    RewindPastTensorsTo<float>(index);
  } else if (type_ == Ort::TypeToTensorType<int8_t>) {
    RewindPastTensorsTo<int8_t>(index);
  } else if (type_ == Ort::TypeToTensorType<uint8_t> || type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN) {
    RewindPastTensorsTo<uint8_t>(index);
  } else {
    RewindPastTensorsTo<Ort::Float16_t>(index);
  }
}

template <typename T>
void CombinedKeyValueCache::RewindPastTensorsTo(size_t index) {
  assert(index > 0 && shape_[3] >= static_cast<int64_t>(index));
  std::array<int64_t, 5> new_shape = shape_;
  new_shape[3] = static_cast<int>(index);
  auto batch_x_num_heads = new_shape[1] * new_shape[2];
  auto new_length_x_head_size = new_shape[3] * new_shape[4];
  auto old_length_x_head_size = shape_[3] * new_shape[4];
  shape_[3] = new_shape[3];

  for (int i = 0; i < layer_count_; i++) {
    OrtValue& present = *presents_[i];
    std::unique_ptr<OrtValue> past = OrtValue::CreateTensor(Allocator(), shape_, type_);
    auto present_span = KeyValueCacheDetail::WrapKvCacheTensor<T>(Device(), present, type_);
    auto past_span = KeyValueCacheDetail::WrapKvCacheTensor<T>(Device(), *past, type_);

    for (int j = 0; j < 2 * batch_x_num_heads; j++) {
      auto present_data = present_span.subspan(j * old_length_x_head_size, new_length_x_head_size);
      auto past_data = past_span.subspan(j * new_length_x_head_size, new_length_x_head_size);
      past_data.CopyFrom(present_data);
    }
    pasts_[i] = std::move(past);
    state_.inputs_[input_index_ + i] = pasts_[i].get();
  }
}

// Copy present state to past state reordered by the beam_indices
template <typename ScoreType>
void CombinedKeyValueCache::PickPastState(DeviceSpan<int32_t> beam_indices_device, int index) {
  std::span<const int32_t> beam_indices = beam_indices_device.CopyDeviceToCpu();
  auto block_size_per_beam = shape_[2] * shape_[3] * shape_[4];
  auto past_key_size = shape_[1] * block_size_per_beam;

  OrtValue& present = *presents_[index];
  std::unique_ptr<OrtValue> past = OrtValue::CreateTensor(Allocator(), shape_, type_);

  auto past_span = KeyValueCacheDetail::WrapKvCacheTensor<ScoreType>(Device(), *past, type_);
  auto present_span = KeyValueCacheDetail::WrapKvCacheTensor<ScoreType>(Device(), present, type_);

  for (size_t j = 0; j < beam_indices.size(); j++) {
    int32_t beam_index = beam_indices[j];
    auto present_key = present_span.subspan(beam_index * block_size_per_beam, block_size_per_beam);
    auto present_value = present_span.subspan(past_key_size + beam_index * block_size_per_beam, block_size_per_beam);

    auto past_key = past_span.subspan(j * block_size_per_beam, block_size_per_beam);
    auto past_value = past_span.subspan(past_key_size + j * block_size_per_beam, block_size_per_beam);
    past_key.CopyFrom(present_key);
    past_value.CopyFrom(present_value);
  }

  pasts_[index] = std::move(past);
}

void CombinedKeyValueCache::PickPastState(DeviceSpan<int32_t> beam_indices, int index) {
  if (type_ == Ort::TypeToTensorType<float>) {
    PickPastState<float>(beam_indices, index);
  } else if (type_ == Ort::TypeToTensorType<int8_t>) {
    PickPastState<int8_t>(beam_indices, index);
  } else if (type_ == Ort::TypeToTensorType<uint8_t> || type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN) {
    PickPastState<uint8_t>(beam_indices, index);
  } else {
    PickPastState<Ort::Float16_t>(beam_indices, index);
  }
}

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

std::string ComposeKeyValueName(const std::string& template_string, int index) {
  constexpr int32_t KeyValueNameLength = 64;
  char key_value_name[KeyValueNameLength];
  if (auto length = snprintf(key_value_name, std::size(key_value_name), template_string.c_str(), index);
      length < 0 || length >= KeyValueNameLength) {
    throw std::runtime_error("Unable to compose key value name from the provided template " + template_string +
                             ". This could be either due to an encoding error or the name being too long.");
  }
  return std::string(key_value_name);
}

ModelManagedKeyValueCache::ModelManagedKeyValueCache(State& state)
    : state_{state} {
  // A new instance of ModelManagedKeyValueCache is created for each Generator.
  // In this case, we need to trigger a KVCache reset on the session before the first Session::Run.
  // This implies that the key-value cache state is coupled with the ONNX Runtime Session and
  // that only 1 Generator can be active for the Model at any given time.
  RewindTo(0);
}

void ModelManagedKeyValueCache::Add() {}

void ModelManagedKeyValueCache::Update(DeviceSpan<int32_t> beam_indices, int total_length) {
  // Eventually we need to set 'beam_idx' tensor here somehow.
}

void ModelManagedKeyValueCache::RewindTo(size_t index) {
  // Add 'kvcache_rewind' EP dynamic option to get applied before the next Session::Run.
  // This will trim the internal KVCache states to the desired position.
  state_.ep_dynamic_options_next_run_.push_back({"kvcache_rewind", std::to_string(index)});
}

namespace {

bool IsAttentionCacheNeeded(const Model& model) {
  const auto& key_template = model.config_->model.decoder.inputs.past_key_names;
  auto prefix = key_template.substr(0, key_template.find('%'));
  auto suffix = key_template.substr(key_template.find('%') + 2);
  for (const auto& name : model.session_info_.GetInputNames()) {
    if (name.size() > prefix.size() + suffix.size() &&
        name.compare(0, prefix.size(), prefix) == 0 &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
      return true;
  }
  return false;
}

}  // namespace

bool UsesNonRewindableWindowedKeyValueCache(
    const Model& model, const Config::Model::Decoder& decoder) {
  return model.p_device_kvcache_->UsesNonRewindableWindowedKeyValueCache(decoder);
}

std::unique_ptr<KeyValueCache> CreateStandardKeyValueCache(State& state) {
  // LFM2 models interleave attention and conv layers, requiring a cache that handles
  // both KV cache for attention layers and fixed-size conv state for conv layers.
  if (HasConvKeyValueCache(state.model_)) {
    return std::make_unique<ConvKeyValueCache>(state);
  }

  if (!IsAttentionCacheNeeded(state.model_)) {
    return nullptr;
  }

  if (UsesNonRewindableWindowedKeyValueCache(
          state.model_, state.model_.config_->model.decoder)) {
    return std::make_unique<WindowedKeyValueCache>(state);
  }

  return std::make_unique<DefaultKeyValueCache>(state);
}

std::unique_ptr<KeyValueCache> CreateModelManagedKeyValueCache(State& state) {
  if (g_log.enabled)
    Log("info", "CreateKeyValueCache: Creating ModelManagedKeyValueCache");
  return std::make_unique<ModelManagedKeyValueCache>(state);
}

}  // namespace Generators
