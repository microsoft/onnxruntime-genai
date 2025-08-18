// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "windowed_kv_cache.h"

#include "../generators.h"
#include "../logging.h"
#include "../make_string.h"
#include "../narrow.h"
#include "model.h"
#include "threadpool.h"
#include "utils.h"

namespace Generators {

namespace {

std::vector<size_t> MakeAllLayerIndices(size_t layer_count) {
  std::vector<size_t> v(layer_count);
  std::iota(v.begin(), v.end(), size_t{0});
  return v;
}

}  // namespace

std::vector<WindowedKeyValueCache::LayerState> WindowedKeyValueCache::MakeInitialPerLayerStates(
    size_t layer_count,
    size_t initial_window_size,
    const CacheTensorShape& initial_key_cache_shape_in,
    const CacheTensorShape& initial_key_cache_shape_out,
    const CacheTensorShape& initial_value_cache_shape_in,
    const CacheTensorShape& initial_value_cache_shape_out) {
  std::vector<LayerState> per_layer_states{};
  per_layer_states.reserve(layer_count);

  for (size_t i = 0; i < layer_count; ++i) {
    LayerState layer_state{};
    layer_state.window_size = initial_window_size;
    layer_state.key_cache_shape_in = initial_key_cache_shape_in;
    layer_state.value_cache_shape_in = initial_value_cache_shape_in;
    layer_state.key_cache_shape_out = initial_key_cache_shape_out;
    layer_state.value_cache_shape_out = initial_value_cache_shape_out;
    per_layer_states.push_back(std::move(layer_state));
  }

  return per_layer_states;
}

WindowedKeyValueCache::WindowedKeyValueCache(State& state)
    : state_{state},
      layer_count_{narrow<size_t>(model_.config_->model.decoder.num_hidden_layers)},
      all_layer_indices_(MakeAllLayerIndices(layer_count_)) {
  if (layer_count_ == 0) {
    throw std::runtime_error("Expected there to be at least 1 layer in the model. Actual: " +
                             std::to_string(layer_count_) + ". Please check the num_hidden_layers attribute in the model configuration.");
  }

  const auto initial_window_size = model_.config_->model.decoder.sliding_window->window_size;

  if (initial_window_size <= 1) {
    throw std::runtime_error("Initial window size must be greater than 1. Actual: " +
                             std::to_string(initial_window_size) +
                             ". Please check the sliding_window.window_size attribute in the model configuration.");
  }

  const auto num_key_value_heads = model_.config_->model.decoder.num_key_value_heads;
  const auto head_size = model_.config_->model.decoder.head_size;
  const auto context_length = model_.config_->model.context_length;

  const auto initial_key_cache_shape_in =
      CacheTensorShape{num_key_value_heads, 1, head_size, context_length - initial_window_size};

  const auto initial_key_cache_shape_out =
      CacheTensorShape{num_key_value_heads, 1, head_size, initial_window_size};

  const auto initial_value_cache_shape_in =
      CacheTensorShape{num_key_value_heads, 1, context_length - initial_window_size, head_size};

  const auto initial_value_cache_shape_out =
      CacheTensorShape{num_key_value_heads, 1, initial_window_size, head_size};

  per_layer_states_ = MakeInitialPerLayerStates(layer_count_, static_cast<size_t>(initial_window_size),
                                                initial_key_cache_shape_in, initial_key_cache_shape_out,
                                                initial_value_cache_shape_in, initial_value_cache_shape_out);

  for (int i = 0; i < static_cast<int>(layer_count_); ++i) {
    input_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.inputs.past_key_names, i));
    input_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.inputs.past_value_names, i));

    output_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.present_key_names, i));
    output_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.present_value_names, i));
  }

  type_ = model_.session_info_.GetInputDataType(input_name_strings_[0]);
  if (type_ != Ort::TypeToTensorType<uint8_t>) {
    throw std::runtime_error("Expected input data type to be uint8_t for WindowedKeyValueCache. Actual: " +
                             std::to_string(type_));
  }

  for (size_t i = 0; i < layer_count_; ++i) {
    key_caches_in_.push_back(
        OrtValue::CreateTensor(Allocator(), initial_key_cache_shape_in, type_));
    std::fill_n(key_caches_in_[i]->GetTensorMutableData<uint8_t>(),
                ElementCountFromShape(initial_key_cache_shape_in),
                static_cast<uint8_t>(model_.config_->model.decoder.sliding_window->pad_value));

    value_caches_in_.push_back(
        OrtValue::CreateTensor(Allocator(), initial_value_cache_shape_in, type_));
    std::fill_n(value_caches_in_[i]->GetTensorMutableData<uint8_t>(),
                ElementCountFromShape(initial_value_cache_shape_in),
                static_cast<uint8_t>(model_.config_->model.decoder.sliding_window->pad_value));

    key_caches_out_.push_back(
        OrtValue::CreateTensor(Allocator(), initial_key_cache_shape_out, type_));
    value_caches_out_.push_back(
        OrtValue::CreateTensor(Allocator(), initial_value_cache_shape_out, type_));
  }
}

void WindowedKeyValueCache::Add() {
  input_index_ = state_.inputs_.size();
  output_index_ = state_.outputs_.size();

  for (size_t layer_idx = 0; layer_idx < layer_count_; ++layer_idx) {
    state_.inputs_.push_back(key_caches_in_[layer_idx].get());
    state_.input_names_.push_back(input_name_strings_[2 * layer_idx].c_str());

    state_.inputs_.push_back(value_caches_in_[layer_idx].get());
    state_.input_names_.push_back(input_name_strings_[2 * layer_idx + 1].c_str());

    state_.outputs_.push_back(key_caches_out_[layer_idx].get());
    state_.output_names_.push_back(output_name_strings_[2 * layer_idx].c_str());

    state_.outputs_.push_back(value_caches_out_[layer_idx].get());
    state_.output_names_.push_back(output_name_strings_[2 * layer_idx + 1].c_str());
  }
}

void WindowedKeyValueCache::SlideLayer(size_t layer_idx) {
  const auto& layer_state = per_layer_states_[layer_idx];

  const auto window_size = layer_state.window_size;
  const auto& key_cache_shape_in = layer_state.key_cache_shape_in;
  const auto& key_cache_shape_out = layer_state.key_cache_shape_out;
  const auto& value_cache_shape_in = layer_state.value_cache_shape_in;
  const auto& value_cache_shape_out = layer_state.value_cache_shape_out;

  uint8_t* key_cache_in_data = key_caches_in_[layer_idx]->GetTensorMutableData<uint8_t>();
  uint8_t* key_cache_out_data = key_caches_out_[layer_idx]->GetTensorMutableData<uint8_t>();

  int64_t num_key_cache_chunks = key_cache_shape_in[0] * key_cache_shape_in[2];
  for (int64_t j = 0; j < num_key_cache_chunks; ++j) {
    {
      cpu_span<uint8_t> key_cache_dst(key_cache_in_data + j * key_cache_shape_in[3],
                                      key_cache_shape_in[3] - window_size);
      cpu_span<uint8_t> key_cache_src(key_cache_in_data + j * key_cache_shape_in[3] + window_size,
                                      key_cache_shape_in[3] - window_size);
      std::copy(key_cache_src.begin(), key_cache_src.end(), key_cache_dst.begin());
    }
    {
      cpu_span<uint8_t> key_cache_dst(key_cache_in_data + j * key_cache_shape_in[3] + key_cache_shape_in[3] - window_size,
                                      window_size);
      cpu_span<uint8_t> key_cache_src(key_cache_out_data + j * key_cache_shape_out[3],
                                      window_size);
      std::copy(key_cache_src.begin(), key_cache_src.end(), key_cache_dst.begin());
    }
  }

  uint8_t* value_cache_in_data = value_caches_in_[layer_idx]->GetTensorMutableData<uint8_t>();
  uint8_t* value_cache_out_data = value_caches_out_[layer_idx]->GetTensorMutableData<uint8_t>();

  for (int64_t j = 0; j < value_cache_shape_in[0]; ++j) {
    {
      cpu_span<uint8_t> value_cache_dst(value_cache_in_data + (j * value_cache_shape_in[2] * value_cache_shape_in[3]),
                                        (value_cache_shape_in[2] - window_size) * value_cache_shape_in[3]);
      cpu_span<uint8_t> value_cache_src(value_cache_in_data + (j * value_cache_shape_in[2] * value_cache_shape_in[3]) +
                                            (window_size * value_cache_shape_in[3]),
                                        (value_cache_shape_in[2] - window_size) * value_cache_shape_in[3]);
      std::copy(value_cache_src.begin(), value_cache_src.end(), value_cache_dst.begin());
    }
    {
      cpu_span<uint8_t> value_cache_dst(value_cache_in_data + (j * value_cache_shape_in[2] * value_cache_shape_in[3]) +
                                            ((value_cache_shape_in[2] - window_size) * value_cache_shape_in[3]),
                                        window_size * value_cache_shape_in[3]);
      cpu_span<uint8_t> value_cache_src(value_cache_out_data + (j * value_cache_shape_out[2] * value_cache_shape_out[3]),
                                        window_size * value_cache_shape_out[3]);
      std::copy(value_cache_src.begin(), value_cache_src.end(), value_cache_dst.begin());
    }
  }
}

void WindowedKeyValueCache::TransitionLayerToTokenGeneration(size_t layer_idx) {
  // Transition from prompt processing to token generation.
  // Concatenate the last window_size elements to the end of the cache

  // key_caches_in = Concat(key_caches_in[:, :, :, 1:], key_caches_out)
  // [num_key_value_heads, 1, head_size, context_length-1] = [num_key_value_heads, 1, head_size, context_length - window_size_ - 1] +
  //                                                         [num_key_value_heads, 1, head_size, window_size]
  // value_cache = Concat(value_caches_in[:, :, 1:, :], value_caches_out)
  // [num_key_value_heads, 1, context_length - 1, head_size] = [num_key_value_heads, 1, context_length - window_size - 1, head_size] +
  //                                                           [num_key_value_heads, 1, window_size_, head_size]

  auto& layer_state = per_layer_states_[layer_idx];

  const auto window_size = layer_state.window_size;
  const auto& key_cache_shape_in = layer_state.key_cache_shape_in;
  const auto& key_cache_shape_out = layer_state.key_cache_shape_out;
  const auto& value_cache_shape_in = layer_state.value_cache_shape_in;
  const auto& value_cache_shape_out = layer_state.value_cache_shape_out;

  constexpr int updated_window_size = 1;

  const auto num_key_value_heads = model_.config_->model.decoder.num_key_value_heads;
  const auto head_size = model_.config_->model.decoder.head_size;
  const auto context_length = model_.config_->model.context_length;

  const auto updated_key_cache_shape_in = std::array<int64_t, 4>{num_key_value_heads, 1,
                                                                 head_size,
                                                                 context_length - updated_window_size};

  const auto updated_value_cache_shape_in = std::array<int64_t, 4>{num_key_value_heads, 1,
                                                                   context_length - updated_window_size,
                                                                   head_size};

  const auto updated_key_cache_shape_out = std::array<int64_t, 4>{num_key_value_heads, 1,
                                                                  head_size,
                                                                  updated_window_size};

  const auto updated_value_cache_shape_out = std::array<int64_t, 4>{num_key_value_heads, 1,
                                                                    updated_window_size,
                                                                    head_size};

  std::unique_ptr<OrtValue> key_cache = OrtValue::CreateTensor(Allocator(), updated_key_cache_shape_in, type_);

  uint8_t* key_cache_data = key_cache->GetTensorMutableData<uint8_t>();
  uint8_t* key_cache_in_data = key_caches_in_[layer_idx]->GetTensorMutableData<uint8_t>();
  uint8_t* key_cache_out_data = key_caches_out_[layer_idx]->GetTensorMutableData<uint8_t>();

  int64_t num_key_cache_chunks = updated_key_cache_shape_in[0] * updated_key_cache_shape_in[2];
  for (int64_t j = 0; j < num_key_cache_chunks; ++j) {
    {
      cpu_span<uint8_t> key_cache_dst(key_cache_data + j * updated_key_cache_shape_in[3],
                                      updated_key_cache_shape_in[3] - updated_window_size);
      cpu_span<uint8_t> key_cache_src(key_cache_in_data + j * key_cache_shape_in[3] + updated_window_size,
                                      key_cache_shape_in[3] - updated_window_size);
      std::copy(key_cache_src.begin(), key_cache_src.end(), key_cache_dst.begin());
    }
    {
      cpu_span<uint8_t> key_cache_dst(key_cache_data + j * updated_key_cache_shape_in[3] +
                                          key_cache_shape_in[3] - updated_window_size,
                                      window_size);
      cpu_span<uint8_t> key_cache_src(key_cache_out_data + j * key_cache_shape_out[3],
                                      window_size);
      std::copy(key_cache_src.begin(), key_cache_src.end(), key_cache_dst.begin());
    }
  }

  key_caches_in_[layer_idx] = std::move(key_cache);
  key_caches_out_[layer_idx] = OrtValue::CreateTensor(Allocator(), updated_key_cache_shape_out, type_);

  std::unique_ptr<OrtValue> value_cache = OrtValue::CreateTensor(Allocator(), updated_value_cache_shape_in, type_);

  uint8_t* value_cache_data = value_cache->GetTensorMutableData<uint8_t>();
  uint8_t* value_cache_in_data = value_caches_in_[layer_idx]->GetTensorMutableData<uint8_t>();
  uint8_t* value_cache_out_data = value_caches_out_[layer_idx]->GetTensorMutableData<uint8_t>();

  for (int64_t j = 0; j < updated_value_cache_shape_in[0]; ++j) {
    {
      cpu_span<uint8_t> value_cache_dst(value_cache_data + (j * updated_value_cache_shape_in[2] * updated_value_cache_shape_in[3]),
                                        (value_cache_shape_in[2] - updated_window_size) * updated_value_cache_shape_in[3]);
      cpu_span<uint8_t> value_cache_src(value_cache_in_data + (j * value_cache_shape_in[2] * value_cache_shape_in[3]) +
                                            (updated_window_size * value_cache_shape_in[3]),
                                        (value_cache_shape_in[2] - updated_window_size) * value_cache_shape_in[3]);
      std::copy(value_cache_src.begin(), value_cache_src.end(), value_cache_dst.begin());
    }
    {
      cpu_span<uint8_t> value_cache_dst(value_cache_data + (j * updated_value_cache_shape_in[2] * updated_value_cache_shape_in[3]) +
                                            ((value_cache_shape_in[2] - updated_window_size) * updated_value_cache_shape_in[3]),
                                        value_cache_shape_out[2] * value_cache_shape_out[3]);
      cpu_span<uint8_t> value_cache_src(value_cache_out_data + (j * value_cache_shape_out[2] * value_cache_shape_out[3]),
                                        value_cache_shape_out[2] * value_cache_shape_out[3]);
      std::copy(value_cache_src.begin(), value_cache_src.end(), value_cache_dst.begin());
    }
  }

  value_caches_in_[layer_idx] = std::move(value_cache);
  value_caches_out_[layer_idx] = OrtValue::CreateTensor(Allocator(), updated_value_cache_shape_out, type_);

  // update values in per-layer state
  layer_state.window_size = updated_window_size;
  layer_state.key_cache_shape_in = updated_key_cache_shape_in;
  layer_state.value_cache_shape_in = updated_value_cache_shape_in;
  layer_state.key_cache_shape_out = updated_key_cache_shape_out;
  layer_state.value_cache_shape_out = updated_value_cache_shape_out;

  state_.inputs_[input_index_ + 2 * layer_idx] = key_caches_in_[layer_idx].get();
  state_.inputs_[input_index_ + 2 * layer_idx + 1] = value_caches_in_[layer_idx].get();
  state_.outputs_[output_index_ + 2 * layer_idx] = key_caches_out_[layer_idx].get();
  state_.outputs_[output_index_ + 2 * layer_idx + 1] = value_caches_out_[layer_idx].get();
}

void WindowedKeyValueCache::UpdateLayer(DeviceSpan<int32_t> /*beam_indices*/, int total_length, size_t layer_idx) {
  assert(layer_idx < layer_count_);

  auto& layer_state = per_layer_states_[layer_idx];

  if (layer_state.is_first_update) {
    layer_state.num_windows = (total_length + layer_state.window_size - 1) / layer_state.window_size;
    layer_state.is_first_update = false;
    ++layer_state.window_index;
    return;
  }

  if (layer_state.window_size == 1 || layer_state.window_index < layer_state.num_windows) {
    SlideLayer(layer_idx);
    ++layer_state.window_index;
    return;
  }

  TransitionLayerToTokenGeneration(layer_idx);
}

void WindowedKeyValueCache::Update(DeviceSpan<int32_t> beam_indices, int current_length) {
  PartialUpdate(beam_indices, current_length, all_layer_indices_);
}

void WindowedKeyValueCache::PartialUpdate(DeviceSpan<int32_t> beam_indices, int total_length,
                                          std::span<const size_t> layer_indices) {
  ThreadPool thread_pool{layer_indices.size()};
  thread_pool.Compute([&](size_t i) {
    UpdateLayer(beam_indices, total_length, layer_indices[i]);
  });
}

}  // namespace Generators
