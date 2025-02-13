// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string_view>  // include string_view first so that ortx_tokenizer.h include will compile
                        // https://github.com/microsoft/onnxruntime-extensions/issues/879

#include "windowed_kv_cache.h"

#include "../generators.h"
#include "../logging.h"
#include "../make_string.h"
#include "model.h"
#include "threadpool.h"
#include "utils.h"

namespace Generators {

WindowedKeyValueCache::WindowedKeyValueCache(State& state)
    : state_{state},
      layer_count_{model_.config_->model.decoder.num_hidden_layers},
      window_size_{model_.config_->model.decoder.sliding_window->window_size},
      key_cache_shape_in_{model_.config_->model.decoder.num_key_value_heads, 1,
                          model_.config_->model.decoder.head_size, model_.config_->model.context_length - window_size_},
      key_cache_shape_out_{model_.config_->model.decoder.num_key_value_heads, 1,
                           model_.config_->model.decoder.head_size, window_size_},
      value_cache_shape_in_{model_.config_->model.decoder.num_key_value_heads, 1,
                            model_.config_->model.context_length - window_size_, model_.config_->model.decoder.head_size},
      value_cache_shape_out_{model_.config_->model.decoder.num_key_value_heads, 1,
                             window_size_, model_.config_->model.decoder.head_size} {
  if (layer_count_ == 0) {
    throw std::runtime_error("Expected there to be at least 1 layer in the model. Actual: " +
                             std::to_string(layer_count_) + ". Please check the num_hidden_layers attribute in the model configuration.");
  }
  for (int i = 0; i < layer_count_; ++i) {
    input_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.inputs.past_key_names, i));
    input_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.inputs.past_value_names, i));

    output_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.present_key_names, i));
    output_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.present_value_names, i));
  }

  type_ = model_.session_info_->GetInputDataType(input_name_strings_[0]);
  if (type_ != Ort::TypeToTensorType<uint8_t>) {
    throw std::runtime_error("Expected input data type to be uint8_t for WindowedKeyValueCache. Actual: " +
                             std::to_string(type_));
  }

  for (int i = 0; i < layer_count_; ++i) {
    key_caches_in_.push_back(
        OrtValue::CreateTensor(Allocator(), key_cache_shape_in_, type_));
    std::fill_n(key_caches_in_[i]->GetTensorMutableData<uint8_t>(),
                ElementCountFromShape(key_cache_shape_in_),
                static_cast<uint8_t>(model_.config_->model.decoder.sliding_window->pad_value));

    value_caches_in_.push_back(
        OrtValue::CreateTensor(Allocator(), value_cache_shape_in_, type_));
    std::fill_n(value_caches_in_[i]->GetTensorMutableData<uint8_t>(),
                ElementCountFromShape(value_cache_shape_in_),
                static_cast<uint8_t>(model_.config_->model.decoder.sliding_window->pad_value));

    key_caches_out_.push_back(
        OrtValue::CreateTensor(Allocator(), key_cache_shape_out_, type_));
    value_caches_out_.push_back(
        OrtValue::CreateTensor(Allocator(), value_cache_shape_out_, type_));
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
  assert(layer_idx < layer_count_);

  uint8_t* key_cache_in_data = key_caches_in_[layer_idx]->GetTensorMutableData<uint8_t>();
  uint8_t* key_cache_out_data = key_caches_out_[layer_idx]->GetTensorMutableData<uint8_t>();

  int64_t num_key_cache_chunks = key_cache_shape_in_[0] * key_cache_shape_in_[2];
  for (int64_t j = 0; j < num_key_cache_chunks; ++j) {
    {
      cpu_span<uint8_t> key_cache_dst(key_cache_in_data + j * key_cache_shape_in_[3],
                                      key_cache_shape_in_[3] - window_size_);
      cpu_span<uint8_t> key_cache_src(key_cache_in_data + j * key_cache_shape_in_[3] + window_size_,
                                      key_cache_shape_in_[3] - window_size_);
      std::copy(key_cache_src.begin(), key_cache_src.end(), key_cache_dst.begin());
    }
    {
      cpu_span<uint8_t> key_cache_dst(key_cache_in_data + j * key_cache_shape_in_[3] + key_cache_shape_in_[3] - window_size_,
                                      window_size_);
      cpu_span<uint8_t> key_cache_src(key_cache_out_data + j * key_cache_shape_out_[3],
                                      window_size_);
      std::copy(key_cache_src.begin(), key_cache_src.end(), key_cache_dst.begin());
    }
  }

  uint8_t* value_cache_in_data = value_caches_in_[layer_idx]->GetTensorMutableData<uint8_t>();
  uint8_t* value_cache_out_data = value_caches_out_[layer_idx]->GetTensorMutableData<uint8_t>();

  for (int64_t j = 0; j < value_cache_shape_in_[0]; ++j) {
    {
      cpu_span<uint8_t> value_cache_dst(value_cache_in_data + (j * value_cache_shape_in_[2] * value_cache_shape_in_[3]),
                                        (value_cache_shape_in_[2] - window_size_) * value_cache_shape_in_[3]);
      cpu_span<uint8_t> value_cache_src(value_cache_in_data + (j * value_cache_shape_in_[2] * value_cache_shape_in_[3]) +
                                            (window_size_ * value_cache_shape_in_[3]),
                                        (value_cache_shape_in_[2] - window_size_) * value_cache_shape_in_[3]);
      std::copy(value_cache_src.begin(), value_cache_src.end(), value_cache_dst.begin());
    }
    {
      cpu_span<uint8_t> value_cache_dst(value_cache_in_data + (j * value_cache_shape_in_[2] * value_cache_shape_in_[3]) +
                                            ((value_cache_shape_in_[2] - window_size_) * value_cache_shape_in_[3]),
                                        window_size_ * value_cache_shape_in_[3]);
      cpu_span<uint8_t> value_cache_src(value_cache_out_data + (j * value_cache_shape_out_[2] * value_cache_shape_out_[3]),
                                        window_size_ * value_cache_shape_out_[3]);
      std::copy(value_cache_src.begin(), value_cache_src.end(), value_cache_dst.begin());
    }
  }
}

void WindowedKeyValueCache::SlideAllLayers() {
  ThreadPool thread_pool{static_cast<size_t>(layer_count_)};
  thread_pool.Compute([this](size_t layer_idx) {
    SlideLayer(layer_idx);
  });
}

void WindowedKeyValueCache::SlideLayers(std::span<const size_t> layer_indices) {
  ThreadPool thread_pool{layer_indices.size()};
  thread_pool.Compute([&](size_t idx) {
    const size_t layer_idx = layer_indices[idx];
    SlideLayer(layer_idx);
  });
}

void WindowedKeyValueCache::Update(DeviceSpan<int32_t> /* beam_indices */, int current_length) {
  if (is_first_update_) {
    num_windows_ = (current_length + window_size_ - 1) / window_size_;
    is_first_update_ = false;
    window_index_++;
    return;
  } else if (window_size_ == 1 || window_index_ < num_windows_) {
    SlideAllLayers();
    window_index_++;
    return;
  }

  // Transition from prompt processing to token generation.
  // Concatenate the last window_size_ elements to the end of the cache

  // key_caches_in_ = Concat(key_caches_in_[:, :, :, 1:], key_caches_out_)
  // [num_key_value_heads, 1, head_size, context_length-1] = [num_key_value_heads, 1, head_size, context_length - window_size_ - 1] +
  //                                                         [num_key_value_heads, 1, head_size, window_size_]
  // value_cache = Concat(value_caches_in_[:, :, 1:, :], value_caches_out_)
  // [num_key_value_heads, 1, context_length - 1, head_size] = [num_key_value_heads, 1, context_length - window_size_ - 1, head_size] +
  //                                                           [num_key_value_heads, 1, window_size_, head_size]

  int updated_window_size = 1;
  auto updated_key_cache_shape_in = std::array<int64_t, 4>{model_.config_->model.decoder.num_key_value_heads, 1,
                                                           model_.config_->model.decoder.head_size,
                                                           model_.config_->model.context_length - updated_window_size};

  auto updated_value_cache_shape_in = std::array<int64_t, 4>{model_.config_->model.decoder.num_key_value_heads, 1,
                                                             model_.config_->model.context_length - updated_window_size,
                                                             model_.config_->model.decoder.head_size};

  auto updated_key_cache_shape_out = std::array<int64_t, 4>{model_.config_->model.decoder.num_key_value_heads, 1,
                                                            model_.config_->model.decoder.head_size,
                                                            updated_window_size};

  auto updated_value_cache_shape_out = std::array<int64_t, 4>{model_.config_->model.decoder.num_key_value_heads, 1,
                                                              updated_window_size,
                                                              model_.config_->model.decoder.head_size};

  ThreadPool thread_pool{static_cast<size_t>(layer_count_)};
  thread_pool.Compute([&](size_t layer_idx) {
    std::unique_ptr<OrtValue> key_cache = OrtValue::CreateTensor(Allocator(), updated_key_cache_shape_in, type_);

    uint8_t* key_cache_data = key_cache->GetTensorMutableData<uint8_t>();
    uint8_t* key_cache_in_data = key_caches_in_[layer_idx]->GetTensorMutableData<uint8_t>();
    uint8_t* key_cache_out_data = key_caches_out_[layer_idx]->GetTensorMutableData<uint8_t>();

    int64_t num_key_cache_chunks = updated_key_cache_shape_in[0] * updated_key_cache_shape_in[2];
    for (int64_t j = 0; j < num_key_cache_chunks; ++j) {
      {
        cpu_span<uint8_t> key_cache_dst(key_cache_data + j * updated_key_cache_shape_in[3],
                                        updated_key_cache_shape_in[3] - updated_window_size);
        cpu_span<uint8_t> key_cache_src(key_cache_in_data + j * key_cache_shape_in_[3] + updated_window_size,
                                        key_cache_shape_in_[3] - updated_window_size);
        std::copy(key_cache_src.begin(), key_cache_src.end(), key_cache_dst.begin());
      }
      {
        cpu_span<uint8_t> key_cache_dst(key_cache_data + j * updated_key_cache_shape_in[3] +
                                            key_cache_shape_in_[3] - updated_window_size,
                                        window_size_);
        cpu_span<uint8_t> key_cache_src(key_cache_out_data + j * key_cache_shape_out_[3],
                                        window_size_);
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
                                          (value_cache_shape_in_[2] - updated_window_size) * updated_value_cache_shape_in[3]);
        cpu_span<uint8_t> value_cache_src(value_cache_in_data + (j * value_cache_shape_in_[2] * value_cache_shape_in_[3]) +
                                              (updated_window_size * value_cache_shape_in_[3]),
                                          (value_cache_shape_in_[2] - updated_window_size) * value_cache_shape_in_[3]);
        std::copy(value_cache_src.begin(), value_cache_src.end(), value_cache_dst.begin());
      }
      {
        cpu_span<uint8_t> value_cache_dst(value_cache_data + (j * updated_value_cache_shape_in[2] * updated_value_cache_shape_in[3]) +
                                              ((value_cache_shape_in_[2] - updated_window_size) * updated_value_cache_shape_in[3]),
                                          value_cache_shape_out_[2] * value_cache_shape_out_[3]);
        cpu_span<uint8_t> value_cache_src(value_cache_out_data + (j * value_cache_shape_out_[2] * value_cache_shape_out_[3]),
                                          value_cache_shape_out_[2] * value_cache_shape_out_[3]);
        std::copy(value_cache_src.begin(), value_cache_src.end(), value_cache_dst.begin());
      }
    }

    value_caches_in_[layer_idx] = std::move(value_cache);
    value_caches_out_[layer_idx] = OrtValue::CreateTensor(Allocator(), updated_value_cache_shape_out, type_);
  });

  window_size_ = 1;
  key_cache_shape_in_ = updated_key_cache_shape_in;
  value_cache_shape_in_ = updated_value_cache_shape_in;
  key_cache_shape_out_ = updated_key_cache_shape_out;
  value_cache_shape_out_ = updated_value_cache_shape_out;

  for (size_t layer_idx = 0; layer_idx < layer_count_; ++layer_idx) {
    state_.inputs_[input_index_ + 2 * layer_idx] = key_caches_in_[layer_idx].get();
    state_.inputs_[input_index_ + 2 * layer_idx + 1] = value_caches_in_[layer_idx].get();
    state_.outputs_[output_index_ + 2 * layer_idx] = key_caches_out_[layer_idx].get();
    state_.outputs_[output_index_ + 2 * layer_idx + 1] = value_caches_out_[layer_idx].get();
  }
}

void WindowedKeyValueCache::PartialTokenGenerationUpdate(DeviceSpan<int32_t> /* beam_indices */, int /* total_length */,
                                                         std::span<const size_t> layer_indices_to_update) {
  assert(window_size_ == 1);
  SlideLayers(layer_indices_to_update);
}

}  // namespace Generators
