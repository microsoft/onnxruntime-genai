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
      kv_cache_shape_in_{1, model_.config_->model.decoder.num_key_value_heads,
                          model_.config_->model.context_length - window_size_, model_.config_->model.decoder.head_size},
      kv_cache_shape_out_{1, model_.config_->model.decoder.num_key_value_heads,
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

  if (type_ == Ort::TypeToTensorType<uint8_t>) {
    InitializeCaches<uint8_t>();
  } else if (type_ == Ort::TypeToTensorType<uint16_t>) {
    InitializeCaches<uint16_t>();
  } else if (type_ == Ort::TypeToTensorType<float>) {
    InitializeCaches<float>();
  } else {
    throw std::runtime_error("Unsupported data type in WindowedKeyValueCache.");
  }

}

template <typename T>
void WindowedKeyValueCache::InitializeCaches() {
  for (int i = 0; i < layer_count_; ++i) {
    key_caches_in_.push_back(
        OrtValue::CreateTensor(Allocator(), kv_cache_shape_in_, type_));
    std::fill_n(key_caches_in_[i]->GetTensorMutableData<T>(),
                ElementCountFromShape(kv_cache_shape_in_),
                static_cast<T>(model_.config_->model.decoder.sliding_window->pad_value));

    value_caches_in_.push_back(
        OrtValue::CreateTensor(Allocator(), kv_cache_shape_in_, type_));
    std::fill_n(value_caches_in_[i]->GetTensorMutableData<T>(),
                ElementCountFromShape(kv_cache_shape_in_),
                static_cast<T>(model_.config_->model.decoder.sliding_window->pad_value));

    key_caches_out_.push_back(
        OrtValue::CreateTensor(Allocator(), kv_cache_shape_out_, type_));
    value_caches_out_.push_back(
        OrtValue::CreateTensor(Allocator(), kv_cache_shape_out_, type_));
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

template <typename T>
void WindowedKeyValueCache::SlideLayer(size_t layer_idx) {
  assert(layer_idx < layer_count_);

  T* key_cache_in_data = key_caches_in_[layer_idx]->GetTensorMutableData<T>();
  T* key_cache_out_data = key_caches_out_[layer_idx]->GetTensorMutableData<T>();
  T* value_cache_in_data = value_caches_in_[layer_idx]->GetTensorMutableData<T>();
  T* value_cache_out_data = value_caches_out_[layer_idx]->GetTensorMutableData<T>();

  for (int64_t j = 0; j < kv_cache_shape_in_[1]; ++j) {
    int64_t offset =  j * kv_cache_shape_in_[2] * kv_cache_shape_in_[3];
    int64_t num_carry_over = (kv_cache_shape_in_[2] - window_size_) * kv_cache_shape_in_[3];
    int64_t num_left_behind = window_size_ * kv_cache_shape_in_[3];
    {
      // slide the last (context_length - window_size_) tokens to the left
      cpu_span<T> key_cache_dst(key_cache_in_data + offset, num_carry_over);
      cpu_span<T> key_cache_src(key_cache_in_data + offset + num_left_behind, num_carry_over);
      std::copy(key_cache_src.begin(), key_cache_src.end(), key_cache_dst.begin());

      cpu_span<T> value_cache_dst(value_cache_in_data + offset, num_carry_over);
      cpu_span<T> value_cache_src(value_cache_in_data + offset + num_left_behind, num_carry_over);
      std::copy(value_cache_src.begin(), value_cache_src.end(), value_cache_dst.begin());
    }
    {
      int64_t output_offset = j * kv_cache_shape_out_[2] * kv_cache_shape_out_[3];
      // copy the new window_size_ tokens from the output cache to the input cache
      cpu_span<T> key_cache_dst(key_cache_in_data + offset + num_carry_over, num_left_behind);
      cpu_span<T> key_cache_src(key_cache_out_data + output_offset, num_left_behind);
      std::copy(key_cache_src.begin(), key_cache_src.end(), key_cache_dst.begin());

      cpu_span<T> value_cache_dst(value_cache_in_data + offset + num_carry_over, num_left_behind);
      cpu_span<T> value_cache_src(value_cache_out_data + output_offset, num_left_behind);
      std::copy(value_cache_src.begin(), value_cache_src.end(), value_cache_dst.begin());
    }
  }
}

void WindowedKeyValueCache::SlideAllLayers() {
  ThreadPool thread_pool{static_cast<size_t>(layer_count_)};
  thread_pool.Compute([this](size_t layer_idx) {
    if (type_ == Ort::TypeToTensorType<uint8_t>) {
      SlideLayer<uint8_t>(layer_idx);
    } else if (type_ == Ort::TypeToTensorType<uint16_t>) {
      SlideLayer<uint16_t>(layer_idx);
    } else if (type_ == Ort::TypeToTensorType<float>) {
      SlideLayer<float>(layer_idx);
    } else {
      throw std::runtime_error("Unsupported data type in WindowedKeyValueCache.");
    }
  });
}

void WindowedKeyValueCache::SlideLayers(std::span<const size_t> layer_indices) {
  ThreadPool thread_pool{layer_indices.size()};
  thread_pool.Compute([&](size_t idx) {
    const size_t layer_idx = layer_indices[idx];
    if (type_ == Ort::TypeToTensorType<uint8_t>) {
      SlideLayer<uint8_t>(layer_idx);
    } else if (type_ == Ort::TypeToTensorType<uint16_t>) {
      SlideLayer<uint16_t>(layer_idx);
    } else if (type_ == Ort::TypeToTensorType<float>) {
      SlideLayer<float>(layer_idx);
    } else {
      throw std::runtime_error("Unsupported data type in WindowedKeyValueCache.");
    }
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

  if (type_ == Ort::TypeToTensorType<uint8_t>) {
    TransitionToTokenGeneration<uint8_t>();
  } else if (type_ == Ort::TypeToTensorType<uint16_t>) {
    TransitionToTokenGeneration<uint16_t>();
  } else if (type_ == Ort::TypeToTensorType<float>) {
    TransitionToTokenGeneration<float>();
  } else {
    throw std::runtime_error("Unsupported data type in WindowedKeyValueCache.");
  }
  
}

template <typename T>
void WindowedKeyValueCache::TransitionToTokenGeneration() {
  // Transition from prompt processing to token generation.
  // Concatenate the last window_size_ elements to the end of the cache

  // kv_caches_in_ = Concat(kv_caches_in_[:, :, 1:, :], kv_caches_out_)
  // [1, num_key_value_heads, context_length-1, head_size] = [1, num_key_value_heads, context_length - window_size_ - 1, head_size] +
  //                                                         [1, num_key_value_heads, window_size_, head_size]

  int updated_window_size = 1;
  auto updated_kv_cache_shape_in = std::array<int64_t, 4>{1, model_.config_->model.decoder.num_key_value_heads,
                                                           model_.config_->model.context_length - updated_window_size,
                                                           model_.config_->model.decoder.head_size};

  auto updated_kv_cache_shape_out = std::array<int64_t, 4>{1, model_.config_->model.decoder.num_key_value_heads,
                                                            updated_window_size,
                                                            model_.config_->model.decoder.head_size};

  ThreadPool thread_pool{static_cast<size_t>(layer_count_)};
  thread_pool.Compute([&](size_t layer_idx) {
    std::unique_ptr<OrtValue> key_cache = OrtValue::CreateTensor(Allocator(), updated_kv_cache_shape_in, type_);
    std::unique_ptr<OrtValue> value_cache = OrtValue::CreateTensor(Allocator(), updated_kv_cache_shape_in, type_);

    T* key_cache_data = key_cache->GetTensorMutableData<T>();
    T* key_cache_in_data = key_caches_in_[layer_idx]->GetTensorMutableData<T>();
    T* key_cache_out_data = key_caches_out_[layer_idx]->GetTensorMutableData<T>();
    T* value_cache_data = value_cache->GetTensorMutableData<T>();
    T* value_cache_in_data = value_caches_in_[layer_idx]->GetTensorMutableData<T>();
    T* value_cache_out_data = value_caches_out_[layer_idx]->GetTensorMutableData<T>();

    for (int64_t j = 0; j < updated_kv_cache_shape_in[1]; ++j) {
      int64_t offset =  j * kv_cache_shape_in_[2] * kv_cache_shape_in_[3];
      int64_t updated_offset = j * updated_kv_cache_shape_in[2] * updated_kv_cache_shape_in[3];
      int64_t num_carry_over = (kv_cache_shape_in_[2] - updated_window_size) * kv_cache_shape_in_[3];
      {
        int64_t num_left_behind = updated_window_size * kv_cache_shape_in_[3];
        // copy over the last (context_length - window_size_ - 1) tokens to the new cache
        cpu_span<T> key_cache_dst(key_cache_data + updated_offset, num_carry_over);
        cpu_span<T> key_cache_src(key_cache_in_data + offset + num_left_behind, num_carry_over);
        std::copy(key_cache_src.begin(), key_cache_src.end(), key_cache_dst.begin());
  
        cpu_span<T> value_cache_dst(value_cache_data + updated_offset, num_carry_over);
        cpu_span<T> value_cache_src(value_cache_in_data + offset + num_left_behind, num_carry_over);
        std::copy(value_cache_src.begin(), value_cache_src.end(), value_cache_dst.begin());
      }
      {
        int64_t output_offset = j * kv_cache_shape_out_[2] * kv_cache_shape_out_[3];
        int64_t num_copy = window_size_ * kv_cache_shape_out_[3];
        // copy the new window_size_ tokens from the output cache to the new cache
        cpu_span<T> key_cache_dst(key_cache_data + updated_offset + num_carry_over, num_copy);
        cpu_span<T> key_cache_src(key_cache_out_data + output_offset, num_copy);
        std::copy(key_cache_src.begin(), key_cache_src.end(), key_cache_dst.begin());
  
        cpu_span<T> value_cache_dst(value_cache_data + updated_offset + num_carry_over, num_copy);
        cpu_span<T> value_cache_src(value_cache_out_data + output_offset, num_copy);
        std::copy(value_cache_src.begin(), value_cache_src.end(), value_cache_dst.begin());
      }
    }

    key_caches_in_[layer_idx] = std::move(key_cache);
    key_caches_out_[layer_idx] = OrtValue::CreateTensor(Allocator(), updated_kv_cache_shape_out, type_);

    value_caches_in_[layer_idx] = std::move(value_cache);
    value_caches_out_[layer_idx] = OrtValue::CreateTensor(Allocator(), updated_kv_cache_shape_out, type_);
  });

  window_size_ = 1;
  kv_cache_shape_in_ = updated_kv_cache_shape_in;
  kv_cache_shape_out_ = updated_kv_cache_shape_out;

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
