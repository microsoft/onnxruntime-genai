// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "conv_kv_cache.h"

#include "static_kv_cache.h"
#include "../../logging.h"
#include "../model.h"

#include <algorithm>

namespace Generators {

bool HasConvKeyValueCache(const Model& model) {
  return ModelType::IsLFM2(model.config_->model.type) || !model.config_->model.decoder.layer_types.empty();
}

ConvKeyValueCache::ConvKeyValueCache(State& state)
    : state_{state},
      layer_types_{model_.config_->model.decoder.layer_types},
      layer_count_{model_.config_->model.decoder.num_hidden_layers} {
  // Validate layer_types array size matches num_hidden_layers before accessing elements.
  if (layer_count_ < 0) {
    throw std::runtime_error("ConvKeyValueCache: num_hidden_layers must be non-negative. Actual: " + std::to_string(layer_count_));
  }

  const size_t expected_layer_types_size = static_cast<size_t>(layer_count_);
  if (layer_types_.size() != expected_layer_types_size) {
    throw std::runtime_error("ConvKeyValueCache: layer_types array size (" + std::to_string(layer_types_.size()) +
                             ") does not match num_hidden_layers (" + std::to_string(layer_count_) + ")");
  }

  // Classify layers into attention (KV) and conv types
  for (int i = 0; i < layer_count_; ++i) {
    if (layer_types_[i] == "full_attention") {
      kv_layer_indices_.push_back(i);
    } else {
      conv_layer_indices_.push_back(i);
    }
  }
  kv_layer_count_ = static_cast<int>(kv_layer_indices_.size());
  conv_layer_count_ = static_cast<int>(conv_layer_indices_.size());

  // --- KV cache setup (attention layers only) ---
  if (kv_layer_count_ > 0) {
    kv_shape_ = {state_.params_->BatchBeamSize(), model_.config_->model.decoder.num_key_value_heads, 0, model_.config_->model.decoder.head_size};
    kv_pasts_.resize(kv_layer_count_ * 2);
    kv_presents_.reserve(kv_layer_count_ * 2);

    for (int layer_idx : kv_layer_indices_) {
      kv_input_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.inputs.past_key_names, layer_idx));
      kv_input_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.inputs.past_value_names, layer_idx));
      kv_output_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.present_key_names, layer_idx));
      kv_output_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.present_value_names, layer_idx));
    }

    kv_type_ = model_.session_info_.GetInputDataType(kv_input_name_strings_[0]);

    // Shared KV cache: same policy as DefaultKeyValueCache. Start from the genai
    // search option, then let DetectAndConfigureFixedKvShape force share-buffer when
    // the ONNX graph fixes the kv seq_len. The buffer is sized to that fixed dim when
    // present, otherwise to search.max_length for symbolic graphs.
    kv_share_buffer_ = state_.params_->IsPastPresentShareBufferEnabled(model_.config_->model.type);
    const int64_t fixed_kv_seq_len = DetectAndConfigureFixedKvShape(
        model_.session_info_, kv_input_name_strings_, kv_layer_count_,
        state_.params_->search, kv_share_buffer_, "ConvKeyValueCache");
    if (g_log.enabled && g_log.warning && state_.params_->search.past_present_share_buffer && !kv_share_buffer_) {
      Log("warning", "past_present_share_buffer search option set to true, but has been disabled due to the current configuration. See https://aka.ms/generate_config for details");
    } else if (g_log.enabled && !state_.params_->search.past_present_share_buffer && kv_share_buffer_) {
      Log("info", "ConvKeyValueCache: past_present_share_buffer was forced on due to a fixed kv-cache shape in the model graph.");
    }

    if (kv_share_buffer_) {
      const int64_t seq_cap = fixed_kv_seq_len > 0
                                  ? fixed_kv_seq_len
                                  : static_cast<int64_t>(state_.params_->search.max_length);
      if (seq_cap <= 0) {
        throw std::runtime_error(
            "ConvKeyValueCache: past_present_share_buffer requires a fixed kv seq_len in the "
            "model or search.max_length > 0 (set max_length or model.context_length "
            "in genai_config.json).");
      }
      kv_shape_[2] = seq_cap;
      for (int i = 0; i < kv_layer_count_ * 2; ++i) {
        kv_presents_.push_back(OrtValue::CreateTensor(Allocator(), kv_shape_, kv_type_));
        if (Device().ShouldZeroKeyValueCacheTensors()) {
          ByteWrapTensor(Device(), *kv_presents_.back()).Zero();
        }
      }
    } else {
      kv_empty_past_ = OrtValue::CreateTensor(Allocator(), kv_shape_, kv_type_);
      for (int i = 0; i < kv_layer_count_ * 2; ++i) {
        kv_presents_.push_back(OrtValue::CreateTensor(Allocator(), kv_shape_, kv_type_));
      }
    }
  }

  // --- Conv state cache setup (conv layers only) ---
  if (conv_layer_count_ > 0) {
    conv_pasts_.resize(conv_layer_count_);
    conv_presents_.reserve(conv_layer_count_);

    for (int layer_idx : conv_layer_indices_) {
      conv_input_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.inputs.past_conv_names, layer_idx));
      conv_output_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.present_conv_names, layer_idx));
    }

    conv_type_ = model_.session_info_.GetInputDataType(conv_input_name_strings_[0]);

    // Determine conv state shape from the ONNX model input
    auto conv_shape_vec = model_.session_info_.GetInputShape(conv_input_name_strings_[0]);
    if (conv_shape_vec.size() != 3) {
      throw std::runtime_error("ConvKeyValueCache: expected conv state input to be rank 3 [B, H, L], got rank " + std::to_string(conv_shape_vec.size()));
    }
    // Replace batch dimension with actual batch size
    conv_shape_vec[0] = state_.params_->BatchBeamSize();

    for (int i = 0; i < conv_layer_count_; ++i) {
      std::array<int64_t, 3> shape;
      std::copy_n(conv_shape_vec.begin(), 3, shape.begin());
      conv_shapes_.push_back(shape);
      conv_pasts_[i] = OrtValue::CreateTensor(Allocator(), shape, conv_type_);
      // Zero-initialize conv state
      if (Device().ShouldZeroKeyValueCacheTensors()) {
        ByteWrapTensor(Device(), *conv_pasts_[i]).Zero();
      }
      conv_presents_.push_back(OrtValue::CreateTensor(Allocator(), shape, conv_type_));
    }
  }
}

void ConvKeyValueCache::Add() {
  // Add KV cache inputs/outputs. In shared-buffer mode past == present (same fixed
  // buffer), so the input is bound to the present tensor and never changes.
  kv_input_index_ = state_.inputs_.size();
  for (int i = 0; i < kv_layer_count_ * 2; ++i) {
    state_.inputs_.push_back(kv_share_buffer_ ? kv_presents_[i].get() : kv_empty_past_.get());
    state_.input_names_.push_back(kv_input_name_strings_[i].c_str());
  }
  kv_output_index_ = state_.outputs_.size();
  for (int i = 0; i < kv_layer_count_ * 2; ++i) {
    state_.outputs_.push_back(kv_presents_[i].get());
    state_.output_names_.push_back(kv_output_name_strings_[i].c_str());
  }

  // Add conv state inputs/outputs
  conv_input_index_ = state_.inputs_.size();
  for (int i = 0; i < conv_layer_count_; ++i) {
    state_.inputs_.push_back(conv_pasts_[i].get());
    state_.input_names_.push_back(conv_input_name_strings_[i].c_str());
  }
  conv_output_index_ = state_.outputs_.size();
  for (int i = 0; i < conv_layer_count_; ++i) {
    state_.outputs_.push_back(conv_presents_[i].get());
    state_.output_names_.push_back(conv_output_name_strings_[i].c_str());
  }
}

void ConvKeyValueCache::Update(DeviceSpan<int32_t> beam_indices, int total_length) {
  // --- Update KV cache (attention layers) ---
  // In shared-buffer mode the past/present KV buffers are fixed and reused in place
  // (the session writes new tokens into the same buffer), so there is nothing to
  // grow or swap here. Only the dynamic/growing path needs updating.
  if (!kv_share_buffer_) {
    if (!kv_is_first_update_) {
      for (int i = 0; i < kv_layer_count_ * 2; i++) {
        if (beam_indices.empty()) {
          kv_pasts_[i] = std::move(kv_presents_[i]);
        } else {
          PickPastState(beam_indices, i);
        }
        state_.inputs_[kv_input_index_ + i] = kv_pasts_[i].get();
      }
    }

    kv_shape_[2] = total_length;
    for (int i = 0; i < kv_layer_count_ * 2; i++) {
      kv_presents_[i] = OrtValue::CreateTensor(Allocator(), kv_shape_, kv_type_);
      state_.outputs_[kv_output_index_ + i] = kv_presents_[i].get();
    }
    kv_is_first_update_ = false;
  }

  // --- Update conv state cache ---
  if (!conv_is_first_update_) {
    for (int i = 0; i < conv_layer_count_; i++) {
      if (beam_indices.empty()) {
        // Simply swap present -> past (conv state is fixed size)
        conv_pasts_[i] = std::move(conv_presents_[i]);
      } else {
        // Reorder conv state by beam indices
        if (conv_type_ == Ort::TypeToTensorType<float>) {
          PickConvState<float>(beam_indices, i);
        } else {
          PickConvState<Ort::Float16_t>(beam_indices, i);
        }
      }
      state_.inputs_[conv_input_index_ + i] = conv_pasts_[i].get();
    }
  }

  // Allocate new present conv tensors
  for (int i = 0; i < conv_layer_count_; i++) {
    conv_presents_[i] = OrtValue::CreateTensor(Allocator(), conv_shapes_[i], conv_type_);
    state_.outputs_[conv_output_index_ + i] = conv_presents_[i].get();
  }
  conv_is_first_update_ = false;
}

void ConvKeyValueCache::RewindTo(size_t index) {
  // ConvKeyValueCache uses conv layers with fixed-size rolling state buffers that depend on all prior tokens.
  // Rewinding the KV cache without replaying tokens through the conv layers would produce
  // incorrect results, so rewind is not supported for this cache type.
  throw std::runtime_error("ConvKeyValueCache does not support RewindTo.");
}

template <typename ScoreType>
void ConvKeyValueCache::PickPastState(DeviceSpan<int32_t> beam_indices_device, int index) {
  std::span<int32_t> beam_indices = beam_indices_device.CopyDeviceToCpu();
  std::array<int64_t, 4> tensor_shape = kv_shape_;

  auto block_size_per_beam = tensor_shape[1] * tensor_shape[2] * tensor_shape[3];

  OrtValue& present_value = *kv_presents_[index];
  std::unique_ptr<OrtValue> past_value = OrtValue::CreateTensor<ScoreType>(Allocator(), tensor_shape);

  auto past_span = KeyValueCacheDetail::WrapKvCacheTensor<ScoreType>(Device(), *past_value, kv_type_);
  auto present_span = KeyValueCacheDetail::WrapKvCacheTensor<ScoreType>(Device(), present_value, kv_type_);

  for (size_t j = 0; j < beam_indices.size(); j++) {
    int32_t beam_index = beam_indices[j];
    auto present = present_span.subspan(beam_index * block_size_per_beam, block_size_per_beam);
    auto past = past_span.subspan(j * block_size_per_beam, block_size_per_beam);
    past.CopyFrom(present);
  }

  kv_pasts_[index] = std::move(past_value);
}

void ConvKeyValueCache::PickPastState(DeviceSpan<int32_t> beam_indices, int index) {
  if (kv_type_ == Ort::TypeToTensorType<float>) {
    PickPastState<float>(beam_indices, index);
  } else {
    PickPastState<Ort::Float16_t>(beam_indices, index);
  }
}

template <typename T>
void ConvKeyValueCache::PickConvState(DeviceSpan<int32_t> beam_indices_device, int conv_index) {
  std::span<int32_t> beam_indices = beam_indices_device.CopyDeviceToCpu();
  auto& shape = conv_shapes_[conv_index];
  auto block_size_per_beam = shape[1] * shape[2];  // H * L

  OrtValue& present = *conv_presents_[conv_index];
  std::unique_ptr<OrtValue> past = OrtValue::CreateTensor<T>(Allocator(), shape);

  auto past_span = KeyValueCacheDetail::WrapKvCacheTensor<T>(Device(), *past, conv_type_);
  auto present_span = KeyValueCacheDetail::WrapKvCacheTensor<T>(Device(), present, conv_type_);

  for (size_t j = 0; j < beam_indices.size(); j++) {
    int32_t beam_index = beam_indices[j];
    auto present_data = present_span.subspan(beam_index * block_size_per_beam, block_size_per_beam);
    auto past_data = past_span.subspan(j * block_size_per_beam, block_size_per_beam);
    past_data.CopyFrom(present_data);
  }

  conv_pasts_[conv_index] = std::move(past);
}

}  // namespace Generators
