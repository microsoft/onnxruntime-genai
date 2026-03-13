// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "kv_cache.h"
#include "windowed_kv_cache.h"
#include "../openvino/interface.h"
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <mutex>

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
    auto present_span = WrapTensor<T>(Device(), present);
    auto past_span = WrapTensor<T>(Device(), *past);

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
  std::unique_ptr<OrtValue> past = OrtValue::CreateTensor<ScoreType>(Allocator(), shape_);

  auto past_span = WrapTensor<ScoreType>(Device(), *past);
  auto present_span = WrapTensor<ScoreType>(Device(), present);

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
  } else {
    PickPastState<Ort::Float16_t>(beam_indices, index);
  }
}

DefaultKeyValueCache::DefaultKeyValueCache(State& state)
    : state_{state},
      layer_count_{model_.config_->model.decoder.num_hidden_layers},
      past_present_share_buffer_{state_.params_->IsPastPresentShareBufferEnabled(model_.config_->model.type)},
      shape_{state_.params_->BatchBeamSize(), model_.config_->model.decoder.num_key_value_heads, 0, model_.config_->model.decoder.head_size} {
  pasts_.resize(layer_count_ * 2);
  presents_.reserve(layer_count_ * 2);

  for (int i = 0; i < layer_count_; ++i) {
    int layer_idx = kv_layer_indices_.empty() ? i : kv_layer_indices_[i];
    input_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.inputs.past_key_names, layer_idx));
    input_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.inputs.past_value_names, layer_idx));

    output_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.present_key_names, layer_idx));
    output_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.present_value_names, layer_idx));
  }

  if (g_log.enabled && !kv_layer_indices_.empty()) {
    bool is_sequential = true;
    for (int i = 0; i < layer_count_; ++i) {
      if (kv_layer_indices_[i] != i) {
        is_sequential = false;
        break;
      }
    }
    if (!is_sequential) {
      Log("info", "DefaultKeyValueCache: Auto-discovered " + std::to_string(layer_count_) +
                      " KV cache layers at non-sequential indices");
    }
  }

  // Derive the KV data type from the first KV input
  type_ = model_.session_info_.GetInputDataType(input_name_strings_[0]);
  empty_past_ = OrtValue::CreateTensor(Allocator(), shape_, type_);

  const auto maybe_add_auxiliary_state_set = [this](const std::string& input_name_template,
                                                    const std::string& output_name_template,
                                                    const std::vector<int64_t>& shape) {
    if (input_name_template.empty() || output_name_template.empty()) {
      return;
    }

    const std::string first_input_name = ComposeKeyValueName(input_name_template, 0);
    if (!model_.session_info_.HasInput(first_input_name)) {
      return;
    }

    AuxiliaryStateSet state_set;
    state_set.shape = shape;
    state_set.type = model_.session_info_.GetInputDataType(first_input_name);
    state_set.empty_past = OrtValue::CreateTensor(Allocator(), state_set.shape, state_set.type);
    if (Device().GetType() != DeviceType::WEBGPU) {
      ByteWrapTensor(Device(), *state_set.empty_past).Zero();
    }
    state_set.pasts.resize(layer_count_);
    state_set.presents.reserve(layer_count_);

    for (int i = 0; i < layer_count_; ++i) {
      state_set.input_name_strings.emplace_back(ComposeKeyValueName(input_name_template, i));
      state_set.output_name_strings.emplace_back(ComposeKeyValueName(output_name_template, i));
      state_set.presents.push_back(OrtValue::CreateTensor(Allocator(), state_set.shape, state_set.type));
      if (Device().GetType() != DeviceType::WEBGPU) {
        ByteWrapTensor(Device(), *state_set.presents.back()).Zero();
      }
    }

    auxiliary_state_sets_.push_back(std::move(state_set));
  };

  const auto& decoder_config = model_.config_->model.decoder;
  const int64_t linear_conv_dim = static_cast<int64_t>(decoder_config.linear_num_key_heads) * decoder_config.linear_key_head_dim * 2 +
                                  static_cast<int64_t>(decoder_config.linear_num_value_heads) * decoder_config.linear_value_head_dim;
  if (decoder_config.linear_conv_kernel_dim > 0 && linear_conv_dim > 0) {
    maybe_add_auxiliary_state_set(
        decoder_config.inputs.past_conv_state_names,
        decoder_config.outputs.present_conv_state_names,
        {shape_[0], linear_conv_dim, decoder_config.linear_conv_kernel_dim});
  }
  if (decoder_config.linear_num_value_heads > 0 && decoder_config.linear_key_head_dim > 0 && decoder_config.linear_value_head_dim > 0) {
    maybe_add_auxiliary_state_set(
        decoder_config.inputs.past_recurrent_state_names,
        decoder_config.outputs.present_recurrent_state_names,
        {shape_[0], decoder_config.linear_num_value_heads, decoder_config.linear_key_head_dim, decoder_config.linear_value_head_dim});
  }

  if (!auxiliary_state_sets_.empty()) {
    past_present_share_buffer_ = false;
  }

  if (g_log.enabled && g_log.warning && past_present_share_buffer_ != state_.params_->search.past_present_share_buffer)
    Log("warning", "past_present_share_buffer search option set to true, but has been disabled due to the current configuration. See https://aka.ms/generate_config for details");

  if (state_.params_->use_graph_capture && !past_present_share_buffer_) {
    // share buffer is a precondition for graph capture
    throw std::runtime_error("Graph capture is not supported with past_present_share_buffer set to false.");
  }

  // Set the size after empty_past_ has been created with 0 for this field
  if (state_.model_.p_device_->GetType() == DeviceType::NvTensorRtRtx && model_.config_->model.decoder.sliding_window.has_value() &&
      model_.config_->model.decoder.sliding_window->window_size > 0) {
    const int sliding_window_size = model_.config_->model.decoder.sliding_window->window_size;
    const int max_length = state_.params_->search.max_length;

    // Check if we need per-layer allocation for models with alternating attention patterns
    if (!model_.config_->model.decoder.sliding_window->layers.empty()) {
      // Use per-layer allocation based on sliding window layer indices
      layer_shapes_.resize(layer_count_);

      // Initialize all layers with base shape and max_length
      for (int layer_idx = 0; layer_idx < layer_count_; ++layer_idx) {
        layer_shapes_[layer_idx] = shape_;
        layer_shapes_[layer_idx][2] = max_length;
      }

      // Build model-layer-index to cache-slot-index mapping for sparse KV layouts
      std::unordered_map<int, int> model_layer_to_cache_slot;
      for (int slot = 0; slot < layer_count_; ++slot) {
        int model_idx = kv_layer_indices_.empty() ? slot : kv_layer_indices_[slot];
        model_layer_to_cache_slot[model_idx] = slot;
      }

      // Update sliding window layers with constrained cache size
      for (int model_layer_idx : model_.config_->model.decoder.sliding_window->layers) {
        auto it = model_layer_to_cache_slot.find(model_layer_idx);
        if (it != model_layer_to_cache_slot.end()) {
          layer_shapes_[it->second][2] = std::min(max_length, sliding_window_size);
        }
      }
      // Set shape_[2] to max of all layer shapes for RewindTo bounds checking
      shape_[2] = max_length;
    } else {
      // Uniform sliding window allocation (backward compatibility)
      shape_[2] = std::min(max_length, sliding_window_size);
    }
  } else if (past_present_share_buffer_) {
    shape_[2] = state_.params_->search.max_length;
  }

  try {
    // Allocate KV cache tensors - 2 per layer (key and value)
    // For per-layer shapes: alternates between key and value for each layer
    // For uniform shape: all tensors use the same shape
    for (int i = 0; i < layer_count_ * 2; ++i) {
      std::array<int64_t, 4> tensor_shape = shape_;
      if (!layer_shapes_.empty()) {
        // Per-layer allocation: use layer-specific shape
        // i/2 gives us the layer index since we have 2 tensors per layer
        tensor_shape = layer_shapes_[i / 2];
      }

      presents_.push_back(OrtValue::CreateTensor(Allocator(), tensor_shape, type_));
      if (Device().GetType() != DeviceType::WEBGPU) {
        ByteWrapTensor(Device(), *presents_.back()).Zero();
      }
    }
  } catch (const Ort::Exception&) {
    std::ostringstream oss;
    oss << "Could not allocate the key-value cache buffer of shape: ["
        << "batch_size (" << shape_[0] << "), num_key_value_heads ("
        << shape_[1] << "), max_length (" << shape_[2] << "), head_size ("
        << shape_[3] << ")] for " << layer_count_ << " layers. "
        << "Try reducing the max_length requested or reducing the batch size.";
    throw std::runtime_error(oss.str());
  }
}

void DefaultKeyValueCache::Add() {
  input_index_ = state_.inputs_.size();
  output_index_ = state_.outputs_.size();

  for (int i = 0; i < layer_count_ * 2; ++i) {
    state_.inputs_.push_back(empty_past_.get());  // Set empty past here, Update() takes care of the rest
    state_.input_names_.push_back(input_name_strings_[i].c_str());
    state_.outputs_.push_back(presents_[i].get());
    state_.output_names_.push_back(output_name_strings_[i].c_str());
  }

  // For shared_past_present, the past & presents never change, so set the inputs to the present values (outputs are already set above)
  if (past_present_share_buffer_) {
    for (int i = 0; i < layer_count_ * 2; ++i) {
      state_.inputs_[input_index_ + i] = presents_[i].get();
    }
  }

  if (!auxiliary_state_sets_.empty()) {
    auxiliary_input_index_ = state_.inputs_.size();
    auxiliary_output_index_ = state_.outputs_.size();
    for (auto& state_set : auxiliary_state_sets_) {
      for (int i = 0; i < layer_count_; ++i) {
        state_.inputs_.push_back(state_set.empty_past.get());
        state_.input_names_.push_back(state_set.input_name_strings[i].c_str());
        state_.outputs_.push_back(state_set.presents[i].get());
        state_.output_names_.push_back(state_set.output_name_strings[i].c_str());
      }
    }
  }
}

void DefaultKeyValueCache::Update(DeviceSpan<int32_t> beam_indices, int total_length) {
  // If we're sharing past & present buffers there is nothing to do here, so early exit
  if (past_present_share_buffer_)
    return;

  const bool first_update = is_first_update_;

  if (!first_update) {
    for (int i = 0; i < layer_count_ * 2; i++) {
      if (beam_indices.empty()) {
        pasts_[i] = std::move(presents_[i]);
      } else {
        PickPastState(beam_indices, i);
      }
      state_.inputs_[input_index_ + i] = pasts_[i].get();
    }
  }

  if (!layer_shapes_.empty()) {
    // Per-layer allocation with per-layer capacity constraints
    for (int layer_idx = 0; layer_idx < layer_count_; ++layer_idx) {
      std::array<int64_t, 4> current_shape = layer_shapes_[layer_idx];
      const int max_cache_length = static_cast<int>(layer_shapes_[layer_idx][2]);
      current_shape[2] = std::min(total_length, max_cache_length);

      // Key tensor
      presents_[layer_idx * 2] = OrtValue::CreateTensor(Allocator(), current_shape, type_);
      state_.outputs_[output_index_ + layer_idx * 2] = presents_[layer_idx * 2].get();

      // Value tensor
      presents_[layer_idx * 2 + 1] = OrtValue::CreateTensor(Allocator(), current_shape, type_);
      state_.outputs_[output_index_ + layer_idx * 2 + 1] = presents_[layer_idx * 2 + 1].get();
    }
  } else {
    // Uniform allocation
    shape_[2] = total_length;
    for (int i = 0; i < layer_count_ * 2; i++) {
      presents_[i] = OrtValue::CreateTensor(Allocator(), shape_, type_);
      state_.outputs_[output_index_ + i] = presents_[i].get();
    }
  }

  if (!auxiliary_state_sets_.empty()) {
    size_t input_offset = auxiliary_input_index_;
    size_t output_offset = auxiliary_output_index_;
    for (auto& state_set : auxiliary_state_sets_) {
      if (!first_update) {
        for (int i = 0; i < layer_count_; ++i) {
          if (beam_indices.empty()) {
            state_set.pasts[i] = std::move(state_set.presents[i]);
          } else {
            PickPastAuxiliaryState(beam_indices, state_set, i);
          }
          state_.inputs_[input_offset + i] = state_set.pasts[i].get();
        }
      }

      for (int i = 0; i < layer_count_; ++i) {
        state_set.presents[i] = OrtValue::CreateTensor(Allocator(), state_set.shape, state_set.type);
        state_.outputs_[output_offset + i] = state_set.presents[i].get();
      }

      input_offset += layer_count_;
      output_offset += layer_count_;
    }
  }

  is_first_update_ = false;
}

void DefaultKeyValueCache::RewindTo(size_t index) {
  if (past_present_share_buffer_) {
    return;
  } else if (!auxiliary_state_sets_.empty() && index > 0) {
    throw std::runtime_error("RewindTo for models with auxiliary decoder state caches is currently only supported for index 0.");
  } else if (shape_[2] <= static_cast<int>(index)) {
    throw std::runtime_error("Requested length of rewind is greater than the current length.");
  }

  is_first_update_ = true;
  if (index == 0) {
    for (int i = 0; i < layer_count_ * 2; i++) {
      pasts_[i] = nullptr;
      state_.inputs_[input_index_ + i] = empty_past_.get();
    }
    if (!auxiliary_state_sets_.empty()) {
      size_t input_offset = auxiliary_input_index_;
      for (auto& state_set : auxiliary_state_sets_) {
        for (int i = 0; i < layer_count_; ++i) {
          state_set.pasts[i] = nullptr;
          state_.inputs_[input_offset + i] = state_set.empty_past.get();
        }
        input_offset += layer_count_;
      }
    }
  } else if (type_ == Ort::TypeToTensorType<float>) {
    RewindPastTensorsTo<float>(index);
  } else {
    RewindPastTensorsTo<Ort::Float16_t>(index);
  }
}

template <typename T>
void DefaultKeyValueCache::RewindPastTensorsTo(size_t index) {
  assert(index > 0 && !past_present_share_buffer_);

  if (!layer_shapes_.empty()) {
    // Handle per-layer shapes
    // First validate that index doesn't exceed the global max_length
    int max_length = static_cast<int>(shape_[2]);  // Set to max_length in constructor
    if (static_cast<int>(index) > max_length) {
      throw std::runtime_error("Requested rewind length exceeds max_length.");
    }

    for (int i = 0; i < layer_count_ * 2; i++) {
      const int layer_idx = i / 2;
      const std::array<int64_t, 4> layer_shape = layer_shapes_[layer_idx];
      const int layer_max_cache = static_cast<int>(layer_shape[2]);

      // For each layer, rewind to min(index, layer's max capacity)
      // - Full attention layers: min(index, max_length)
      // - Sliding window layers: min(index, sliding_window_size)
      const int actual_rewind_length = std::min(static_cast<int>(index), layer_max_cache);

      std::array<int64_t, 4> new_shape = layer_shape;
      new_shape[2] = actual_rewind_length;
      const auto batch_x_num_heads = new_shape[0] * new_shape[1];
      const auto new_length_x_head_size = new_shape[2] * new_shape[3];

      OrtValue& present = *presents_[i];
      const auto present_shape = present.GetTensorTypeAndShapeInfo()->GetShape();
      const auto old_length_x_head_size = present_shape[2] * new_shape[3];

      std::unique_ptr<OrtValue> past = OrtValue::CreateTensor(Allocator(), new_shape, type_);
      auto past_span = WrapTensor<T>(Device(), *past);
      auto present_span = WrapTensor<T>(Device(), present);

      for (int j = 0; j < batch_x_num_heads; j++) {
        auto present_data = present_span.subspan(j * old_length_x_head_size, new_length_x_head_size);
        auto past_data = past_span.subspan(j * new_length_x_head_size, new_length_x_head_size);
        past_data.CopyFrom(present_data);
      }
      pasts_[i] = std::move(past);
      state_.inputs_[input_index_ + i] = pasts_[i].get();
    }
  } else {
    // Uniform shape handling (existing behavior)
    assert(shape_[2] >= static_cast<int64_t>(index));
    std::array<int64_t, 4> new_shape = shape_;
    new_shape[2] = static_cast<int>(index);
    auto batch_x_num_heads = new_shape[0] * new_shape[1];
    auto new_length_x_head_size = new_shape[2] * new_shape[3];
    auto old_length_x_head_size = shape_[2] * new_shape[3];
    shape_[2] = new_shape[2];

    for (int i = 0; i < layer_count_ * 2; i++) {
      OrtValue& present = *presents_[i];
      std::unique_ptr<OrtValue> past = OrtValue::CreateTensor(Allocator(), shape_, type_);

      auto past_span = WrapTensor<T>(Device(), *past);
      auto present_span = WrapTensor<T>(Device(), present);

      for (int j = 0; j < batch_x_num_heads; j++) {
        auto present_data = present_span.subspan(j * old_length_x_head_size, new_length_x_head_size);
        auto past_data = past_span.subspan(j * new_length_x_head_size, new_length_x_head_size);
        past_data.CopyFrom(present_data);
      }
      pasts_[i] = std::move(past);
      state_.inputs_[input_index_ + i] = pasts_[i].get();
    }
  }
}

// Copy present state to past state reordered by the beam_indices
template <typename ScoreType>
void DefaultKeyValueCache::PickPastState(DeviceSpan<int32_t> beam_indices_device, int index) {
  std::span<int32_t> beam_indices = beam_indices_device.CopyDeviceToCpu();

  std::array<int64_t, 4> tensor_shape;
  if (!layer_shapes_.empty()) {
    // Get shape from the actual tensor for per-layer allocation
    OrtValue& present_value = *presents_[index];
    const auto present_shape = present_value.GetTensorTypeAndShapeInfo()->GetShape();
    std::copy(present_shape.begin(), present_shape.end(), tensor_shape.begin());
  } else {
    tensor_shape = shape_;
  }

  auto block_size_per_beam = tensor_shape[1] * tensor_shape[2] * tensor_shape[3];

  OrtValue& present_value = *presents_[index];
  std::unique_ptr<OrtValue> past_value = OrtValue::CreateTensor<ScoreType>(Allocator(), tensor_shape);

  auto past_span = WrapTensor<ScoreType>(Device(), *past_value);
  auto present_span = WrapTensor<ScoreType>(Device(), present_value);

  for (size_t j = 0; j < beam_indices.size(); j++) {
    int32_t beam_index = beam_indices[j];
    auto present = present_span.subspan(beam_index * block_size_per_beam, block_size_per_beam);
    auto past = past_span.subspan(j * block_size_per_beam, block_size_per_beam);
    past.CopyFrom(present);
  }

  pasts_[index] = std::move(past_value);
}

void DefaultKeyValueCache::PickPastState(DeviceSpan<int32_t> beam_indices, int index) {
  if (type_ == Ort::TypeToTensorType<float>) {
    PickPastState<float>(beam_indices, index);
  } else {
    PickPastState<Ort::Float16_t>(beam_indices, index);
  }
}

namespace {

int64_t GetElementsPerBeam(const AuxiliaryStateSet& state_set) {
  static std::mutex mutex;
  static std::unordered_map<const AuxiliaryStateSet*, int64_t> cache;

  const auto* key = &state_set;

  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
      return it->second;
    }
  }

  int64_t elements_per_beam = std::accumulate(
      state_set.shape.begin() + 1,
      state_set.shape.end(),
      int64_t{1},
      std::multiplies<int64_t>{});

  {
    std::lock_guard<std::mutex> lock(mutex);
    auto [it, inserted] = cache.emplace(key, elements_per_beam);
    if (!inserted) {
      // Another thread inserted the value first; use the existing one.
      return it->second;
    }
  }

  return elements_per_beam;
}

}  // namespace

template <typename ScoreType>
void DefaultKeyValueCache::PickPastAuxiliaryState(DeviceSpan<int32_t> beam_indices_device, AuxiliaryStateSet& state_set, int index) {
  std::span<int32_t> beam_indices = beam_indices_device.CopyDeviceToCpu();
  const int64_t elements_per_beam = GetElementsPerBeam(state_set);

  OrtValue& present_value = *state_set.presents[index];
  std::unique_ptr<OrtValue> past_value = OrtValue::CreateTensor(Allocator(), state_set.shape, state_set.type);

  auto past_span = WrapTensor<ScoreType>(Device(), *past_value);
  auto present_span = WrapTensor<ScoreType>(Device(), present_value);

  for (size_t j = 0; j < beam_indices.size(); ++j) {
    const int32_t beam_index = beam_indices[j];
    auto present = present_span.subspan(beam_index * elements_per_beam, elements_per_beam);
    auto past = past_span.subspan(j * elements_per_beam, elements_per_beam);
    past.CopyFrom(present);
  }

  state_set.pasts[index] = std::move(past_value);
}

void DefaultKeyValueCache::PickPastAuxiliaryState(DeviceSpan<int32_t> beam_indices, AuxiliaryStateSet& state_set, int index) {
  if (state_set.type == Ort::TypeToTensorType<float>) {
    PickPastAuxiliaryState<float>(beam_indices, state_set, index);
  } else {
    PickPastAuxiliaryState<Ort::Float16_t>(beam_indices, state_set, index);
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

bool IsCacheNeeded(const Model& model) {
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

bool HasAuxiliaryDecoderStateCache(const Model& model) {
  const auto& decoder_inputs = model.config_->model.decoder.inputs;
  return (!decoder_inputs.past_conv_state_names.empty() &&
          model.session_info_.HasInput(ComposeKeyValueName(decoder_inputs.past_conv_state_names, 0))) ||
         (!decoder_inputs.past_recurrent_state_names.empty() &&
          model.session_info_.HasInput(ComposeKeyValueName(decoder_inputs.past_recurrent_state_names, 0)));
}

}  // namespace

std::unique_ptr<KeyValueCache> CreateKeyValueCache(State& state) {
  // For OpenVINO Stateful models, they do not contain exposed past/present KV tensors.
  // In this case, 'IsCacheNeeded' below will return false. But in this case we need to create a
  // special 'ModelManagedKeyValueCache' object, and so we check this condition first.
  if (IsOpenVINOStatefulModel(state.model_)) {
    if (g_log.enabled)
      Log("info", "CreateKeyValueCache: Creating ModelManagedKeyValueCache");
    return std::make_unique<ModelManagedKeyValueCache>(state);
  }

  if (!IsCacheNeeded(state.model_)) {
    return nullptr;
  }

  if (HasAuxiliaryDecoderStateCache(state.model_)) {
    if (state.model_.config_->engine.dynamic_batching.has_value()) {
      throw std::runtime_error("Dynamic batching is not currently supported for models with auxiliary decoder state caches.");
    }
    if (state.model_.p_device_->GetType() == DeviceType::NvTensorRtRtx) {
      throw std::runtime_error("NvTensorRtRtx is not currently supported for models with auxiliary decoder state caches.");
    }
  }

  if (state.model_.p_device_->GetType() != DeviceType::NvTensorRtRtx &&
      state.model_.config_->model.decoder.sliding_window &&
      state.model_.config_->model.decoder.sliding_window->slide_key_value_cache) {
    if (HasAuxiliaryDecoderStateCache(state.model_)) {
      throw std::runtime_error("Sliding-window KV cache is not currently supported for models with auxiliary decoder state caches.");
    }
    return std::make_unique<WindowedKeyValueCache>(state);
  }

  return std::make_unique<DefaultKeyValueCache>(state);
}

}  // namespace Generators
