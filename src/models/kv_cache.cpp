// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "kv_cache.h"
#include "windowed_kv_cache.h"
#include "../openvino/interface.h"
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
  if (g_log.enabled && g_log.warning && past_present_share_buffer_ != state_.params_->search.past_present_share_buffer)
    Log("warning", "past_present_share_buffer search option set to true, but has been disabled due to the current configuration. See https://aka.ms/generate_config for details");

  pasts_.resize(layer_count_ * 2);
  presents_.reserve(layer_count_ * 2);

  for (int i = 0; i < layer_count_; ++i) {
    input_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.inputs.past_key_names, i));
    input_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.inputs.past_value_names, i));

    output_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.present_key_names, i));
    output_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.present_value_names, i));
  }

  // Derive the KV data type from the KV input 0
  type_ = model_.session_info_.GetInputDataType(input_name_strings_[0]);
  empty_past_ = OrtValue::CreateTensor(Allocator(), shape_, type_);

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

      // Update sliding window layers with constrained cache size
      for (int layer_idx : model_.config_->model.decoder.sliding_window->layers) {
        layer_shapes_[layer_idx][2] = std::min(max_length, sliding_window_size);
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
}

void DefaultKeyValueCache::Update(DeviceSpan<int32_t> beam_indices, int total_length) {
  // If we're sharing past & present buffers there is nothing to do here, so early exit
  if (past_present_share_buffer_)
    return;

  if (!is_first_update_) {
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

  is_first_update_ = false;
}

void DefaultKeyValueCache::RewindTo(size_t index) {
  if (past_present_share_buffer_) {
    return;
  } else if (shape_[2] <= static_cast<int>(index)) {
    throw std::runtime_error("Requested length of rewind is greater than the current length.");
  }

  is_first_update_ = true;
  if (index == 0) {
    for (int i = 0; i < layer_count_ * 2; i++) {
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

LFM2Cache::LFM2Cache(State& state)
    : state_{state},
      layer_types_{model_.config_->model.decoder.layer_types},
      layer_count_{model_.config_->model.decoder.num_hidden_layers} {
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
    kv_empty_past_ = OrtValue::CreateTensor(Allocator(), kv_shape_, kv_type_);

    for (int i = 0; i < kv_layer_count_ * 2; ++i) {
      kv_presents_.push_back(OrtValue::CreateTensor(Allocator(), kv_shape_, kv_type_));
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
      throw std::runtime_error("LFM2Cache: expected conv state input to be rank 3 [B, H, L], got rank " + std::to_string(conv_shape_vec.size()));
    }
    // Replace batch dimension with actual batch size
    conv_shape_vec[0] = state_.params_->BatchBeamSize();

    for (int i = 0; i < conv_layer_count_; ++i) {
      std::array<int64_t, 3> shape;
      std::copy_n(conv_shape_vec.begin(), 3, shape.begin());
      conv_shapes_.push_back(shape);
      conv_pasts_[i] = OrtValue::CreateTensor(Allocator(), shape, conv_type_);
      // Zero-initialize conv state
      if (Device().GetType() != DeviceType::WEBGPU) {
        ByteWrapTensor(Device(), *conv_pasts_[i]).Zero();
      }
      conv_presents_.push_back(OrtValue::CreateTensor(Allocator(), shape, conv_type_));
    }
  }
}

void LFM2Cache::Add() {
  // Add KV cache inputs/outputs
  kv_input_index_ = state_.inputs_.size();
  for (int i = 0; i < kv_layer_count_ * 2; ++i) {
    state_.inputs_.push_back(kv_empty_past_.get());
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

void LFM2Cache::Update(DeviceSpan<int32_t> beam_indices, int total_length) {
  // --- Update KV cache (attention layers) ---
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

void LFM2Cache::RewindTo(size_t index) {
  // LFM2 uses conv layers with fixed-size rolling state buffers that depend on all prior tokens.
  // Rewinding the KV cache without replaying tokens through the conv layers would produce
  // incorrect results, so rewind is not supported for this cache type.
  throw std::runtime_error("LFM2Cache does not support RewindTo.");
}

template <typename ScoreType>
void LFM2Cache::PickPastState(DeviceSpan<int32_t> beam_indices_device, int index) {
  std::span<int32_t> beam_indices = beam_indices_device.CopyDeviceToCpu();
  std::array<int64_t, 4> tensor_shape = kv_shape_;

  auto block_size_per_beam = tensor_shape[1] * tensor_shape[2] * tensor_shape[3];

  OrtValue& present_value = *kv_presents_[index];
  std::unique_ptr<OrtValue> past_value = OrtValue::CreateTensor<ScoreType>(Allocator(), tensor_shape);

  auto past_span = WrapTensor<ScoreType>(Device(), *past_value);
  auto present_span = WrapTensor<ScoreType>(Device(), present_value);

  for (size_t j = 0; j < beam_indices.size(); j++) {
    int32_t beam_index = beam_indices[j];
    auto present = present_span.subspan(beam_index * block_size_per_beam, block_size_per_beam);
    auto past = past_span.subspan(j * block_size_per_beam, block_size_per_beam);
    past.CopyFrom(present);
  }

  kv_pasts_[index] = std::move(past_value);
}

void LFM2Cache::PickPastState(DeviceSpan<int32_t> beam_indices, int index) {
  if (kv_type_ == Ort::TypeToTensorType<float>) {
    PickPastState<float>(beam_indices, index);
  } else {
    PickPastState<Ort::Float16_t>(beam_indices, index);
  }
}

template <typename T>
void LFM2Cache::PickConvState(DeviceSpan<int32_t> beam_indices_device, int conv_index) {
  std::span<int32_t> beam_indices = beam_indices_device.CopyDeviceToCpu();
  auto& shape = conv_shapes_[conv_index];
  auto block_size_per_beam = shape[1] * shape[2];  // H * L

  OrtValue& present = *conv_presents_[conv_index];
  std::unique_ptr<OrtValue> past = OrtValue::CreateTensor<T>(Allocator(), shape);

  auto past_span = WrapTensor<T>(Device(), *past);
  auto present_span = WrapTensor<T>(Device(), present);

  for (size_t j = 0; j < beam_indices.size(); j++) {
    int32_t beam_index = beam_indices[j];
    auto present_data = present_span.subspan(beam_index * block_size_per_beam, block_size_per_beam);
    auto past_data = past_span.subspan(j * block_size_per_beam, block_size_per_beam);
    past_data.CopyFrom(present_data);
  }

  conv_pasts_[conv_index] = std::move(past);
}

namespace {

bool IsCacheNeeded(const Model& model) {
  return model.session_info_.HasInput(ComposeKeyValueName(model.config_->model.decoder.inputs.past_key_names, 0));
}

bool IsLFM2Model(const Model& model) {
  return !model.config_->model.decoder.layer_types.empty();
}

// For hybrid models, the first KV cache input may not be at layer 0.
// Find the first attention layer index to check if cache is needed.
bool IsLFM2CacheNeeded(const Model& model) {
  const auto& layer_types = model.config_->model.decoder.layer_types;
  for (int i = 0; i < static_cast<int>(layer_types.size()); ++i) {
    if (layer_types[i] == "full_attention") {
      return model.session_info_.HasInput(ComposeKeyValueName(model.config_->model.decoder.inputs.past_key_names, i));
    }
  }
  return false;
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

  // LFM2 models interleave attention and conv layers, requiring a cache that handles
  // both KV cache for attention layers and fixed-size conv state for conv layers.
  if (IsLFM2Model(state.model_) && IsLFM2CacheNeeded(state.model_)) {
    return std::make_unique<LFM2Cache>(state);
  }

  if (!IsCacheNeeded(state.model_)) {
    return nullptr;
  }

  if (state.model_.p_device_->GetType() != DeviceType::NvTensorRtRtx &&
      state.model_.config_->model.decoder.sliding_window &&
      state.model_.config_->model.decoder.sliding_window->slide_key_value_cache) {
    return std::make_unique<WindowedKeyValueCache>(state);
  }

  return std::make_unique<DefaultKeyValueCache>(state);
}

}  // namespace Generators
