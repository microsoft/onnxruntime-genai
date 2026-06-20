// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "kv_cache.h"
#include "windowed_kv_cache.h"
#include "../openvino/interface.h"
#include "../qnn/interface.h"
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

namespace {

// Auto-detect a fixed kv-cache shape from the model's past_key input shapes,
// and, when detected, apply the implied configuration:
//   - reject beam search (num_beams != 1),
//   - force past_present_share_buffer on,
//   - log info on the detected size,
//   - warn if search.max_length exceeds the detected size.
// Returns the detected static seq_len, or 0 if the model has symbolic
// kv-cache dims or per-layer static sizes disagree.
//
// Background: some compiled backends (e.g. AMD RyzenAI) emit models where the
// kv-cache seq_len dimension is a static positive integer instead of a
// symbolic dim. In that case the cache must be allocated to exactly that size
// and reused as a shared past/present buffer; max_length cannot drive the
// size because ORT rejects any tensor that doesn't match the model's static
// dim.
//
// Limitation: only uniform per-layer static sizes are recognised — every
// past_key layer must declare the same fixed seq_len. Models that declare
// different static seq_lens per layer (e.g. a mix of full-attention and
// sliding-window layers with distinct static caps) fall through to dynamic
// handling. Lifting this restriction would extend the existing layer_shapes_
// infrastructure used for per-layer head_dim detection in
// DefaultKeyValueCache: store the per-layer detected seq_len into
// layer_shapes_[i][2] instead of a single scalar, and let the share-buffer
// branch's per-layer loop do the rest. Deferred until a model in the wild
// actually needs it.
int64_t DetectAndConfigureFixedKvShape(const SessionInfo& session_info,
                                       const std::vector<std::string>& input_name_strings,
                                       int layer_count,
                                       const Config::Search& search,
                                       bool& past_present_share_buffer) {
  if (layer_count <= 0) return 0;

  // input_name_strings stores [past_key.0, past_value.0, past_key.1, past_value.1, ...].
  int64_t common_seq_len = 0;
  for (int i = 0; i < layer_count; ++i) {
    auto input_shape = session_info.GetInputShape(input_name_strings[i * 2]);
    if (input_shape.size() < 2) return 0;
    const int64_t seq_dim = input_shape[input_shape.size() - 2];
    if (seq_dim <= 0) return 0;  // symbolic/dynamic dim (typically -1)
    if (common_seq_len == 0) {
      common_seq_len = seq_dim;
    } else if (common_seq_len != seq_dim) {
      return 0;
    }
  }

  if (search.num_beams != 1) {
    throw std::runtime_error(
        "Beam search (num_beams > 1) is not supported for models with a fixed kv-cache "
        "shape (model expects seq_len=" +
        std::to_string(common_seq_len) + ").");
  }
  past_present_share_buffer = true;
  if (g_log.enabled) {
    Log("info", "DefaultKeyValueCache: auto-detected fixed kv-cache seq_len=" +
                    std::to_string(common_seq_len) +
                    "; allocating shared past/present buffer to that size.");
  }
  if (search.max_length > static_cast<int>(common_seq_len) &&
      g_log.enabled && g_log.warning) {
    Log("warning", "Model has fixed kv-cache seq_len=" +
                       std::to_string(common_seq_len) +
                       " but search.max_length=" +
                       std::to_string(search.max_length) +
                       "; cache is sized to the model's limit, so generation beyond it will fail.");
  }
  return common_seq_len;
}

}  // namespace

DefaultKeyValueCache::DefaultKeyValueCache(State& state)
    : state_{state},
      layer_count_{model_.config_->model.decoder.num_hidden_layers},
      past_present_share_buffer_{state_.params_->IsPastPresentShareBufferEnabled(model_.config_->model.type)},
      shape_{state_.params_->BatchBeamSize(), model_.config_->model.decoder.num_key_value_heads, 0, model_.config_->model.decoder.head_size} {
  if (g_log.enabled && g_log.warning && past_present_share_buffer_ != state_.params_->search.past_present_share_buffer)
    Log("warning", "past_present_share_buffer search option set to true, but has been disabled due to the current configuration. See https://aka.ms/generate_config for details");

  // Auto-discover which layer indices have KV cache inputs
  kv_layer_indices_.clear();
  {
    const auto& key_template = model_.config_->model.decoder.inputs.past_key_names;
    auto prefix = key_template.substr(0, key_template.find('%'));
    auto suffix = key_template.substr(key_template.find('%') + 2);
    for (const auto& name : model_.session_info_.GetInputNames()) {
      if (name.size() > prefix.size() + suffix.size() &&
          name.compare(0, prefix.size(), prefix) == 0 &&
          name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        auto idx_str = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
        kv_layer_indices_.push_back(std::stoi(idx_str));
      }
    }
    std::sort(kv_layer_indices_.begin(), kv_layer_indices_.end());
  }

  if (!kv_layer_indices_.empty()) {
    layer_count_ = static_cast<int>(kv_layer_indices_.size());
  }

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

  // Auto-detect per-layer KV cache shape from ONNX session input shapes.
  // Models like Gemma 4 use a dual attention pattern: sliding-window layers are GQA
  // (e.g. num_kv_heads=8, head_dim=256) while global/full-attention layers are MQA
  // (num_kv_heads=1, head_dim=512). Both the KV-head count and the head_dim can vary
  // per layer, so detect and store both. (Previously only head_dim was handled, which
  // left global layers with the uniform num_kv_heads and broke prefill binding, e.g.
  // "past_key_values.5.key index 1 Got 8 Expected 1".)
  {
    // Past KV inputs are [batch, num_kv_heads, seq, head_dim].
    constexpr size_t kKvHeadsAxis = 1;
    constexpr size_t kHeadDimAxis = 3;
    std::vector<int64_t> per_layer_kv_heads(layer_count_, shape_[kKvHeadsAxis]);
    std::vector<int64_t> per_layer_head_dim(layer_count_, shape_[kHeadDimAxis]);
    // num_kv_heads and head_dim are static per-layer model properties known ahead of
    // time, and shape_ already holds the config defaults (decoder.num_key_value_heads /
    // .head_size), so a single flag suffices: set it if any layer's KV shape differs from
    // those defaults. (The > 0 checks ignore any non-concrete dim, leaving that layer at
    // the config default.)
    bool has_per_layer_variation = false;
    for (int i = 0; i < layer_count_; ++i) {
      const auto input_shape = model_.session_info_.GetInputShape(input_name_strings_[i * 2]);
      if (input_shape.size() == 4) {
        if (input_shape[kKvHeadsAxis] > 0) per_layer_kv_heads[i] = input_shape[kKvHeadsAxis];
        if (input_shape[kHeadDimAxis] > 0) per_layer_head_dim[i] = input_shape[kHeadDimAxis];
        if (per_layer_kv_heads[i] != shape_[kKvHeadsAxis] ||
            per_layer_head_dim[i] != shape_[kHeadDimAxis])
          has_per_layer_variation = true;
      }
    }
    if (has_per_layer_variation) {
      if (layer_shapes_.empty()) {
        layer_shapes_.resize(layer_count_);
        for (int i = 0; i < layer_count_; ++i) {
          layer_shapes_[i] = shape_;
        }
      }
      for (int i = 0; i < layer_count_; ++i) {
        layer_shapes_[i][kKvHeadsAxis] = per_layer_kv_heads[i];
        layer_shapes_[i][kHeadDimAxis] = per_layer_head_dim[i];
      }
      if (g_log.enabled) {
        Log("info",
            "DefaultKeyValueCache: Detected per-layer KV shape variation "
            "(num_kv_heads/head_dim) across " +
                std::to_string(layer_count_) + " KV cache layers");
      }

      // Create per-layer empty past tensors since the KV shape varies across layers
      empty_pasts_.resize(layer_count_);
      for (int i = 0; i < layer_count_; ++i) {
        std::array<int64_t, 4> empty_shape = layer_shapes_[i];
        empty_shape[2] = 0;  // sequence length = 0 for empty past
        empty_pasts_[i] = OrtValue::CreateTensor(Allocator(), empty_shape, type_);
      }
    }
  }

  const int64_t fixed_kv_seq_len = DetectAndConfigureFixedKvShape(
      model_.session_info_, input_name_strings_, layer_count_,
      state_.params_->search, past_present_share_buffer_);

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
      // Use per-layer allocation based on sliding window layer indices.
      // If layer_shapes_ already exists (from head_dim auto-detection), preserve
      // the per-layer head_dim values — only update the sequence length dimension.
      if (layer_shapes_.empty()) {
        layer_shapes_.resize(layer_count_);
        for (int layer_idx = 0; layer_idx < layer_count_; ++layer_idx) {
          layer_shapes_[layer_idx] = shape_;
        }
      }

      // Set all layers to max_length (sequence dim only)
      for (int layer_idx = 0; layer_idx < layer_count_; ++layer_idx) {
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
    // For fixed kv-cache models the cache size comes from the model graph,
    // not from max_length — see the auto-detection block earlier in this ctor.
    const int64_t cache_seq_len = fixed_kv_seq_len > 0
                                      ? fixed_kv_seq_len
                                      : static_cast<int64_t>(state_.params_->search.max_length);
    shape_[2] = cache_seq_len;

    // If per-layer shapes exist (from head_dim auto-detection), update their sequence dim too
    if (!layer_shapes_.empty()) {
      for (int i = 0; i < layer_count_; ++i) {
        layer_shapes_[i][2] = cache_seq_len;
      }
    }
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
    // Use per-layer empty past when head_dim varies across layers
    if (!empty_pasts_.empty()) {
      state_.inputs_.push_back(empty_pasts_[i / 2].get());
    } else {
      state_.inputs_.push_back(empty_past_.get());
    }
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
      // If max_cache_length is 0 (unconstrained), use total_length directly
      current_shape[2] = (max_cache_length > 0) ? std::min(total_length, max_cache_length) : total_length;

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
      if (!empty_pasts_.empty()) {
        state_.inputs_[input_index_ + i] = empty_pasts_[i / 2].get();
      } else {
        state_.inputs_[input_index_ + i] = empty_past_.get();
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

bool StaticScatterKeyValueCache::IsStaticScatterCache(const Model& model) {
  // Both driver inputs must be present. input_ids.cpp only produces the indices
  // when it sees write_indices AND nonpad_kv_seqlen, so requiring just one here
  // would create the cache for a model whose indices never get bound, surfacing
  // as an obscure unbound-input error at Run. Keep this predicate in lockstep
  // with the producer gate in input_ids.cpp.
  return model.session_info_.HasInput(model.config_->model.decoder.inputs.write_indices) &&
         model.session_info_.HasInput(model.config_->model.decoder.inputs.nonpad_kv_seqlen);
}

StaticScatterKeyValueCache::StaticScatterKeyValueCache(State& state)
    : state_{state},
      layer_count_{model_.config_->model.decoder.num_hidden_layers} {
  if (state_.params_->search.num_beams != 1) {
    throw std::runtime_error("Beam search (num_beams > 1) is not supported by the static-scatter KV cache.");
  }

  // Auto-discover which layer indices have KV cache inputs (mirrors
  // DefaultKeyValueCache so sparse/hybrid layouts work the same way).
  {
    const auto& key_template = model_.config_->model.decoder.inputs.past_key_names;
    auto prefix = key_template.substr(0, key_template.find('%'));
    auto suffix = key_template.substr(key_template.find('%') + 2);
    for (const auto& name : model_.session_info_.GetInputNames()) {
      if (name.size() > prefix.size() + suffix.size() &&
          name.compare(0, prefix.size(), prefix) == 0 &&
          name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        auto idx_str = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
        kv_layer_indices_.push_back(std::stoi(idx_str));
      }
    }
    std::sort(kv_layer_indices_.begin(), kv_layer_indices_.end());
  }

  if (!kv_layer_indices_.empty()) {
    layer_count_ = static_cast<int>(kv_layer_indices_.size());
  }

  for (int i = 0; i < layer_count_; ++i) {
    int layer_idx = kv_layer_indices_.empty() ? i : kv_layer_indices_[i];
    input_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.inputs.past_key_names, layer_idx));
    input_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.inputs.past_value_names, layer_idx));
    output_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.present_key_names, layer_idx));
    output_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.present_value_names, layer_idx));
  }

  type_ = model_.session_info_.GetInputDataType(input_name_strings_[0]);

  // Each KV input declares shape [batch, max_seq_len, kv_hidden]. The batch dim
  // is a runtime property (symbolic in the graph), so take it from the params
  // like DefaultKeyValueCache; only max_seq_len and kv_hidden must be static.
  // kv_hidden = num_kv_heads * head_dim and may vary per layer, so read each
  // layer's own declared shape rather than assuming a uniform value.
  const int64_t batch_size = state_.params_->BatchBeamSize();
  layer_shapes_.resize(layer_count_ * 2);
  caches_.reserve(layer_count_ * 2);
  for (int i = 0; i < layer_count_ * 2; ++i) {
    const auto input_shape = model_.session_info_.GetInputShape(input_name_strings_[i]);
    if (input_shape.size() != 3) {
      throw std::runtime_error(
          "StaticScatterKeyValueCache expects 3D [batch, max_seq_len, kv_hidden] KV inputs, but '" +
          input_name_strings_[i] + "' has rank " + std::to_string(input_shape.size()) + ".");
    }
    // max_seq_len (axis 1) and kv_hidden (axis 2) size the fixed buffer and must
    // be concrete; the batch dim (axis 0) is allowed to be symbolic.
    for (size_t axis = 1; axis < 3; ++axis) {
      if (input_shape[axis] <= 0) {
        throw std::runtime_error(
            "StaticScatterKeyValueCache requires a static max_seq_len and kv_hidden, but '" +
            input_name_strings_[i] + "' has a non-concrete dim at axis " + std::to_string(axis) + ".");
      }
    }
    std::array<int64_t, 3> tensor_shape{batch_size, input_shape[1], input_shape[2]};
    layer_shapes_[i] = tensor_shape;

    caches_.push_back(OrtValue::CreateTensor(Allocator(), tensor_shape, type_));
    if (Device().GetType() != DeviceType::WEBGPU) {
      ByteWrapTensor(Device(), *caches_.back()).Zero();
    }
  }
}

void StaticScatterKeyValueCache::Add() {
  input_index_ = state_.inputs_.size();
  output_index_ = state_.outputs_.size();

  // Past and present share one buffer: TensorScatter writes new rows in place,
  // so key_cache.{i} (input) and updated_key_cache.{i} (output) point at the
  // same OrtValue and never need rebinding between steps.
  for (int i = 0; i < layer_count_ * 2; ++i) {
    state_.inputs_.push_back(caches_[i].get());
    state_.input_names_.push_back(input_name_strings_[i].c_str());
    state_.outputs_.push_back(caches_[i].get());
    state_.output_names_.push_back(output_name_strings_[i].c_str());
  }
}

void StaticScatterKeyValueCache::Update(DeviceSpan<int32_t> /*beam_indices*/, int /*total_length*/) {
  // No-op: the shared buffer is updated in place by the graph's TensorScatter,
  // and the write offset / valid length are carried by the write_indices /
  // nonpad_kv_seqlen inputs (see input_ids.cpp), not by rebinding tensors here.
}

void StaticScatterKeyValueCache::RewindTo(size_t /*index*/) {
  // Fail loud: rewind is NOT wired for the static-scatter cache. The
  // write_indices/nonpad_kv_seqlen stream lives in InputIDs and has no RewindTo
  // hook, so a silent no-op here would leave the index tracker stale -> wrong
  // scatter slots and an over-reported nonpad => silently wrong logits with no
  // error. Throw until rewind is properly wired, matching the LFM2Cache and
  // WindowedKeyValueCache siblings.
  throw std::runtime_error("StaticScatterKeyValueCache does not support RewindTo.");
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

std::unique_ptr<KeyValueCache> CreateKeyValueCache(State& state) {
  // For OpenVINO and QNN Stateful models, they do not contain exposed past/present KV tensors.
  // In this case, 'IsCacheNeeded' below will return false. But in this case we need to create a
  // special 'ModelManagedKeyValueCache' object, and so we check this condition first.
  if (IsOpenVINOStatefulModel(state.model_) || IsQNNStatefulModel(state.model_)) {
    if (g_log.enabled)
      Log("info", "CreateKeyValueCache: Creating ModelManagedKeyValueCache");
    return std::make_unique<ModelManagedKeyValueCache>(state);
  }

  // LFM2 models interleave attention and conv layers, requiring a cache that handles
  // both KV cache for attention layers and fixed-size conv state for conv layers.
  if (ModelType::IsLFM2(state.model_.config_->model.type)) {
    return std::make_unique<LFM2Cache>(state);
  }

  if (!IsCacheNeeded(state.model_)) {
    return nullptr;
  }

  // mobius static-cache decoders drive an in-place TensorScatter KV buffer via
  // the write_indices input; auto-detect that (no user-visible search flag,
  // mirroring DetectAndConfigureFixedKvShape) before the default fallback.
  if (StaticScatterKeyValueCache::IsStaticScatterCache(state.model_)) {
    if (g_log.enabled)
      Log("info", "CreateKeyValueCache: Creating StaticScatterKeyValueCache");
    return std::make_unique<StaticScatterKeyValueCache>(state);
  }

  if (state.model_.p_device_->GetType() != DeviceType::NvTensorRtRtx &&
      state.model_.config_->model.decoder.sliding_window &&
      state.model_.config_->model.decoder.sliding_window->slide_key_value_cache) {
    return std::make_unique<WindowedKeyValueCache>(state);
  }

  return std::make_unique<DefaultKeyValueCache>(state);
}

}  // namespace Generators
