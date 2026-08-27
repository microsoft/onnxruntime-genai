// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "static_kv_cache.h"

#include "quantized_kv_cache.h"
#include "windowed_kv_cache.h"
#include "../../config_utils.h"
#include "../../logging.h"
#include "../model.h"

#include <algorithm>
#include <sstream>
#include <unordered_map>

namespace Generators {

// Auto-detect a fixed kv-cache shape from the model's past_key input shapes,
// and, when detected, apply the implied configuration:
//   - reject beam search (num_beams != 1),
//   - force past_present_share_buffer on,
//   - log info on the detected size,
//   - warn if search.max_length exceeds the detected size.
// Returns the detected static seq_len, or 0 if the model has symbolic
// kv-cache dims or per-layer static sizes disagree.
//
// Background: some compiled backends emit models where the
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
// DefaultKeyValueCacheBase: store the per-layer detected seq_len into
// layer_shapes_[i][2] instead of a single scalar, and let the share-buffer
// branch's per-layer loop do the rest. Deferred until a model in the wild
// actually needs it.
int64_t DetectAndConfigureFixedKvShape(const SessionInfo& session_info,
                                       const std::vector<std::string>& input_name_strings,
                                       int layer_count,
                                       const Config::Search& search,
                                       bool& past_present_share_buffer,
                                       const char* cache_name) {
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
    Log("info", std::string(cache_name) + ": auto-detected fixed kv-cache seq_len=" +
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

namespace {

std::vector<std::string> MakePastKeyValueInputNames(const Model& model) {
  std::vector<int> kv_layer_indices;
  const auto& key_template = model.config_->model.decoder.inputs.past_key_names;
  const auto placeholder = key_template.find('%');
  if (placeholder == std::string::npos) {
    return {};
  }

  auto prefix = key_template.substr(0, placeholder);
  auto suffix = key_template.substr(placeholder + 2);
  for (const auto& name : model.session_info_.GetInputNames()) {
    if (name.size() > prefix.size() + suffix.size() &&
        name.compare(0, prefix.size(), prefix) == 0 &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      auto idx_str = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
      kv_layer_indices.push_back(std::stoi(idx_str));
    }
  }
  std::sort(kv_layer_indices.begin(), kv_layer_indices.end());

  const int layer_count = kv_layer_indices.empty()
                              ? model.config_->model.decoder.num_hidden_layers
                              : static_cast<int>(kv_layer_indices.size());
  std::vector<std::string> input_name_strings;
  input_name_strings.reserve(layer_count * 2);
  for (int i = 0; i < layer_count; ++i) {
    const int layer_idx = kv_layer_indices.empty() ? i : kv_layer_indices[i];
    input_name_strings.emplace_back(ComposeKeyValueName(model.config_->model.decoder.inputs.past_key_names, layer_idx));
    input_name_strings.emplace_back(ComposeKeyValueName(model.config_->model.decoder.inputs.past_value_names, layer_idx));
  }
  return input_name_strings;
}

}  // namespace

bool ShouldUseSharedPastPresentKeyValueCache(State& state) {
  bool past_present_share_buffer =
      state.params_->IsPastPresentShareBufferEnabled(state.model_.config_->model.type);
  const auto input_name_strings = MakePastKeyValueInputNames(state.model_);
  if (!input_name_strings.empty()) {
    DetectAndConfigureFixedKvShape(
        state.model_.session_info_, input_name_strings,
        static_cast<int>(input_name_strings.size() / 2),
        state.params_->search, past_present_share_buffer, "DefaultKeyValueCache");
  }
  return past_present_share_buffer;
}

DefaultKeyValueCacheBase::DefaultKeyValueCacheBase(State& state)
    : state_{state},
      layer_count_{model_.config_->model.decoder.num_hidden_layers},
      past_present_share_buffer_{state_.params_->IsPastPresentShareBufferEnabled(model_.config_->model.type)},
      shape_{state_.params_->BatchBeamSize(), model_.config_->model.decoder.num_key_value_heads, 0,
             model_.config_->model.decoder.head_size} {
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
      Log("info", "DefaultKeyValueCacheBase: Auto-discovered " + std::to_string(layer_count_) +
                      " KV cache layers at non-sequential indices");
    }
  }

  // Derive the KV data type from the first KV input
  type_ = model_.session_info_.GetInputDataType(input_name_strings_[0]);

  // Detect KV cache quantization configuration from the active provider's options.
  const int kv_cache_quantization_bits = Device().GetKeyValueCacheQuantizationBits(model_.config_->model.decoder.session_options);

  // When KV cache quantization is enabled, compute the compressed KV cache head dimension.
  if (kv_cache_quantization_bits == 4 || kv_cache_quantization_bits == 8) {
    shape_[3] = ComputeQuantizedKvCacheHeadSize(model_.config_->model.decoder.head_size, kv_cache_quantization_bits, type_);
    if (g_log.enabled) {
      Log("info", "DefaultKeyValueCacheBase: KV cache quantization " + std::to_string(kv_cache_quantization_bits) +
                      "-bit enabled, compressed kv_cache head_size=" + std::to_string(shape_[3]));
    }
  }

  empty_past_ = OrtValue::CreateTensor(Allocator(), shape_, type_);

  // Auto-detect per-layer KV cache shape from ONNX session input shapes.
  // Models like Gemma 4 use a dual attention pattern: sliding-window layers are GQA
  // (e.g. num_kv_heads=8, head_dim=256) while global/full-attention layers are MQA
  // (num_kv_heads=1, head_dim=512). Both the KV-head count and the head_dim can vary
  // per layer, so detect and store both. (Previously only head_dim was handled, which
  // left global layers with the uniform num_kv_heads and broke prefill binding, e.g.
  // "past_key_values.5.key index 1 Got 8 Expected 1".)
  // When KV cache quantization is active, the ONNX model reports uncompressed head dimensions;
  // we must apply compression before comparing/storing.
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
        if (input_shape[kHeadDimAxis] > 0) {
          // The ONNX model reports uncompressed head dimensions; when KV cache quantization
          // is active apply compression before comparing/storing (no-op when disabled).
          per_layer_head_dim[i] =
              ComputeQuantizedKvCacheHeadSize(static_cast<int>(input_shape[kHeadDimAxis]), kv_cache_quantization_bits, type_);
        }
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
            "DefaultKeyValueCacheBase: Detected per-layer KV shape variation "
            "(num_kv_heads/head_dim) across " +
                std::to_string(layer_count_) + " KV cache layers");
      }

      // Create per-layer empty past tensors since the KV shape varies across layers
      empty_pasts_.resize(layer_count_);
      for (int i = 0; i < layer_count_; ++i) {
        std::array<int64_t, 4> layer_empty_shape = layer_shapes_[i];
        layer_empty_shape[2] = 0;  // sequence length = 0 for empty past
        empty_pasts_[i] = OrtValue::CreateTensor(Allocator(), layer_empty_shape, type_);
      }
    }
  }

  const int64_t fixed_kv_seq_len = DetectAndConfigureFixedKvShape(
      model_.session_info_, input_name_strings_, layer_count_,
      state_.params_->search, past_present_share_buffer_, "DefaultKeyValueCache");

  if (state_.params_->use_graph_capture && !past_present_share_buffer_) {
    // share buffer is a precondition for graph capture
    throw std::runtime_error("Graph capture is not supported with past_present_share_buffer set to false.");
  }

  // Compute the capacity for sliding-window layers and apply it to the cache shapes.
  // ComputeWindowedKvCacheSize() returns 0 when the config does not call for a windowed cache.
  const int max_length = state_.params_->search.max_length;
  const int windowed_cache_size = Device().GetWindowedKeyValueCacheSize(
      model_.config_->model.decoder, state_.params_->search, max_length);

  if (windowed_cache_size > 0 &&
      // Beam reordering across a compacted cache is correct in principle but untested.
      state_.params_->search.num_beams == 1) {
    // Only a cache smaller than max_length ever evicts, and only then is RewindTo restricted.
    if (windowed_cache_size < max_length) {
      windowed_cache_size_ = windowed_cache_size;
    }

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
          layer_shapes_[it->second][2] = windowed_cache_size;
        }
      }
      // Set shape_[2] to max of all layer shapes for RewindTo bounds checking
      shape_[2] = max_length;
    } else {
      // Uniform sliding window allocation (backward compatibility)
      shape_[2] = windowed_cache_size;
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

      // Non-share-buffer caches start with a zero-length sequence dim; these
      // tensors are placeholders (Update() reallocates presents_ at the real
      // total_length before every Run), but some device allocators reject
      // zero-sized buffers. Let the active device policy clamp when needed.
      if (Device().ShouldClampZeroLengthKeyValueCacheOutputPlaceholders()) {
        tensor_shape[2] = std::max<int64_t>(1, tensor_shape[2]);
      }

      presents_.push_back(OrtValue::CreateTensor(Allocator(), tensor_shape, type_));
      if (Device().ShouldZeroKeyValueCacheTensors()) {
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

void DefaultKeyValueCacheBase::Add() {
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

// Copy present state to past state reordered by the beam_indices
template <typename ScoreType>
void DefaultKeyValueCacheBase::PickPastState(DeviceSpan<int32_t> beam_indices_device, int index) {
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
  std::unique_ptr<OrtValue> past_value = OrtValue::CreateTensor(Allocator(), tensor_shape, type_);

  auto past_span = KeyValueCacheDetail::WrapKvCacheTensor<ScoreType>(Device(), *past_value, type_);
  auto present_span = KeyValueCacheDetail::WrapKvCacheTensor<ScoreType>(Device(), present_value, type_);

  for (size_t j = 0; j < beam_indices.size(); j++) {
    int32_t beam_index = beam_indices[j];
    auto present = present_span.subspan(beam_index * block_size_per_beam, block_size_per_beam);
    auto past = past_span.subspan(j * block_size_per_beam, block_size_per_beam);
    past.CopyFrom(present);
  }

  pasts_[index] = std::move(past_value);
}

void DefaultKeyValueCacheBase::PickPastState(DeviceSpan<int32_t> beam_indices, int index) {
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

}  // namespace Generators
