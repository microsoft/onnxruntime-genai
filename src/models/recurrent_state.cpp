// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "env_utils.h"
#include "model.h"
#include "kv_cache.h"  // For ComposeKeyValueName
#include "recurrent_state.h"
#include <algorithm>

namespace Generators {

std::string ComposeCacheName(const std::string& template_string, int index) {
  constexpr int32_t CacheNameLength = 64;
  char cache_name[CacheNameLength];
  if (auto length = snprintf(cache_name, std::size(cache_name), template_string.c_str(), index);
      length < 0 || length >= CacheNameLength) {
    throw std::runtime_error("Unable to compose cache name from the provided template " + template_string +
                             ". This could be either due to an encoding error or the name being too long.");
  }
  return std::string(cache_name);
}

RecurrentState::RecurrentState(State& state)
    : state_{state} {
  // Discover recurrent layer indices by scanning all session input names
  const auto& past_recurrent_template = model_.config_->model.decoder.inputs.past_recurrent_names;
  const auto placeholder_pos = past_recurrent_template.find("%d");
  if (placeholder_pos == std::string::npos) return;

  const auto prefix = past_recurrent_template.substr(0, placeholder_pos);
  const auto suffix = past_recurrent_template.substr(placeholder_pos + 2);

  for (const auto& name : model_.session_info_.GetInputNames()) {
    // Try to match against the recurrent state template (e.g. "past.%d.recurrent")
    // Extract the layer index from names that match
    if (name.size() > prefix.size() + suffix.size() &&
        name.compare(0, prefix.size(), prefix) == 0 &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      auto idx_str = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
      int idx = std::stoi(idx_str);
      layer_indices_.push_back(idx);
    }
  }
  std::sort(layer_indices_.begin(), layer_indices_.end());

  if (layer_indices_.empty()) return;

  if (g_log.enabled)
    Log("info", "RecurrentState: Auto-discovered " + std::to_string(layer_indices_.size()) + " recurrent layers (indices: " + [&]() {
                      std::string s;
                      for (size_t i = 0; i < layer_indices_.size(); ++i) {
                        if (i) s += ",";
                        s += std::to_string(layer_indices_[i]);
                      }
                      return s; }() + ")");

  for (int idx : layer_indices_) {
    input_name_strings_.push_back(ComposeCacheName(model_.config_->model.decoder.inputs.past_conv_names, idx));
    input_name_strings_.push_back(ComposeCacheName(model_.config_->model.decoder.inputs.past_recurrent_names, idx));
    output_name_strings_.push_back(ComposeCacheName(model_.config_->model.decoder.outputs.present_conv_names, idx));
    output_name_strings_.push_back(ComposeCacheName(model_.config_->model.decoder.outputs.present_recurrent_names, idx));
  }

  conv_type_ = model_.session_info_.GetInputDataType(input_name_strings_[0]);
  recurrent_type_ = model_.session_info_.GetInputDataType(input_name_strings_[1]);

  // The window axis (when present) leads the batch axis, so the dynamic batch dim is at axis 1 for
  // a windowed shape (conv_state rank 4, recurrent_state rank 5) and at axis 0 otherwise.
  auto fix_batch_dim = [&](std::vector<int64_t> shape, size_t unwindowed_rank) -> std::vector<int64_t> {
    const size_t batch_axis = shape.size() > unwindowed_rank ? 1 : 0;
    if (shape.size() > batch_axis && shape[batch_axis] <= 0) {
      shape[batch_axis] = state_.params_->BatchBeamSize();
    }
    return shape;
  };

  conv_shape_ = fix_batch_dim(model_.session_info_.GetInputShape(input_name_strings_[0]), 3);
  recurrent_shape_ = fix_batch_dim(model_.session_info_.GetInputShape(input_name_strings_[1]), 4);

  // Validate all dims are positive (only batch dim is expected to be dynamic)
  auto validate_shape = [](const std::vector<int64_t>& shape, const std::string& name) {
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] <= 0)
        throw std::runtime_error("RecurrentState: " + name + " has unsupported dynamic dim " +
                                 std::to_string(shape[i]) + " at axis " + std::to_string(i));
    }
  };
  validate_shape(conv_shape_, "conv");
  validate_shape(recurrent_shape_, "recurrent");

  // A model built with `state_window=W` carries a LEADING window axis (conv_state
  // becomes rank 4 and recurrent_state rank 5). Only the last W per-token states are kept, which
  // is what lets the MTP loop crop to an accepted prefix without a replay forward. The window axis
  // leads the batch axis so each slot is one contiguous block, which makes CropToPosition a single
  // copy for any batch size. Both node types must agree on W. Rank-3 / rank-4 shapes are the
  // legacy unwindowed layout (W == 1).
  const bool conv_windowed = conv_shape_.size() == 4;
  const bool recurrent_windowed = recurrent_shape_.size() == 5;
  if (conv_windowed != recurrent_windowed)
    throw std::runtime_error("RecurrentState: conv_state and recurrent_state must both be windowed or neither");
  if (conv_windowed) {
    if (conv_shape_[0] != recurrent_shape_[0])
      throw std::runtime_error("RecurrentState: conv_state and recurrent_state must use the same state window");
    state_window_ = conv_shape_[0];
  }

  const int num_layers = static_cast<int>(layer_indices_.size());

  const bool share_buffers_configured =
      state_.params_->IsPastPresentShareBufferEnabled(model_.config_->model.type);

  // WebGPU prohibits binding the same buffer as both read-only (input) and
  // read-write (output) storage in the same compute pass, so it must use
  // separate past/present buffers with swap. All other EPs share buffers
  // for stable addresses (required by TRT-RTX graph replay, beneficial elsewhere).
  // TODO: Remove WebGPU special case once the ORT WebGPU EP adds a
  // LinearAttention kernel with native past/present buffer sharing support.
  const bool is_webgpu = model_.p_device_kvcache_->GetType() == DeviceType::WEBGPU;

  // Under CUDA-graph capture the recurrent (conv + linear-attention) state is updated in
  // place (present_state aliased onto past_state), and ORT re-runs the model several times
  // inside the first Run() of each captured shape to warm up and capture. Save/restore around
  // that first capture (see ShouldFixUpGraphCapture) keeps the update correct while using half
  // the recurrent-state memory and one graph variant. The environment override can disable
  // sharing to retain double buffering as a diagnostic fallback.
  bool share_under_graph_capture = true;
  GetEnv("ORTGENAI_SHARE_RECURRENT_STATE_UNDER_GRAPH_CAPTURE", share_under_graph_capture);
  const bool graph_capture_enabled = !is_webgpu && state_.params_->use_graph_capture;
  share_buffers_ = !is_webgpu &&
                   (graph_capture_enabled ? share_under_graph_capture : share_buffers_configured);
  graph_double_buffer_ = graph_capture_enabled && !share_buffers_;

  if (!share_buffers_) {
    pasts_.resize(num_layers * 2);
  }
  presents_.reserve(num_layers * 2);

  auto& allocator = model_.p_device_kvcache_->GetAllocator();

  for (int i = 0; i < num_layers; ++i) {
    if (!share_buffers_) {
      pasts_[i * 2] = OrtValue::CreateTensor(allocator, conv_shape_, conv_type_);
      pasts_[i * 2 + 1] = OrtValue::CreateTensor(allocator, recurrent_shape_, recurrent_type_);
    }
    presents_.push_back(OrtValue::CreateTensor(allocator, conv_shape_, conv_type_));
    presents_.push_back(OrtValue::CreateTensor(allocator, recurrent_shape_, recurrent_type_));
  }

  if (!share_buffers_) {
    ZeroStates(pasts_);
  }
  ZeroStates(presents_);
}

void RecurrentState::Add() {
  if (layer_indices_.empty()) return;

  input_index_ = state_.inputs_.size();
  output_index_ = state_.outputs_.size();

  const int num_layers = static_cast<int>(layer_indices_.size());
  for (int i = 0; i < num_layers * 2; ++i) {
    // Shared buffers: alias input=output for stable addresses (required for graph capture).
    // Separate buffers: use distinct past/present allocations with per-step pointer swap.
    state_.inputs_.push_back(share_buffers_ ? presents_[i].get() : pasts_[i].get());
    state_.input_names_.push_back(input_name_strings_[i].c_str());
    state_.outputs_.push_back(presents_[i].get());
    state_.output_names_.push_back(output_name_strings_[i].c_str());
  }
}

void RecurrentState::SetForwardLength(int sequence_length) {
  forward_length_ = sequence_length;
}

void RecurrentState::CropToPosition(size_t position) {
  if (!IsWindowed())
    throw std::runtime_error(
        "RecurrentState::CropToPosition requires the model exported with state_window > 1");
  // The live state is in presents_ (the buffers the last forward wrote; Update()/the swap has not
  // run yet at this point). Window slot j holds the state AFTER token (seq_len - W + j), so token
  // `position` lives in slot position + W - seq_len. Promote it to slot W-1, which is the slot the
  // ops read back as past_state on the next forward -- that alone commits the accepted prefix.
  const int64_t signed_slot = static_cast<int64_t>(position) + state_window_ - forward_length_;
  if (signed_slot < 0)
    throw std::runtime_error(
        "RecurrentState::CropToPosition(" + std::to_string(position) + ") is outside the state window of " +
        std::to_string(state_window_) + " for a forward of length " + std::to_string(forward_length_) +
        "; rebuild the model with a larger state_window");
  if (signed_slot >= state_window_)
    throw std::runtime_error(
        "RecurrentState::CropToPosition(" + std::to_string(position) + ") is past the last position of a forward of length " +
        std::to_string(forward_length_));
  const size_t slot = static_cast<size_t>(signed_slot);
  if (slot + 1 == static_cast<size_t>(state_window_)) return;  // Already the committed slot.

  auto& device = *model_.p_device_kvcache_;
  // Fast path: one kernel for all 2*num_layers tensors. The per-tensor loop below issues one
  // cudaMemcpyAsync each, and at 30 layers that is 60 host-side driver calls on a step that is
  // already GPU-idle-bound.
  if (TryBatchedSlotPromote(slot)) return;

  for (auto& present : presents_) {
    auto window = ByteWrapTensor(device, *present);
    const size_t slot_bytes = window.size() / static_cast<size_t>(state_window_);
    window.subspan((static_cast<size_t>(state_window_) - 1) * slot_bytes, slot_bytes)
        .CopyFrom(window.subspan(slot * slot_bytes, slot_bytes));
  }
}

bool RecurrentState::TryBatchedSlotPromote(size_t slot) {
  auto& device = *model_.p_device_kvcache_;

  // The state buffers are stable across steps when inputs alias outputs (which graph capture
  // requires), but re-derive the descriptors and compare so a reallocation cannot go unnoticed.
  bool descriptors_changed = slot_descs_cpu_.size() != presents_.size();
  slot_descs_cpu_.resize(presents_.size());
  for (size_t i = 0; i < presents_.size(); ++i) {
    auto& present = presents_[i];
    auto window = ByteWrapTensor(device, *present);
    auto bytes = window.Span();
    const StateSlotDesc desc{reinterpret_cast<uint8_t*>(bytes.data()),
                             static_cast<uint64_t>(bytes.size()) / static_cast<uint64_t>(state_window_)};
    descriptors_changed |= slot_descs_cpu_[i] != desc;
    slot_descs_cpu_[i] = desc;
  }

  if (descriptors_changed) {
    if (slot_descs_.size() != slot_descs_cpu_.size())
      slot_descs_ = device.Allocate<StateSlotDesc>(slot_descs_cpu_.size());
    std::copy(slot_descs_cpu_.begin(), slot_descs_cpu_.end(), slot_descs_.CpuSpan().begin());
    slot_descs_.CopyCpuToDevice();
  }

  return device.CopyStateSlots(slot_descs_.Span().data(), static_cast<int>(slot_descs_.size()),
                               static_cast<int>(slot), static_cast<int>(state_window_) - 1);
}

void RecurrentState::Update() {
  if (layer_indices_.empty() || share_buffers_) return;

  const int num_layers = static_cast<int>(layer_indices_.size());
  for (int i = 0; i < num_layers * 2; ++i) {
    std::swap(pasts_[i], presents_[i]);
    state_.inputs_[input_index_ + i] = pasts_[i].get();
    state_.outputs_[output_index_ + i] = presents_[i].get();
  }
  if (graph_double_buffer_) graph_buffer_variant_ ^= 1;
}

void RecurrentState::RewindTo(size_t index) {
  if (layer_indices_.empty()) return;

  if (index != 0) {
    // The recurrent (conv + linear-attention) state cannot be cropped like the
    // attention KV cache. A partial rewind is only possible by restoring a
    // snapshot captured at the target length (used by speculative decoding to
    // roll back a rejected draft). The caller (e.g. the MTP orchestrator)
    // guarantees the snapshot was taken at exactly `index`.
    if (snapshot_valid_) {
      if (index != snapshot_position_) {
        throw std::runtime_error(
            "RecurrentState::RewindTo(" + std::to_string(index) + ") cannot restore snapshot at length " +
            std::to_string(snapshot_position_));
      }
      RestoreSnapshot();
      return;
    }
    throw std::runtime_error(
        "RecurrentState::RewindTo(" + std::to_string(index) +
        ") is not supported without a snapshot. Recurrent states cannot be partially rewound; "
        "call Snapshot() at the target length first (e.g. for speculative decoding).");
  }
  // Full reset to length 0.
  snapshot_valid_ = false;
  if (share_buffers_) {
    // Shared buffers: zero in place, addresses stay stable.
    ZeroStates(presents_);
  } else {
    // Zero and rebind all state buffers.
    ZeroStates(pasts_);
    ZeroStates(presents_);
    const int num_layers = static_cast<int>(layer_indices_.size());
    for (int i = 0; i < num_layers * 2; ++i) {
      state_.inputs_[input_index_ + i] = pasts_[i].get();
      state_.outputs_[output_index_ + i] = presents_[i].get();
    }
  }
}

void RecurrentState::ZeroStates(std::vector<std::unique_ptr<OrtValue>>& states) {
  auto& device = *model_.p_device_kvcache_;
  for (auto& val : states) {
    ByteWrapTensor(device, *val).Zero();
  }
}

void RecurrentState::CopyStates(const std::vector<std::unique_ptr<OrtValue>>& src,
                                std::vector<std::unique_ptr<OrtValue>>& dst) {
  auto& device = *model_.p_device_kvcache_;
  for (size_t i = 0; i < src.size(); ++i) {
    ByteWrapTensor(device, *dst[i]).CopyFrom(ByteWrapTensor(device, *src[i]));
  }
}

void RecurrentState::Snapshot(size_t position) {
  if (layer_indices_.empty()) return;

  // The live state is in presents_ (shared-buffer EPs alias input==output to it;
  // non-shared EPs keep the latest in presents_ after Update()'s swap).
  if (snapshot_.empty()) {
    auto& allocator = model_.p_device_kvcache_->GetAllocator();
    const int num_layers = static_cast<int>(layer_indices_.size());
    snapshot_.reserve(num_layers * 2);
    for (int i = 0; i < num_layers; ++i) {
      snapshot_.push_back(OrtValue::CreateTensor(allocator, conv_shape_, conv_type_));
      snapshot_.push_back(OrtValue::CreateTensor(allocator, recurrent_shape_, recurrent_type_));
    }
  }
  CopyStates(presents_, snapshot_);
  snapshot_position_ = position;
  snapshot_valid_ = true;
}

void RecurrentState::RestoreSnapshot() {
  if (layer_indices_.empty()) return;
  if (!snapshot_valid_) {
    throw std::runtime_error("RecurrentState::RestoreSnapshot called before Snapshot");
  }
  // Copy back into the live buffers in place so their addresses stay stable
  // (required by CUDA-graph replay, which captures fixed buffer pointers).
  CopyStates(snapshot_, presents_);
}

bool RecurrentState::ShouldFixUpGraphCapture(int graph_id) const {
  if (layer_indices_.empty() || !share_buffers_ || !state_.params_->use_graph_capture)
    return false;
  return std::find(graph_capture_fixed_up_.begin(), graph_capture_fixed_up_.end(), graph_id) ==
         graph_capture_fixed_up_.end();
}

void RecurrentState::SaveForGraphCapture() {
  auto& device = *model_.p_device_kvcache_;
  graph_capture_backup_.clear();
  graph_capture_backup_.reserve(presents_.size());
  for (auto& present : presents_) {
    auto span = ByteWrapTensor(device, *present);
    span.CopyDeviceToCpu();
    graph_capture_backup_.push_back(std::move(span));
  }
}

void RecurrentState::RestoreAfterGraphCapture(int graph_id) {
  auto& device = *model_.p_device_kvcache_;
  for (auto& span : graph_capture_backup_) {
    span.CopyCpuToDevice();
  }
  device.Synchronize();
  graph_capture_backup_.clear();
  graph_capture_fixed_up_.push_back(graph_id);
}

std::unique_ptr<RecurrentState> CreateRecurrentState(State& state) {
  auto recurrent_state = std::make_unique<RecurrentState>(state);
  if (recurrent_state->IsEmpty()) {
    return nullptr;
  }
  return recurrent_state;
}

}  // namespace Generators
