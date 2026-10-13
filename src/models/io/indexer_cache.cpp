// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "models/io/indexer_cache.h"
#include "models/io/static_kv_cache.h"
#include <algorithm>

namespace Generators {
namespace {

std::string ComposeIndexerName(const std::string& name_template, int layer_index) {
  constexpr size_t buffer_size = 128;
  char name[buffer_size];
  const int length = snprintf(name, buffer_size, name_template.c_str(), layer_index);
  if (length < 0 || static_cast<size_t>(length) >= buffer_size)
    throw std::runtime_error("Unable to compose indexer cache name from template " + name_template);
  return name;
}

}  // namespace

IndexerCache::IndexerCache(State& state) : state_{state} {
  const auto& inputs = model_.config_->model.decoder.inputs;
  const auto& outputs = model_.config_->model.decoder.outputs;
  const auto placeholder = inputs.past_indexer_names.find("%d");
  if (placeholder == std::string::npos) return;

  const auto prefix = inputs.past_indexer_names.substr(0, placeholder);
  const auto suffix = inputs.past_indexer_names.substr(placeholder + 2);
  for (const auto& name : model_.session_info_.GetInputNames()) {
    if (name.size() > prefix.size() + suffix.size() &&
        name.compare(0, prefix.size(), prefix) == 0 &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      layer_indices_.push_back(std::stoi(name.substr(prefix.size(), name.size() - prefix.size() - suffix.size())));
    }
  }
  std::sort(layer_indices_.begin(), layer_indices_.end());
  if (layer_indices_.empty()) return;
  if (outputs.present_indexer_names.empty())
    throw std::runtime_error("IndexerCache: present indexer name template must be configured");

  for (int layer_index : layer_indices_) {
    input_name_strings_.push_back(ComposeIndexerName(inputs.past_indexer_names, layer_index));
    output_name_strings_.push_back(ComposeIndexerName(outputs.present_indexer_names, layer_index));
    if (!model_.session_info_.HasOutput(output_name_strings_.back()))
      throw std::runtime_error("IndexerCache: missing output for layer " + std::to_string(layer_index));
  }
  const auto& update_value_template =
      outputs.state_update_indexer_value_names;
  const auto& update_row_template =
      outputs.state_update_indexer_row_names;
  if (update_value_template.empty() != update_row_template.empty()) {
    throw std::runtime_error(
        "IndexerCache: state update value and row templates must be configured together");
  }
  if (!update_value_template.empty()) {
    for (int layer_index : layer_indices_) {
      state_update_value_name_strings_.push_back(
          ComposeIndexerName(update_value_template, layer_index));
      state_update_row_name_strings_.push_back(
          ComposeIndexerName(update_row_template, layer_index));
    }
    state_update_values_.resize(layer_indices_.size());
    state_update_rows_values_.resize(layer_indices_.size());
  }

  type_ = model_.session_info_.GetInputDataType(input_name_strings_[0]);
  shape_ = model_.session_info_.GetInputShape(input_name_strings_[0]);
  if (shape_.size() != 3)
    throw std::runtime_error("IndexerCache: expected rank-3 cache tensors");
  shape_[0] = state_.params_->BatchBeamSize();
  if (shape_[2] <= 0)
    throw std::runtime_error("IndexerCache: head dimension must be static");
  const int64_t fixed_sequence_length = shape_[1] > 0 ? shape_[1] : 0;
  share_buffer_ = ShouldUseSharedPastPresentKeyValueCache(state_);
  if (fixed_sequence_length > 0) {
    if (state_.params_->search.num_beams != 1)
      throw std::runtime_error("IndexerCache: beam search is not supported with a fixed cache shape");
    share_buffer_ = true;
  }

  for (const auto& output_name : output_name_strings_) {
    const auto output_shape = model_.session_info_.GetOutputShape(output_name);
    if (output_shape.size() != 3 || output_shape[2] != shape_[2])
      throw std::runtime_error("IndexerCache: input and output cache head dimensions must match");
    if (fixed_sequence_length > 0 && output_shape[1] > 0 && output_shape[1] != fixed_sequence_length)
      throw std::runtime_error("IndexerCache: fixed input and output cache shapes must match");
  }

  shape_[1] = share_buffer_
                  ? (fixed_sequence_length > 0 ? fixed_sequence_length : state_.params_->search.max_length)
                  : 0;
  if (share_buffer_ && shape_[1] <= 0)
    throw std::runtime_error("IndexerCache: shared caches require search.max_length > 0");

  auto& allocator = model_.p_device_kvcache_->GetAllocator();
  pasts_.resize(layer_indices_.size());
  presents_.resize(layer_indices_.size());
  empty_pasts_.reserve(layer_indices_.size());
  for (size_t index = 0; index < layer_indices_.size(); ++index)
    empty_pasts_.push_back(OrtValue::CreateTensor(allocator, shape_, type_));

  const auto& length_name = inputs.past_sequence_length;
  if (!length_name.empty() && model_.session_info_.HasInput(length_name)) {
    if (model_.session_info_.GetInputDataType(length_name) != Ort::TypeToTensorType<int32_t>)
      throw std::runtime_error("IndexerCache: past_sequence_length input must be int32");
    past_sequence_length_ = OrtValue::CreateTensor(
        model_.allocator_cpu_, std::array<int64_t, 1>{1}, Ort::TypeToTensorType<int32_t>);
  } else if (share_buffer_) {
    throw std::runtime_error("IndexerCache: shared caches require an int32 past_sequence_length input");
  }
}

void IndexerCache::Add() {
  if (layer_indices_.empty()) return;
  input_index_ = state_.inputs_.size();
  output_index_ = state_.outputs_.size();
  for (size_t index = 0; index < layer_indices_.size(); ++index) {
    state_.inputs_.push_back(empty_pasts_[index].get());
    state_.input_names_.push_back(input_name_strings_[index].c_str());
    state_.outputs_.push_back(share_buffer_ ? empty_pasts_[index].get() : nullptr);
    state_.output_names_.push_back(output_name_strings_[index].c_str());
  }
  if (past_sequence_length_) {
    state_.inputs_.push_back(past_sequence_length_.get());
    state_.input_names_.push_back(model_.config_->model.decoder.inputs.past_sequence_length.c_str());
  }
  if (!state_update_value_name_strings_.empty()) {
    state_update_output_index_ = state_.outputs_.size();
    for (size_t index = 0; index < layer_indices_.size(); ++index) {
      state_.outputs_.push_back(nullptr);
      state_.output_names_.push_back(
          state_update_value_name_strings_[index].c_str());
      state_.outputs_.push_back(nullptr);
      state_.output_names_.push_back(
          state_update_row_name_strings_[index].c_str());
    }
  }
}

void IndexerCache::Update(DeviceSpan<int32_t> beam_indices, int total_length, int current_length) {
  if (!beam_indices.empty())
    throw std::runtime_error("IndexerCache does not support beam reordering");
  if (past_sequence_length_) {
    if (current_length < 0 || current_length > total_length)
      throw std::runtime_error("IndexerCache: current length must be in [0, total length]");
    *past_sequence_length_->GetTensorMutableData<int32_t>() = total_length - current_length;
  }
  if (share_buffer_) {
    if (total_length > shape_[1])
      throw std::runtime_error("IndexerCache: total length exceeds the shared cache capacity");
    if (!state_update_value_name_strings_.empty()) {
      state_update_length_ = static_cast<size_t>(current_length);
      auto& allocator = model_.p_device_kvcache_->GetAllocator();
      for (size_t index = 0; index < layer_indices_.size(); ++index) {
        state_update_values_[index] = OrtValue::CreateTensor(
            allocator,
            std::array<int64_t, 3>{
                state_.params_->BatchBeamSize(), current_length, shape_[2]},
            type_);
        state_update_rows_values_[index] = OrtValue::CreateTensor(
            allocator,
            std::array<int64_t, 2>{
                state_.params_->BatchBeamSize(), current_length},
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
        auto rows = WrapTensor<int32_t>(
            *model_.p_device_kvcache_, *state_update_rows_values_[index]);
        std::fill(rows.CpuSpan().begin(), rows.CpuSpan().end(), -1);
        rows.CopyCpuToDevice();
        state_.outputs_[state_update_output_index_ + index * 2] =
            state_update_values_[index].get();
        state_.outputs_[state_update_output_index_ + index * 2 + 1] =
            state_update_rows_values_[index].get();
      }
    }
    return;
  }

  if (!first_update_) {
    for (size_t index = 0; index < layer_indices_.size(); ++index) {
      pasts_[index] = std::move(presents_[index]);
      state_.inputs_[input_index_ + index] = pasts_[index].get();
    }
  }

  auto output_shape = shape_;
  output_shape[1] = total_length;
  auto& allocator = model_.p_device_kvcache_->GetAllocator();
  for (size_t index = 0; index < layer_indices_.size(); ++index) {
    presents_[index] = OrtValue::CreateTensor(allocator, output_shape, type_);
    state_.outputs_[output_index_ + index] = presents_[index].get();
  }
  first_update_ = false;
}

void IndexerCache::CommitAcceptedPrefix(size_t token_count) {
  if (!HasStateUpdates() || token_count > state_update_length_) {
    throw std::runtime_error(
        "IndexerCache accepted-prefix commit exceeds captured state updates");
  }
  RewindTo(snapshot_position_);
  auto& device = *model_.p_device_kvcache_;
  const size_t representative_capacity =
      static_cast<size_t>(shape_[1]) / 4;

  for (size_t layer = 0; layer < layer_indices_.size(); ++layer) {
    auto cache = ByteWrapTensor(device, *empty_pasts_[layer]);
    auto updates = ByteWrapTensor(device, *state_update_values_[layer]);
    for (size_t token = 0; token < token_count; ++token) {
      const size_t position = snapshot_position_ + token;
      const size_t row =
          position % 4 == 3 ? position / 4
                            : representative_capacity + position % 4;
      if (row >= static_cast<size_t>(shape_[1])) {
        throw std::runtime_error(
            "IndexerCache state update contains invalid cache row " +
            std::to_string(row) + " at layer " + std::to_string(layer) +
            " token " + std::to_string(token) + " with capacity " +
            std::to_string(shape_[1]));
      }
      cache
          .subspan(row * snapshot_row_bytes_, snapshot_row_bytes_)
          .CopyFrom(updates.subspan(token * snapshot_row_bytes_,
                                    snapshot_row_bytes_));
    }
  }
}

void IndexerCache::Snapshot(size_t position) {
  if (!share_buffer_ || empty_pasts_.empty()) return;

  const size_t capacity = static_cast<size_t>(shape_[1]);
  snapshot_rows_.clear();
  const auto add_row = [&](size_t row) {
    if (row < capacity &&
        std::find(snapshot_rows_.begin(), snapshot_rows_.end(), row) ==
            snapshot_rows_.end()) {
      snapshot_rows_.push_back(row);
    }
  };

  // Qwen's compressed indexer cache stores completed block representatives in the first quarter
  // and the current four raw keys in scratch rows immediately after that prefix.
  const size_t representative_capacity = capacity / 4;
  add_row(position / 4);
  for (size_t row = 0; row < 4; ++row) {
    add_row(representative_capacity + row);
  }
  // Also preserve ordinary append positions for non-compressed/fallback layouts.
  for (size_t row = position; row < position + 8; ++row) {
    add_row(row);
  }
  std::sort(snapshot_rows_.begin(), snapshot_rows_.end());

  snapshot_row_bytes_ =
      static_cast<size_t>(shape_[2]) * Ort::SizeOf(type_);
  const size_t total_bytes =
      layer_indices_.size() * snapshot_rows_.size() * snapshot_row_bytes_;
  auto& device = *model_.p_device_kvcache_;
  if (snapshot_data_.size() != total_bytes) {
    snapshot_data_ = device.Allocate<uint8_t>(total_bytes);
  }
  for (size_t layer = 0; layer < empty_pasts_.size(); ++layer) {
    auto cache = ByteWrapTensor(device, *empty_pasts_[layer]);
    for (size_t row_index = 0; row_index < snapshot_rows_.size(); ++row_index) {
      snapshot_data_
          .subspan(
              (layer * snapshot_rows_.size() + row_index) * snapshot_row_bytes_,
              snapshot_row_bytes_)
          .CopyFrom(cache.subspan(
              snapshot_rows_[row_index] * snapshot_row_bytes_,
              snapshot_row_bytes_));
    }
  }
  snapshot_position_ = position;
}

void IndexerCache::RewindTo(size_t index) {
  if (layer_indices_.empty()) return;
  if (share_buffer_) {
    if (index > static_cast<size_t>(shape_[1]))
      throw std::runtime_error("IndexerCache rewind exceeds the shared cache capacity");
    if (!snapshot_rows_.empty() && index == snapshot_position_) {
      auto& device = *model_.p_device_kvcache_;
      for (size_t layer = 0; layer < empty_pasts_.size(); ++layer) {
        auto cache = ByteWrapTensor(device, *empty_pasts_[layer]);
        for (size_t row_index = 0; row_index < snapshot_rows_.size(); ++row_index) {
          cache
              .subspan(
                  snapshot_rows_[row_index] * snapshot_row_bytes_,
                  snapshot_row_bytes_)
              .CopyFrom(snapshot_data_.subspan(
                  (layer * snapshot_rows_.size() + row_index) *
                      snapshot_row_bytes_,
                  snapshot_row_bytes_));
        }
      }
    }
    return;
  }
  if (index != 0)
    throw std::runtime_error("IndexerCache only supports rewinding to zero");
  first_update_ = true;
  for (size_t cache_index = 0; cache_index < layer_indices_.size(); ++cache_index) {
    pasts_[cache_index].reset();
    presents_[cache_index].reset();
    state_.inputs_[input_index_ + cache_index] = empty_pasts_[cache_index].get();
    state_.outputs_[output_index_ + cache_index] = nullptr;
  }
}

std::unique_ptr<IndexerCache> CreateIndexerCache(State& state) {
  auto cache = std::make_unique<IndexerCache>(state);
  return cache->IsEmpty() ? nullptr : std::move(cache);
}

}  // namespace Generators