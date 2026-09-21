// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "models/io/indexer_cache.h"
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

  type_ = model_.session_info_.GetInputDataType(input_name_strings_[0]);
  shape_ = model_.session_info_.GetInputShape(input_name_strings_[0]);
  if (shape_.size() != 3)
    throw std::runtime_error("IndexerCache: expected rank-3 cache tensors");
  shape_[0] = state_.params_->BatchBeamSize();
  if (shape_[2] <= 0)
    throw std::runtime_error("IndexerCache: head dimension must be static");
  share_buffer_ = shape_[1] > 0;
  if (share_buffer_) {
    for (const auto& output_name : output_name_strings_) {
      const auto output_shape = model_.session_info_.GetOutputShape(output_name);
      if (output_shape.size() != 3 || output_shape[1] != shape_[1] || output_shape[2] != shape_[2])
        throw std::runtime_error("IndexerCache: fixed input and output cache shapes must match");
    }
  } else {
    shape_[1] = 0;
  }

  auto& allocator = model_.p_device_kvcache_->GetAllocator();
  pasts_.resize(layer_indices_.size());
  presents_.resize(layer_indices_.size());
  empty_pasts_.reserve(layer_indices_.size());
  for (size_t index = 0; index < layer_indices_.size(); ++index)
    empty_pasts_.push_back(OrtValue::CreateTensor(allocator, shape_, type_));

  if (share_buffer_) {
    const auto& length_name = inputs.past_sequence_length;
    if (!model_.session_info_.HasInput(length_name) ||
        model_.session_info_.GetInputDataType(length_name) != Ort::TypeToTensorType<int32_t>)
      throw std::runtime_error("IndexerCache: fixed caches require an int32 past_sequence_length input");
    past_sequence_length_ = OrtValue::CreateTensor(
        model_.allocator_cpu_, std::array<int64_t, 1>{1}, Ort::TypeToTensorType<int32_t>);
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
}

void IndexerCache::Update(DeviceSpan<int32_t> beam_indices, int total_length, int current_length) {
  if (!beam_indices.empty())
    throw std::runtime_error("IndexerCache does not support beam reordering");
  if (share_buffer_) {
    if (current_length < 0 || current_length > total_length)
      throw std::runtime_error("IndexerCache: current length must be in [0, total length]");
    *past_sequence_length_->GetTensorMutableData<int32_t>() = total_length - current_length;
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

void IndexerCache::RewindTo(size_t index) {
  if (layer_indices_.empty()) return;
  if (index != 0)
    throw std::runtime_error("IndexerCache only supports rewinding to zero");
  if (share_buffer_)
    return;
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