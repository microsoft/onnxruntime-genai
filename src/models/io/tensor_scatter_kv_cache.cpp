// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tensor_scatter_kv_cache.h"

#include "generator/generators.h"
#include "models/model.h"

#include <algorithm>

namespace Generators {
namespace {

DeviceInterface& DeviceForTensor(const OrtValue& value,
                                 DeviceInterface& model_device) {
  const bool on_cpu =
      value.GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_CPU;
  return on_cpu ? *GetDeviceInterface(DeviceType::CPU) : model_device;
}

void ValidateTensorElementType(ONNXTensorElementDataType actual,
                               ONNXTensorElementDataType expected,
                               const std::string& name) {
  if (actual != expected) {
    throw std::runtime_error(name + " has type " + TypeToString(actual) +
                             ", expected " + TypeToString(expected));
  }
}

}  // namespace

TensorScatterKeyValueCache::TensorScatterKeyValueCache(State& state)
    : state_{state},
      layer_count_{state.model_.config_->model.decoder.num_hidden_layers},
      cache_write_indices_name_{
          state.model_.config_->model.decoder.inputs.cache_write_indices},
      cache_write_indices_type_{state.model_.session_info_.GetInputDataType(
          cache_write_indices_name_)} {
  if (state_.params_->search.num_beams != 1) {
    throw std::runtime_error(
        "TensorScatterKeyValueCache does not support beam search");
  }
  if (cache_write_indices_type_ != Ort::TypeToTensorType<int32_t> &&
      cache_write_indices_type_ != Ort::TypeToTensorType<int64_t>) {
    throw std::runtime_error(
        "TensorScatterKeyValueCache cache_write_indices must be int32 or int64");
  }

  cache_write_indices_ = std::make_unique<Tensor>(
      state_.model_.p_device_inputs_, cache_write_indices_type_);
  const std::array<int64_t, 1> write_indices_shape{
      state_.params_->BatchBeamSize()};
  cache_write_indices_->CreateTensor(write_indices_shape, true);

  const auto& session_info = state_.model_.session_info_;
  const auto& config = state_.model_.config_->model.decoder;
  input_names_.reserve(layer_count_ * 2);
  output_names_.reserve(layer_count_ * 2);
  values_.reserve(layer_count_ * 2);

  for (int layer = 0; layer < layer_count_; ++layer) {
    input_names_.push_back(
        ComposeKeyValueName(config.inputs.past_key_names, layer));
    input_names_.push_back(
        ComposeKeyValueName(config.inputs.past_value_names, layer));
    output_names_.push_back(
        ComposeKeyValueName(config.outputs.present_key_names, layer));
    output_names_.push_back(
        ComposeKeyValueName(config.outputs.present_value_names, layer));
  }

  for (size_t i = 0; i < input_names_.size(); ++i) {
    const auto& input_name = input_names_[i];
    const auto& output_name = output_names_[i];
    if (!session_info.HasInput(input_name) ||
        !session_info.HasOutput(output_name)) {
      throw std::runtime_error(
          "TensorScatterKeyValueCache graph is missing cache pair " +
          input_name + " -> " + output_name);
    }

    auto shape = session_info.GetInputShape(input_name);
    const auto output_shape = session_info.GetOutputShape(output_name);
    if (shape.size() != 4 || output_shape.size() != 4) {
      throw std::runtime_error(
          "TensorScatterKeyValueCache tensors must be rank 4");
    }
    shape[0] = state_.params_->BatchBeamSize();
    if (shape[1] <= 0 || shape[2] <= 0 || shape[3] <= 0) {
      throw std::runtime_error(
          "TensorScatterKeyValueCache requires fixed non-batch dimensions");
    }
    if (cache_sequence_length_ == 0) {
      cache_sequence_length_ = static_cast<int>(shape[2]);
    } else if (shape[2] != cache_sequence_length_) {
      throw std::runtime_error(
          "TensorScatterKeyValueCache tensors have inconsistent sequence dimensions");
    }
    for (size_t axis = 1; axis < shape.size(); ++axis) {
      if (output_shape[axis] > 0 && output_shape[axis] != shape[axis]) {
        throw std::runtime_error(
            "TensorScatterKeyValueCache past/present shapes differ");
      }
    }

    const auto type = session_info.GetInputDataType(input_name);
    ValidateTensorElementType(
        session_info.GetOutputDataType(output_name), type, output_name);
    values_.push_back(OrtValue::CreateTensor(
        state_.model_.p_device_kvcache_->GetAllocator(), shape, type));
    ByteWrapTensor(*state_.model_.p_device_kvcache_, *values_.back()).Zero();
  }
}

void TensorScatterKeyValueCache::Add() {
  state_.input_names_.push_back(cache_write_indices_name_.c_str());
  state_.inputs_.push_back(cache_write_indices_->GetOrtTensor());

  for (size_t i = 0; i < values_.size(); ++i) {
    state_.input_names_.push_back(input_names_[i].c_str());
    state_.inputs_.push_back(values_[i].get());
    state_.output_names_.push_back(output_names_[i].c_str());
    state_.outputs_.push_back(values_[i].get());
  }
}

void TensorScatterKeyValueCache::Update(DeviceSpan<int32_t>,
                                        int total_length) {
  if (total_length <= 0 || total_length > cache_sequence_length_) {
    throw std::runtime_error(
        "TensorScatterKeyValueCache update exceeds the fixed cache capacity");
  }

  const int write_index = total_length - 1;
  if (cache_write_indices_type_ == Ort::TypeToTensorType<int32_t>) {
    auto values = cache_write_indices_->GetDeviceSpan<int32_t>();
    auto cpu_values = values.CpuSpan();
    std::fill(cpu_values.begin(), cpu_values.end(), write_index);
    values.CopyCpuToDevice();
  } else {
    auto values = cache_write_indices_->GetDeviceSpan<int64_t>();
    auto cpu_values = values.CpuSpan();
    std::fill(cpu_values.begin(), cpu_values.end(),
              static_cast<int64_t>(write_index));
    values.CopyCpuToDevice();
  }
}

void TensorScatterKeyValueCache::RewindTo(size_t) {
  throw std::runtime_error(
      "TensorScatterKeyValueCache does not support rewind");
}

void TensorScatterKeyValueCache::Initialize(
    const std::vector<std::unique_ptr<OrtValue>>& compact_values) {
  if (compact_values.size() != values_.size()) {
    throw std::runtime_error(
        "TensorScatterKeyValueCache received an unexpected prefill cache count");
  }

  for (size_t i = 0; i < values_.size(); ++i) {
    const auto source_info = compact_values[i]->GetTensorTypeAndShapeInfo();
    const auto target_info = values_[i]->GetTensorTypeAndShapeInfo();
    const auto source_shape = source_info->GetShape();
    const auto target_shape = target_info->GetShape();
    if (source_shape.size() != 4 ||
        source_shape[0] != target_shape[0] ||
        source_shape[1] != target_shape[1] ||
        source_shape[3] != target_shape[3] ||
        source_shape[2] > target_shape[2]) {
      throw std::runtime_error(
          "TensorScatterKeyValueCache prefill cache does not fit the decode cache");
    }
    ValidateTensorElementType(
        source_info->GetElementType(), target_info->GetElementType(),
        input_names_[i]);

    const size_t row_bytes =
        static_cast<size_t>(source_shape[2] * source_shape[3]) *
        Ort::SizeOf(source_info->GetElementType());
    const size_t target_stride =
        static_cast<size_t>(target_shape[2] * target_shape[3]) *
        Ort::SizeOf(target_info->GetElementType());
    auto& cache_device = *state_.model_.p_device_kvcache_;
    auto source = ByteWrapTensor(
        DeviceForTensor(*compact_values[i], cache_device), *compact_values[i]);
    auto target = ByteWrapTensor(cache_device, *values_[i]);
    const size_t rows =
        static_cast<size_t>(source_shape[0] * source_shape[1]);
    for (size_t row = 0; row < rows; ++row) {
      target.subspan(row * target_stride, row_bytes)
          .CopyFrom(source.subspan(row * row_bytes, row_bytes));
    }
  }
}

}  // namespace Generators
