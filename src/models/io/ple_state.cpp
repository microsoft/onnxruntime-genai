// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "models/io/ple_state.h"
#include <algorithm>

namespace Generators {
namespace {

std::string ComposePleName(const std::string& name_template, int layer_index) {
  constexpr size_t buffer_size = 128;
  char name[buffer_size];
  const int length = snprintf(name, buffer_size, name_template.c_str(), layer_index);
  if (length < 0 || static_cast<size_t>(length) >= buffer_size)
    throw std::runtime_error("Unable to compose PLE state name from template " + name_template);
  return name;
}

void FixAndValidateBatchDimension(std::vector<int64_t>& shape, int batch_size, const std::string& name) {
  if (shape.empty())
    throw std::runtime_error("PleState: " + name + " must have a batch dimension");
  if (shape[0] <= 0) shape[0] = batch_size;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    if (shape[axis] <= 0)
      throw std::runtime_error("PleState: " + name + " has unsupported dynamic dimension at axis " +
                               std::to_string(axis));
  }
}

}  // namespace

PleState::PleState(State& state) : state_{state} {
  const auto& inputs = model_.config_->model.decoder.inputs;
  const auto& outputs = model_.config_->model.decoder.outputs;
  const auto placeholder = inputs.past_ple_token_names.find("%d");
  if (placeholder == std::string::npos) return;

  const auto prefix = inputs.past_ple_token_names.substr(0, placeholder);
  const auto suffix = inputs.past_ple_token_names.substr(placeholder + 2);
  for (const auto& name : model_.session_info_.GetInputNames()) {
    if (name.size() > prefix.size() + suffix.size() &&
        name.compare(0, prefix.size(), prefix) == 0 &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      layer_indices_.push_back(std::stoi(name.substr(prefix.size(), name.size() - prefix.size() - suffix.size())));
    }
  }
  std::sort(layer_indices_.begin(), layer_indices_.end());
  if (layer_indices_.empty()) return;

  if (inputs.past_ple_conv_names.empty() || outputs.present_ple_token_names.empty() ||
      outputs.present_ple_conv_names.empty())
    throw std::runtime_error("PleState: all PLE input and output name templates must be configured");

  for (int layer_index : layer_indices_) {
    input_name_strings_.push_back(ComposePleName(inputs.past_ple_token_names, layer_index));
    input_name_strings_.push_back(ComposePleName(inputs.past_ple_conv_names, layer_index));
    output_name_strings_.push_back(ComposePleName(outputs.present_ple_token_names, layer_index));
    output_name_strings_.push_back(ComposePleName(outputs.present_ple_conv_names, layer_index));
    if (!model_.session_info_.HasInput(input_name_strings_[input_name_strings_.size() - 1]) ||
        !model_.session_info_.HasOutput(output_name_strings_[output_name_strings_.size() - 2]) ||
        !model_.session_info_.HasOutput(output_name_strings_[output_name_strings_.size() - 1]))
      throw std::runtime_error("PleState: incomplete PLE state pair for layer " + std::to_string(layer_index));
  }

  if (model_.session_info_.GetInputDataType(input_name_strings_[0]) != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
    throw std::runtime_error("PleState: token history must use INT64");
  conv_type_ = model_.session_info_.GetInputDataType(input_name_strings_[1]);
  token_shape_ = model_.session_info_.GetInputShape(input_name_strings_[0]);
  conv_shape_ = model_.session_info_.GetInputShape(input_name_strings_[1]);
  FixAndValidateBatchDimension(token_shape_, state_.params_->BatchBeamSize(), "token history");
  FixAndValidateBatchDimension(conv_shape_, state_.params_->BatchBeamSize(), "convolution state");

  auto& allocator = model_.p_device_kvcache_->GetAllocator();
  pasts_.reserve(layer_indices_.size() * 2);
  presents_.reserve(layer_indices_.size() * 2);
  for (size_t index = 0; index < layer_indices_.size(); ++index) {
    pasts_.push_back(OrtValue::CreateTensor(allocator, token_shape_, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
    pasts_.push_back(OrtValue::CreateTensor(allocator, conv_shape_, conv_type_));
    presents_.push_back(OrtValue::CreateTensor(allocator, token_shape_, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
    presents_.push_back(OrtValue::CreateTensor(allocator, conv_shape_, conv_type_));
  }
  InitializeStates(pasts_);
  InitializeStates(presents_);
}

void PleState::Add() {
  if (layer_indices_.empty()) return;
  input_index_ = state_.inputs_.size();
  output_index_ = state_.outputs_.size();
  for (size_t index = 0; index < pasts_.size(); ++index) {
    state_.inputs_.push_back(pasts_[index].get());
    state_.input_names_.push_back(input_name_strings_[index].c_str());
    state_.outputs_.push_back(presents_[index].get());
    state_.output_names_.push_back(output_name_strings_[index].c_str());
  }
}

void PleState::Update() {
  for (size_t index = 0; index < pasts_.size(); ++index) {
    std::swap(pasts_[index], presents_[index]);
    state_.inputs_[input_index_ + index] = pasts_[index].get();
    state_.outputs_[output_index_ + index] = presents_[index].get();
  }
  if (!pasts_.empty()) graph_buffer_variant_ ^= 1;
}

void PleState::RewindTo(size_t index) {
  if (layer_indices_.empty()) return;
  if (index != 0)
    throw std::runtime_error("PleState only supports rewinding to the initial state");
  InitializeStates(pasts_);
  InitializeStates(presents_);
}

void PleState::InitializeStates(std::vector<std::unique_ptr<OrtValue>>& states) {
  auto& device = *model_.p_device_kvcache_;
  for (size_t index = 0; index < states.size(); index += 2) {
    auto tokens = WrapTensor<int64_t>(device, *states[index]);
    std::fill(tokens.CpuSpan().begin(), tokens.CpuSpan().end(), model_.config_->model.decoder.ple_token_pad_id);
    tokens.CopyCpuToDevice();
    ByteWrapTensor(device, *states[index + 1]).Zero();
  }
}

std::unique_ptr<PleState> CreatePleState(State& state) {
  auto ple_state = std::make_unique<PleState>(state);
  return ple_state->IsEmpty() ? nullptr : std::move(ple_state);
}

}  // namespace Generators