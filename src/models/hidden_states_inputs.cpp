// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "hidden_states_inputs.h"

namespace Generators {

HiddenStatesInputs::HiddenStatesInputs(State& state)
    : state_{state} {
  const std::string& name = model_.config_->model.decoder.inputs.hidden_states;
  type_ = model_.session_info_.GetInputDataType(name);
  shape_ = {state_.params_->BatchBeamSize(), 0, model_.config_->model.decoder.hidden_size};
  value_ = std::make_unique<Tensor>(model_.p_device_inputs_, type_);
}

void HiddenStatesInputs::Add() {
  input_index_ = state_.inputs_.size();
  state_.inputs_.push_back(value_->GetOrtTensor());
  state_.input_names_.push_back(model_.config_->model.decoder.inputs.hidden_states.c_str());
}

void HiddenStatesInputs::Update(int sequence_length) {
  if (!pending_source_)
    throw std::runtime_error("HiddenStatesInputs::Update called before SetValue");

  // Resize the device tensor to [batch, sequence_length, hidden_size] if needed.
  if (static_cast<int64_t>(sequence_length) != shape_[1]) {
    shape_[1] = sequence_length;
    value_->CreateTensor(shape_);
    state_.inputs_[input_index_] = value_->GetOrtTensor();
  }

  // Copy the caller-provided values into the device tensor. The source is expected to hold
  // exactly batch*sequence_length*hidden_size elements of the same type, laid out [B, S, H].
  auto source_info = pending_source_->GetTensorTypeAndShapeInfo();
  const size_t source_elements = source_info->GetElementCount();
  auto dst = value_->GetByteSpan();
  const size_t dst_bytes = dst.size();
  const size_t element_size = Ort::SizeOf(type_);
  if (source_elements * element_size != dst_bytes)
    throw std::runtime_error("HiddenStatesInputs::Update size mismatch: source has " +
                             std::to_string(source_elements) + " elements, expected " +
                             std::to_string(dst_bytes / element_size));

  // Prefer a device-to-device copy on the shared compute stream when the source is already on
  // the model's device (e.g. the main model's hidden_states output in an in-engine MTP loop).
  // All CUDA sessions share one stream, so the enqueued D2D copy is correctly ordered after the
  // producer's Run and before this model's Run -- no host round-trip and no host synchronization
  // (see onnxruntime issue #28539 on the async IO-binding pattern). Falls back to a host-staged
  // copy only when the source lives on the CPU.
  const bool source_on_cpu =
      pending_source_->GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_CPU;
  if (!source_on_cpu) {
    auto source_span = ByteWrapTensor(*model_.p_device_, *pending_source_);
    dst.CopyFrom(source_span);
  } else {
    // Source is on the CPU: stage through the CPU-accessible span, then push to device.
    auto dst_cpu = dst.CpuSpan();
    std::memcpy(dst_cpu.data(), pending_source_->GetTensorRawData(), dst_bytes);
    dst.CopyCpuToDevice();
  }
}

HiddenStatesOutputs::HiddenStatesOutputs(State& state)
    : state_{state} {
  const std::string& name = model_.config_->model.decoder.outputs.hidden_states;
  type_ = model_.session_info_.GetOutputDataType(name);
  shape_ = {state_.params_->BatchBeamSize(), 0, model_.config_->model.decoder.hidden_size};
  value_ = std::make_unique<Tensor>(model_.p_device_inputs_, type_);
}

void HiddenStatesOutputs::Add() {
  output_index_ = state_.outputs_.size();
  state_.outputs_.push_back(value_->GetOrtTensor());
  state_.output_names_.push_back(model_.config_->model.decoder.outputs.hidden_states.c_str());
}

void HiddenStatesOutputs::Update(int sequence_length) {
  if (static_cast<int64_t>(sequence_length) != shape_[1]) {
    shape_[1] = sequence_length;
    // Static buffer when graph capture is active so the captured graph can bind to a stable
    // output address. Pre-size to the max captured length (e.g. the 2-token MTP verify shape)
    // so the buffer base address is stable across the 1- and 2-token captured graphs.
    const int max_cap = state_.params_->max_graph_capture_length;
    const bool use_static = state_.params_->use_graph_capture && shape_[1] >= 1 && shape_[1] <= max_cap;
    const size_t static_cap_bytes = use_static ? static_cast<size_t>(shape_[0]) * max_cap * shape_[2] * Ort::SizeOf(type_) : 0;
    value_->CreateTensor(shape_, use_static, static_cap_bytes);
    state_.outputs_[output_index_] = value_->GetOrtTensor();
  }
}

}  // namespace Generators
