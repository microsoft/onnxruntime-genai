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
}

void HiddenStatesInputs::Add() {
  input_index_ = state_.inputs_.size();
  // The concrete per-length buffer is bound in Update() (called before every Run); start unbound.
  state_.inputs_.push_back(nullptr);
  state_.input_names_.push_back(model_.config_->model.decoder.inputs.hidden_states.c_str());
}

Tensor* HiddenStatesInputs::GetOrCreateBuffer(int sequence_length) {
  const bool capture_length = state_.params_->use_graph_capture &&
                              sequence_length <= state_.params_->max_graph_capture_length;
  if (capture_length) {
    auto it = buffers_by_len_.find(sequence_length);
    if (it != buffers_by_len_.end())
      return it->second.get();
  } else if (dynamic_buffer_ && dynamic_sequence_length_ == sequence_length) {
    return dynamic_buffer_.get();
  }

  auto tensor = std::make_unique<Tensor>(model_.p_device_inputs_, type_);
  const std::array<int64_t, 3> shape{shape_[0], sequence_length, shape_[2]};
  const size_t bytes = static_cast<size_t>(shape[0]) * static_cast<size_t>(shape[1]) *
                       static_cast<size_t>(shape[2]) * Ort::SizeOf(type_);
  tensor->CreateTensor(shape, capture_length, capture_length ? bytes : 0);
  Tensor* raw = tensor.get();
  if (capture_length) {
    buffers_by_len_[sequence_length] = std::move(tensor);
  } else {
    dynamic_buffer_ = std::move(tensor);
    dynamic_sequence_length_ = sequence_length;
  }
  return raw;
}

void HiddenStatesInputs::Update(int sequence_length) {
  if (!pending_source_)
    throw std::runtime_error("HiddenStatesInputs::Update called before SetValue");

  // Bind this step's dedicated per-length device buffer as the model input.
  Tensor* buffer = GetOrCreateBuffer(sequence_length);
  state_.inputs_[input_index_] = buffer->GetOrtTensor();

  // Copy the caller-provided values into THIS length's dedicated device buffer. The source is
  // expected to hold exactly batch*sequence_length*hidden_size elements of the same type ([B,S,H]).
  auto source_info = pending_source_->GetTensorTypeAndShapeInfo();
  if (source_info->GetElementType() != type_)
    throw std::runtime_error("HiddenStatesInputs::Update type mismatch");
  const size_t source_elements = source_info->GetElementCount();
  auto dst = buffer->GetByteSpan();
  const size_t dst_bytes = dst.size();
  const size_t element_size = Ort::SizeOf(type_);
  if (source_elements * element_size != dst_bytes)
    throw std::runtime_error("HiddenStatesInputs::Update size mismatch: source has " +
                             std::to_string(source_elements) + " elements, expected " +
                             std::to_string(dst_bytes / element_size));

  // Prefer a device-to-device copy on the shared compute stream when the source is already on
  // the model's device (e.g. the main model's hidden_states output in an in-engine MTP loop).
  // All CUDA sessions share one stream, so the enqueued async D2D copy into the length-specific
  // buffer is correctly ordered after the producer's Run and before this model's Run / CUDA-graph
  // replay -- no host round-trip and no host synchronization (see onnxruntime issue #28539 on the
  // async IO-binding pattern). Falls back to a host-staged copy only when the source is on the CPU.
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
}

void HiddenStatesOutputs::Add() {
  output_index_ = state_.outputs_.size();
  // The concrete per-length buffer is bound in Update() (called before every Run); start unbound.
  state_.outputs_.push_back(nullptr);
  state_.output_names_.push_back(model_.config_->model.decoder.outputs.hidden_states.c_str());
}

Tensor* HiddenStatesOutputs::GetOrCreateBuffer(int sequence_length) {
  const bool capture_length = state_.params_->use_graph_capture &&
                              sequence_length <= state_.params_->max_graph_capture_length;
  if (capture_length) {
    auto it = buffers_by_len_.find(sequence_length);
    if (it != buffers_by_len_.end())
      return it->second.get();
  } else if (dynamic_buffer_ && dynamic_sequence_length_ == sequence_length) {
    return dynamic_buffer_.get();
  }

  auto tensor = std::make_unique<Tensor>(model_.p_device_inputs_, type_);
  const std::array<int64_t, 3> shape{shape_[0], sequence_length, shape_[2]};
  const size_t bytes = static_cast<size_t>(shape[0]) * static_cast<size_t>(shape[1]) *
                       static_cast<size_t>(shape[2]) * Ort::SizeOf(type_);
  tensor->CreateTensor(shape, capture_length, capture_length ? bytes : 0);
  Tensor* raw = tensor.get();
  if (capture_length) {
    buffers_by_len_[sequence_length] = std::move(tensor);
  } else {
    dynamic_buffer_ = std::move(tensor);
    dynamic_sequence_length_ = sequence_length;
  }
  return raw;
}

void HiddenStatesOutputs::Update(int sequence_length) {
  // Bind this step's dedicated per-length device buffer as the model output.
  Tensor* buffer = GetOrCreateBuffer(sequence_length);
  state_.outputs_[output_index_] = buffer->GetOrtTensor();
}

}  // namespace Generators
