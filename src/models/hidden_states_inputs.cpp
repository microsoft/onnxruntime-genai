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

  // Stage through the CPU-accessible span, then push to device (handles host->device for
  // non-CPU EPs; a no-op copy for CPU).
  auto dst_cpu = dst.CpuSpan();
  std::memcpy(dst_cpu.data(), pending_source_->GetTensorRawData(), dst_bytes);
  dst.CopyCpuToDevice();
}

}  // namespace Generators
