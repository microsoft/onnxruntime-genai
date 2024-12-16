#include "../generators.h"
#include "model.h"
#include "extra_inputs.h"
#include "kernels.h"

namespace Generators {

ExtraInputs::ExtraInputs(State& state)
    : state_{state} {
  extra_inputs_.reserve(state_.params_->extra_inputs.size());

  if (state_.GetCapturedGraphInfo()) {
    owned_extra_inputs_.reserve(state_.params_->extra_inputs.size());

    for (int i = 0; i < state_.params_->extra_inputs.size(); ++i) {
      auto type_and_shape_info = state_.params_->extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
      const auto& input_name = state_.params_->extra_inputs[i].name;

      sb_extra_inputs_.emplace(input_name, state_.GetCapturedGraphInfo()->sb_extra_inputs_.at(input_name).get());
      owned_extra_inputs_.push_back(sb_extra_inputs_.at(input_name)->CreateTensorOnStaticBuffer(type_and_shape_info->GetShape(), type_and_shape_info->GetElementType()));
      extra_inputs_.push_back(owned_extra_inputs_.back().get());
    }
  } else {
    // We don't use graph capture, so simply use the existing pointers
    for (auto& extra_input : state_.params_->extra_inputs) {
      extra_inputs_.push_back(extra_input.tensor->ort_tensor_.get());
    }
  }
}

void ExtraInputs::Add() {
  // Add extra user inputs
  for (int i = 0; i < state_.params_->extra_inputs.size(); ++i) {
    state_.input_names_.push_back(state_.params_->extra_inputs[i].name.c_str());
    state_.inputs_.push_back(extra_inputs_[i]);
  }

  // Copy the data from the CPU-backed ORT value to the static buffers
  for (int i = 0; i < sb_extra_inputs_.size(); ++i) {
    auto tensor = ByteWrapTensor(*model_.p_device_, *extra_inputs_[i]);
    auto source = std::span{state_.params_->extra_inputs[i].tensor->ort_tensor_->GetTensorData<uint8_t>(), tensor.size()};
    copy(source, tensor.CpuSpan());
    tensor.CopyCpuToDevice();
  }
}

}  // namespace Generators
