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

#pragma warning(push)
#pragma warning(disable : 4065)  // switch statement contains 'default' but no 'case' labels
#pragma warning(disable : 4189)  // local variable is initialized but not referenced
#pragma warning(disable : 4702)  // unreachable code

void ExtraInputs::Add() {
  // Add extra user inputs
  for (int i = 0; i < state_.params_->extra_inputs.size(); ++i) {
    state_.input_names_.push_back(state_.params_->extra_inputs[i].name.c_str());
    state_.inputs_.push_back(extra_inputs_[i]);
  }

  // Copy the data from the CPU-backed ORT value to the static buffers
  for (int i = 0; i < sb_extra_inputs_.size(); ++i) {
    auto type_and_shape_info = extra_inputs_[i]->GetTensorTypeAndShapeInfo();
    auto shape = type_and_shape_info->GetShape();
    auto element_count = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    auto copy_size_in_bytes = element_count * SizeOf(type_and_shape_info->GetElementType());

    switch (model_.device_type_) {
#if USE_DML
      case DeviceType::DML: {
        ComPtr<ID3D12Resource> target_resource;
        Ort::ThrowOnError(model_.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(model_.allocator_device_, extra_inputs_[i]->GetTensorMutableRawData(), &target_resource));

        auto source = std::span(state_.params_->extra_inputs[i].tensor->ort_tensor_->GetTensorData<const uint8_t>(), copy_size_in_bytes);

        model_.GetDmlUploadHeap()->BeginUploadToGpu(
            target_resource.Get(),
            0,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            source);
      } break;
#endif

#if USE_CUDA
      case DeviceType::CUDA: {
        cudaMemcpyAsync(
            extra_inputs_[i]->GetTensorMutableRawData(),
            state_.params_->extra_inputs[i].tensor->ort_tensor_->GetTensorMutableRawData(),
            copy_size_in_bytes,
            cudaMemcpyHostToDevice,
            model_.cuda_stream_);
      } break;
#endif

      default:
        throw std::runtime_error("Unsupported device for graph capture");
    }
  }
}

#pragma warning(pop)

}  // namespace Generators
