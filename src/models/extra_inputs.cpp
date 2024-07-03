// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "model.h"
#include "extra_inputs.h"

namespace Generators {

ExtraInputs::ExtraInputs(const Model& model, State& state) : model_{model}, state_{state} {
  // We take extra inputs from LoraAdapters
  auto& lora_management = model_.GetLoraAdapterManagement();
  lora_management.OutputAdaptersParameters(std::back_inserter(extra_input_names_), std::back_inserter(extra_inputs_));
}

#pragma warning(push)
#pragma warning(disable : 4065)  // switch statement contains 'default' but no 'case' labels
#pragma warning(disable : 4189)  // local variable is initialized but not referenced
#pragma warning(disable : 4702)  // unreachable code

void ExtraInputs::Add() {
  // Add extra user inputs to the state
  for (int i = 0, lim = extra_input_names_.size(); i < lim; ++i) {
    state_.input_names_.push_back(extra_input_names_[i].c_str());
    state_.inputs_.push_back(extra_inputs_[i].get());
  }

  // Copy the data from the CPU-backed ORT value to the static buffers
//  for (int i = 0, lim = device_buffers_.size(); i < lim; ++i) {
//    auto type_and_shape_info = extra_inputs_[i]->GetTensorTypeAndShapeInfo();
//    auto shape = type_and_shape_info->GetShape();
//    const auto element_count = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
//
//    if (element_count > 0) {
//      const auto copy_size_in_bytes = element_count * SizeOf(type_and_shape_info->GetElementType());
//      switch (model_.device_type_) {
//#if USE_DML
//        case DeviceType::DML: {
//          ComPtr<ID3D12Resource> target_resource;
//          Ort::ThrowOnError(model_.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(
//              model_.allocator_device_, extra_inputs_[i]->GetTensorMutableRawData(), &target_resource));
//
//          auto source = std::span(state_.params_->extra_inputs[i].tensor->ort_tensor_->GetTensorData<const uint8_t>(),
//                                  copy_size_in_bytes);
//
//          model_.GetDmlUploadHeap()->BeginUploadToGpu(target_resource.Get(), 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
//                                                      source);
//        } break;
//#endif
//
//#if USE_CUDA
//        case DeviceType::CUDA: {
//          cudaMemcpyAsync(extra_inputs_[i]->GetTensorMutableRawData(),
//                          state_.params_->extra_inputs[i].tensor->ort_tensor_->GetTensorMutableRawData(),
//                          copy_size_in_bytes, cudaMemcpyHostToDevice, model_.cuda_stream_);
//        } break;
//#endif
//
//        default:
//          throw std::runtime_error("Unsupported device for graph capture");
//      }
//    }
//  }
}

#pragma warning(pop)

}  // namespace Generators
