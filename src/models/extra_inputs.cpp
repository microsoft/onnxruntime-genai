#include "../generators.h"
#include "model.h"
#include "extra_inputs.h"
#include "kernels.h"

namespace Generators {

ExtraInputs::ExtraInputs(const Model& model, State& state) : model_{model}, state_{state} {
  // We take extra inputs from LoraAdapters
  auto& lora_management = model_.GetLoraAdapterManagement();
  lora_management.OutputAdaptersParameters(std::back_inserter(lora_input_names_), std::back_inserter(lora_tensors_));

  extra_inputs_.reserve(lora_tensors_.size());

  // for devices we create Static buffers and on Add() we place them on device.
  if (model.device_type_ == DeviceType::CUDA || model.device_type_ == DeviceType::DML) {
    for (const auto& tensor : lora_tensors_) {
      auto type_and_shape_info = tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
      const auto element_count = type_and_shape_info->GetElementCount();
      const auto shape = type_and_shape_info->GetShape();

      // Some extra inputs would come from inactive adapters, those would lora_r == 0 in their shape
      if (!shape.empty() && element_count > 0) {
        const auto first_dim = shape[0];
        device_buffers_.emplace_back(model.allocator_device_, first_dim);
        owned_extra_inputs_.push_back(
            device_buffers_.back().CreateTensorOnStaticBuffer(shape, type_and_shape_info->GetElementType()));
      } else {
        // input is empty, create empty OrtValue tensor
        const auto& mem_info = model.allocator_device_->GetInfo();
        static int64_t empty[] = {0}; // should not matter where it points
        auto ort_value = OrtValue::CreateTensor(mem_info, &empty, 0, shape,
                                                type_and_shape_info->GetElementType());
        owned_extra_inputs_.push_back(std::move(ort_value));
      }
      extra_inputs_.push_back(owned_extra_inputs_.back().get());
    }
  } else {
    // For CPU we simply use the returned tensors
    for (const auto& tensor : lora_tensors_) {
      extra_inputs_.push_back(tensor->ort_tensor_.get());
    }
  }

  //if (state_.GetCapturedGraphInfo()) {
  //  owned_extra_inputs_.reserve(state_.params_->extra_inputs.size());

  //  for (int i = 0; i < state_.params_->extra_inputs.size(); ++i) {
  //    auto type_and_shape_info = state_.params_->extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
  //    const auto& input_name = state_.params_->extra_inputs[i].name;

  //    sb_extra_inputs_.emplace(input_name, state_.GetCapturedGraphInfo()->sb_extra_inputs_.at(input_name).get());
  //    owned_extra_inputs_.push_back(
  //        sb_extra_inputs_.at(input_name)
  //            ->CreateTensorOnStaticBuffer(type_and_shape_info->GetShape(), type_and_shape_info->GetElementType()));
  //    extra_inputs_.push_back(owned_extra_inputs_.back().get());
  //  }
  //} else {
  //  // We don't use graph capture, so simply use the existing pointers
  //  for (auto& extra_input : state_.params_->extra_inputs) {
  //    extra_inputs_.push_back(extra_input.tensor->ort_tensor_.get());
  //  }
  //}
}

#pragma warning(push)
#pragma warning(disable : 4065)  // switch statement contains 'default' but no 'case' labels
#pragma warning(disable : 4189)  // local variable is initialized but not referenced
#pragma warning(disable : 4702)  // unreachable code

void ExtraInputs::Add() {
  // Add extra user inputs to the state
  for (int i = 0, lim = lora_input_names_.size(); i < lim; ++i) {
    state_.input_names_.push_back(lora_input_names_[i]);
    state_.inputs_.push_back(extra_inputs_[i]);
  }

  // Copy the data from the CPU-backed ORT value to the static buffers
  for (int i = 0, lim = device_buffers_.size(); i < lim; ++i) {
    auto type_and_shape_info = extra_inputs_[i]->GetTensorTypeAndShapeInfo();
    auto shape = type_and_shape_info->GetShape();
    const auto element_count = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());

    if (element_count > 0) {
      const auto copy_size_in_bytes = element_count * SizeOf(type_and_shape_info->GetElementType());
      switch (model_.device_type_) {
#if USE_DML
        case DeviceType::DML: {
          ComPtr<ID3D12Resource> target_resource;
          Ort::ThrowOnError(model_.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(
              model_.allocator_device_, extra_inputs_[i]->GetTensorMutableRawData(), &target_resource));

          auto source = std::span(state_.params_->extra_inputs[i].tensor->ort_tensor_->GetTensorData<const uint8_t>(),
                                  copy_size_in_bytes);

          model_.GetDmlUploadHeap()->BeginUploadToGpu(target_resource.Get(), 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                      source);
        } break;
#endif

#if USE_CUDA
        case DeviceType::CUDA: {
          cudaMemcpyAsync(extra_inputs_[i]->GetTensorMutableRawData(),
                          state_.params_->extra_inputs[i].tensor->ort_tensor_->GetTensorMutableRawData(),
                          copy_size_in_bytes, cudaMemcpyHostToDevice, model_.cuda_stream_);
        } break;
#endif

        default:
          throw std::runtime_error("Unsupported device for graph capture");
      }
    }
  }
}

#pragma warning(pop)

}  // namespace Generators
