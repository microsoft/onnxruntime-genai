// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_api.h"
#include "lora_adapter.h"
#include "model.h"
#include "../generators.h"
#include "../span.h"
#include "utils.h"

namespace Generators {
namespace {

uint64_t empty_input_buf[] = {0xdeadbeefbeefdead};
}  // namespace

namespace details {

std::shared_ptr<OrtValue> CreateEmptyInput(const OrtValue& original) {
  auto type_and_shape = original.GetTensorTypeAndShapeInfo();
  auto shape = type_and_shape->GetShape();

  // Modify shape
  const auto num_dims = shape.size();
  if (num_dims < 2) {
    throw std::runtime_error("Shape must have at least 2 dimensions");
  }

  // Zero out lora_r dim
  const size_t last_dim = shape[num_dims - 1];
  const size_t penal_dim = shape[num_dims - 2];
  if (last_dim < penal_dim) {
    shape[num_dims - 1] = 0;
  } else {
    shape[num_dims - 2] = 0;
  }

  const auto& mem_info = original.GetTensorMemoryInfo();
  return OrtValue::CreateTensor(mem_info, &empty_input_buf, 0, shape, type_and_shape->GetElementType());
}

LoraParam::LoraParam(std::string name, const std::shared_ptr<Tensor>& parameter) : name_(std::move(name)) {
  // Create a duplicate of the ort_value over the same user supplied buffer
  // we want ort_value to be owned by a shared_ptr so it can be shared
  // We could still the unique_ptr from the original tensor, but that would not
  // be a good practice and the internal ORT OrtValue copy constructor is not public.
  ort_user_supplied_value_ = DuplicateOrtValue(*parameter->ort_tensor_);
}

void LoraAdapter::SetActive(const Model* model) {
  std::unique_lock lock(mutex_);
  // Make sure data is copied to devices as needed
  if (parameters_.empty()) {
    throw std::runtime_error("Adapter: " + name_ + " has no parameters");
  }

  // Check if the target device is not CPU
  if (model != nullptr && model->device_type_ != DeviceType::CPU) {
    for (auto& param : parameters_) {
      // XXX: Adjust for caching when implemented
      if (!param.ort_device_value_) {
        // Check if the user has already supplied his buffers on the target device
        auto& source_value = param.ort_user_supplied_value_;
        const auto& mem_info = source_value->GetTensorMemoryInfo();
        auto src_device_type = mem_info.GetDeviceType();

        if ((model->device_type_ == DeviceType::CUDA &&
             src_device_type == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU)) {
            // Re-use what user has supplied on GPU
            param.ort_device_value_ = param.ort_user_supplied_value_;
        } else if(src_device_type != OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU) {
          // XXX: Can the user supply buffers already on DML?
          throw std::runtime_error("Loara parameter buffers are on unsupported device: " +
                                   std::to_string(static_cast<int>(model->device_type_)));
        } else {
          param.ort_device_value_ = CopyToDevice(*source_value, *model);
        }
      }
    }
  }
  active_ = true;
}

}  // namespace details

LoraAdapterManagement::LoraAdapterManagement(const Model* model) : model_(model) {}

void LoraAdapterManagement::CreateAdapter(const std::string& adapter_name) {
  std::unique_lock lock(mutex_);
  auto hit = adapters_.find(adapter_name);
  if (hit != adapters_.end()) {
    throw std::runtime_error("Adapter: " + adapter_name + " already exist");
  }
  auto& adapter = adapters_[adapter_name];
  adapter.SetName(adapter_name);
}

void LoraAdapterManagement::AddParameter(const std::string& adapter_name, std::string param_name,
                                         const std::shared_ptr<Tensor>& tensor) {
  std::shared_lock lock(mutex_);
  auto hit = adapters_.find(adapter_name);
  if (hit == adapters_.end()) {
    throw std::runtime_error("Adapter: " + adapter_name + " does not exist");
  }

  auto& adapter = hit->second;

  if (adapter.IsActive()) {
    throw std::runtime_error("Adapter: " + adapter_name + " is active can not add parameters");
  }

  adapter.AddParameter(std::move(param_name), tensor);
}

void LoraAdapterManagement::RemoveAdapter(const std::string& adapter_name) {
  std::unique_lock lock(mutex_);
  auto hit = adapters_.find(adapter_name);
  if (hit == adapters_.end()) {
    throw std::runtime_error("Adapter: " + adapter_name + " does not exist");
  }

  if (hit->second.IsActive()) {
    throw std::runtime_error("Adapter: " + adapter_name + " is active and can not be deleted");
  }

  adapters_.erase(hit);
}

void LoraAdapterManagement::ActivateAdapters(std::span<const std::string> adapter_names) {
  std::shared_lock lock(mutex_);
  for (const auto& adapter_name : adapter_names) {
    auto hit = adapters_.find(adapter_name);
    if (hit == adapters_.end()) {
      throw std::runtime_error("Adapter: " + adapter_name + " does not exist");
    }
    if (hit->second.IsActive()) {
      throw std::runtime_error("Adapter: " + adapter_name + " is already active");
    }
  }

  for (const auto& adapter_name : adapter_names) {
    auto& adapter = adapters_[adapter_name];
    adapter.SetActive(model_);
  }
}

void LoraAdapterManagement::DeactivateAdapters(std::span<const std::string> adapter_names) {
  std::shared_lock lock(mutex_);
  for (const auto& adapter_name : adapter_names) {
    auto hit = adapters_.find(adapter_name);
    if (hit == adapters_.end()) {
      throw std::runtime_error("Adapter: " + adapter_name + " does not exist");
    }

    if (hit->second.IsActive()) {
      hit->second.Deactivate();
    }
  }
}

void LoraAdapterManagement::DeactiveAllAdapters() {
  std::shared_lock lock(mutex_);
  // Deactivate all adapters that are active
  for (auto& [_, adapter] : adapters_) {
    if (adapter.IsActive()) {
      adapter.Deactivate();
    }
  }
}

std::vector<const char*> LoraAdapterManagement::GetActiveAdapterNames() const {
  std::shared_lock lock(mutex_);
  std::vector<const char*> names;
  names.reserve(adapters_.size());
  for (const auto& [name, adapter] : adapters_) {
    if (adapter.IsActive()) {
      names.push_back(name.c_str());
    }
  }
  return names;
}

}  // namespace Generators