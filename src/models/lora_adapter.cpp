// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_api.h"
#include "lora_adapter.h"
#include "../span.h"

namespace Generators {
namespace {

//std::string LoraCacheKey(std::string_view adapter_name, std::string param_name) {
//  std::string result;
//  result.reserve(adapter_name.size() + param_name.size() + 1U);
//  result.append(adapter_name).append(".").append(param_name);
//  return result;
//}

std::shared_ptr<OrtValue> CreateEmptyInput(Ort::Allocator* allocator, std::span<const int64_t> shape,
                                           ONNXTensorElementDataType type) {
  return OrtValue::CreateTensor(allocator->GetInfo(), nullptr, 0, shape, type);
}

constexpr double empty_input_buf[] = {0};
}

LoraAdapterManagement::LoraAdapterManagement() = default;

std::shared_ptr<Tensor> Generators::LoraAdapterManagement::CreateEmptyInput(const Tensor& tensor) {
  const auto& mem_info = tensor.ort_tensor_->GetTensorMemoryInfo();
  auto type_and_shape = tensor.ort_tensor_->GetTensorTypeAndShapeInfo();
  auto shape = type_and_shape->GetShape();

  auto ort_value =  OrtValue::CreateTensor(mem_info, const_cast<double*>(empty_input_buf), 0,
    shape,
    type_and_shape->GetElementType());

  auto result = std::make_shared<Generators::Tensor>();
  result->ort_tensor_ = std::move(ort_value);
  return result;
}

void LoraAdapterManagement::CreateAdapter(const std::string& adapter_name) {
  auto result = adapters_.emplace(adapter_name, details::LoraAdapter{});
  if (!result.second) {
    throw std::runtime_error("Adapter: " + adapter_name + " already exist");
  }
  result.first->second.SetName(adapter_name);
}

void LoraAdapterManagement::AddParameter(const std::string& adapter_name, std::string param_name,
                                          std::shared_ptr<Tensor> p) {
  auto hit = adapters_.find(adapter_name);
  if (hit == adapters_.end()) {
    throw std::runtime_error("Adapter: " + adapter_name + " does not exist");
  }

  auto& adapter = hit->second;

  if (adapter.IsActive()) {
    throw std::runtime_error("Adapter: " + adapter_name + " is active can not add parameters");
  }

  adapter.AddParameter(std::move(param_name), std::move(p));
}

void LoraAdapterManagement::RemoveAdapter(const std::string& adapter_name) {
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
    adapter.SetActive();
  }
}

void LoraAdapterManagement::DeactiveAllAdapters() {
  // Deactive all adapaters that are active
  for (auto& [name, adapter] : adapters_) {
    if (adapter.IsActive()) {
      adapter.Deactivate();
    }
  }
}

std::vector<std::string_view> LoraAdapterManagement::GetActiveAdapterNames() const {
  std::vector<std::string_view> names;
  names.reserve(adapters_.size());
  for (const auto& [name, adapter] : adapters_) {
    if (adapter.IsActive()) {
      names.push_back(name);
    }
  }
  return names;
}

}  // namespace Generators