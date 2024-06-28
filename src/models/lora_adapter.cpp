// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_api.h"
#include "lora_adapter.h"
#include "../span.h"

namespace Generators {
namespace details {
std::string LoraCacheKey(std::string_view adapter_name, std::string param_name) {
  std::string result;
  result.reserve(adapter_name.size() + param_name.size() + 1U);
  result.append(adapter_name).append(".").append(param_name);
  return result;
}

constexpr std::array<int64_t, 2> empty_2D_shape = {0, 0};

std::shared_ptr<OrtValue> CreateEmptyInput(Ort::Allocator* allocator, ONNXTensorElementDataType type) {
  return OrtValue::CreateTensor(allocator->GetInfo(), nullptr, 0, empty_2D_shape, type);
}

}  // namespace details

void LoraAdapaterManagement::CreateAdapter(const std::string& adapter_name) {
  auto result = adapters_.emplace(adapter_name, details::LoraAdapter{});
  if (!result.second) {
    throw std::runtime_error("Adapter: " + adapter_name + " already exist");
  }
  result.first->second.SetName(adapter_name);
}


void LoraAdapaterManagement::AddParameter(const std::string& adapter_name, std::string param_name, std::shared_ptr<Tensor> p) {
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

void LoraAdapaterManagement::RemoveAdapter(const std::string& adapter_name) {
  auto hit = adapters_.find(adapter_name);
  if (hit == adapters_.end()) {
    throw std::runtime_error("Adapter: " + adapter_name + " does not exist");
  }

  if (hit->second.IsActive()) {
    throw std::runtime_error("Adapter: " + adapter_name + " is active and can not be deleted");
  }

  adapters_.erase(hit);
}

void LoraAdapaterManagement::ActivateAdapter(const std::string& adapter_name) {
  auto hit = adapters_.find(adapter_name);
  if (hit == adapters_.end()) {
    throw std::runtime_error("Adapter: " + adapter_name + " does not exist");
  }

  if (hit->second.IsActive()) {
    throw std::runtime_error("Adapter: " + adapter_name + " is already active");
  }

  hit->second.SetActive();
}

void Generators::LoraAdapaterManagement::DeactiveAdapter(const std::string& adapter_name) {
  auto hit = adapters_.find(adapter_name);
  if (hit == adapters_.end()) {
    throw std::runtime_error("Adapter: " + adapter_name + " does not exist");
  }

  if (!hit->second.IsActive()) {
    throw std::runtime_error("Adapter: " + adapter_name + " is not active");
  }

  hit->second.Deactivate();
}

}  // namespace Generators