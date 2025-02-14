// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"

namespace Generators {

Adapter::Adapter(const char* adapter_file_path, Ort::Allocator* allocator)
    : adapter_{OrtLoraAdapter::Create(fs::path(adapter_file_path).c_str(), *allocator)} {}

const OrtLoraAdapter* Adapter::AcquireRef() {
  ref_count_++;

  return adapter_.get();
}

void Adapter::ReleaseRef() {
  ref_count_--;
  if (ref_count_ < 0) {
    throw std::runtime_error("Adapter ref count went negative.");
  }
}

int32_t Adapter::RefCount() const {
  return ref_count_;
}

Adapters::Adapters(const Model* model) : model_{model} {}

void Adapters::LoadAdapter(const char* adapter_file_path, const std::string& adapter_name) {
  if (adapters_.find(adapter_name) != adapters_.end()) {
    throw std::runtime_error("Adapter already loaded: " + std::string{adapter_name});
  }

  adapters_.emplace(adapter_name, std::make_unique<Adapter>(adapter_file_path,
                                                            model_->p_device_->GetType() == DeviceType::CUDA
                                                                ? &model_->p_device_->GetAllocator()
                                                                : nullptr));
}

void Adapters::UnloadAdapter(const std::string& adapter_name) {
  auto adapter = adapters_.find(adapter_name);
  if (adapter == adapters_.end()) {
    throw std::runtime_error("Adapter not found: " + std::string{adapter_name});
  }

  if (adapter->second->RefCount() > 0) {
    throw std::runtime_error("Adapter still in use: " + std::string{adapter_name});
  }

  adapters_.erase(adapter);
}

const OrtLoraAdapter* Adapters::AcquireAdapter(const std::string& adapter_name) {
  auto adapter = adapters_.find(adapter_name);
  if (adapter == adapters_.end()) {
    throw std::runtime_error("Adapter not found: " + std::string{adapter_name});
  }

  return adapter->second->AcquireRef();
}

void Adapters::ReleaseAdapter(const std::string& adapter_name) {
  auto adapter = adapters_.find(adapter_name);
  if (adapter == adapters_.end()) {
    throw std::runtime_error("Adapter not found: " + std::string{adapter_name});
  }

  adapter->second->ReleaseRef();
}

}  // namespace Generators
