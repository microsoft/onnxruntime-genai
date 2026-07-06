// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"

namespace Generators {

Adapter::Adapter(const char* adapter_file_path, Ort::Allocator* allocator)
    : adapter_{OrtLoraAdapter::Create(fs::path(adapter_file_path).c_str(), *allocator)} {}

const OrtLoraAdapter* Adapter::AcquireRef() {
  // Caller (Adapters::AcquireAdapter) holds Adapters::mutex_, which
  // serializes all access to ref_count_.
  ref_count_++;
  return adapter_.get();
}

void Adapter::ReleaseRef() {
  // Caller (Adapters::ReleaseAdapter) holds Adapters::mutex_.
  ref_count_--;
  if (ref_count_ < 0) {
    // Restore invariant so a caller catching the exception doesn't leave the
    // counter in a negative state that would trip later releases too.
    ref_count_++;
    throw std::runtime_error("Adapter ref count went negative.");
  }
}

int32_t Adapter::RefCount() const {
  // Caller (Adapters::UnloadAdapter) holds Adapters::mutex_.
  return ref_count_;
}

Adapters::Adapters(const Model* model) : model_{model} {}

void Adapters::LoadAdapter(const char* adapter_file_path, const std::string& adapter_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (adapters_.find(adapter_name) != adapters_.end()) {
    throw std::runtime_error("Adapter already loaded: " + std::string{adapter_name});
  }

  adapters_.emplace(adapter_name, std::make_unique<Adapter>(adapter_file_path,
                                                            model_->p_device_->GetType() == DeviceType::CUDA
                                                                ? &model_->p_device_->GetAllocator()
                                                                : nullptr));
}

void Adapters::UnloadAdapter(const std::string& adapter_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto adapter = adapters_.find(adapter_name);
  if (adapter == adapters_.end()) {
    throw std::runtime_error("Adapter not found: " + std::string{adapter_name});
  }

  // Check-and-erase must happen atomically with respect to AcquireAdapter /
  // ReleaseAdapter, which also acquire mutex_. This closes the TOCTOU window
  // where another thread could AcquireRef() between the RefCount() check and
  // the erase(), producing a use-after-free.
  if (adapter->second->RefCount() > 0) {
    throw std::runtime_error("Adapter still in use: " + std::string{adapter_name});
  }

  adapters_.erase(adapter);
}

const OrtLoraAdapter* Adapters::AcquireAdapter(const std::string& adapter_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto adapter = adapters_.find(adapter_name);
  if (adapter == adapters_.end()) {
    throw std::runtime_error("Adapter not found: " + std::string{adapter_name});
  }

  return adapter->second->AcquireRef();
}

void Adapters::ReleaseAdapter(const std::string& adapter_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto adapter = adapters_.find(adapter_name);
  if (adapter == adapters_.end()) {
    throw std::runtime_error("Adapter not found: " + std::string{adapter_name});
  }

  adapter->second->ReleaseRef();
}

}  // namespace Generators
