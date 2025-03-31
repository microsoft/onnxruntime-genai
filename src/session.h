// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "generators.h"
#include "device.h"
#include "models/onnxruntime_api.h"

namespace Generators {

class Session {
 public:
  static std::unique_ptr<Session> Create(const ORTCHAR_T* model_path, const OrtSessionOptions* options) {
    return std::make_unique<Session>(GetOrtGlobals().Env(), model_path, options);
  }

  Session(OrtEnv& env, const ORTCHAR_T* model_path, const OrtSessionOptions* options)
      : ort_session_(OrtSession::Create(env, model_path, options)) {
  }

  Ort::Allocator& CreateAllocator(DeviceType type);

  OrtSession& GetOrtSession() {
    return *ort_session_;
  }
  const OrtSession& GetOrtSession() const {
    return *ort_session_;
  }

  Ort::Allocator& GetCpuAllocator() {
    return allocator_cpu_;
  }

  DeviceInterface& GetDevice() {
    if (!p_device_) {
      throw std::runtime_error("Device has not been initialized");
    }
    return *p_device_;
  }

  DeviceInterface& GetInputsDevice() {
    if (!p_device_inputs_) {
      throw std::runtime_error("Inputs device has not been initialized");
    }
    return *p_device_inputs_;
  }

  DeviceInterface& GetKVCacheDevice() {
    if (!p_device_kvcache_) {
      throw std::runtime_error("KVCache device has not been initialized");
    }
    return *p_device_kvcache_;
  }

 private:
  void InitDeviceAllocator(Session& session);

  // allocators are per-Session (technically per-EP instance but ).
  // The allocator implementation is free to remain valid across sessions but that is an implementation detail
  // that we can't rely on. e.g. WebGPU does not do this.
  std::unique_ptr<Ort::Allocator> device_allocator_[static_cast<int>(DeviceType::MAX)];

  std::unique_ptr<OrtSession> ort_session_;

  // The device we're running on (matches device_type_) used for things that work the same on all devices
  DeviceInterface* p_device_{};
  // For some model inputs, the device might be the CPU device (all but KV cache currently for WebGPU and DML)
  DeviceInterface* p_device_inputs_{};
  // The kvcache is always allocated in device memory  (TODO: Remove in favor of just p_device_?)
  DeviceInterface* p_device_kvcache_{};

  Ort::Allocator& allocator_cpu_{GetDeviceInterface(DeviceType::CPU)->GetAllocator()};
};
}  // namespace Generators
