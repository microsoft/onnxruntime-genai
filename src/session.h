// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "device.h"
#include "onnxruntime_api.h"

#pragma once

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

 private:
  // allocators are per-Session
  std::unique_ptr<Ort::Allocator> device_allocator_[static_cast<int>(DeviceType::MAX)];

  std::unique_ptr<OrtSession> ort_session_;
};
}  // namespace Generators
