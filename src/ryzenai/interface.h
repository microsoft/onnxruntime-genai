// Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Generators {

// Note: memory allocated through RyzenAI interface is host/cpu accessible
struct RyzenAIInterface : DeviceInterface {
  using ProviderOptions = std::vector<std::pair<std::string, std::string>>;

  std::unique_ptr<OrtMemoryInfo> GetMemoryInfo() const override {
    return OrtMemoryInfo::Create("Cpu",
                                 OrtAllocatorType::OrtDeviceAllocator,
                                 0,
                                 OrtMemType::OrtMemTypeDefault);
  }

  virtual void SetupProvider(OrtSessionOptions&, const ProviderOptions&) = 0;
};

// Creates a fresh RyzenAI DeviceInterface instance. Ownership is taken by OrtGlobals.
// `env` is the OrtGlobals env this interface belongs to (created before the interface and
// destroyed after it, per the reverse-order teardown).
std::unique_ptr<DeviceInterface> CreateRyzenAIInterface(OrtEnv& env);

}  // namespace Generators
