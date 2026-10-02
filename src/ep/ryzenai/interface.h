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

  // RyzenAI buffers are host memory under this CPU memory info, so they would qualify for
  // IsHostAccessible(). It stays false until that is verified on RyzenAI hardware: a RyzenAI
  // decoder's inputs don't need it, since each buffer is its own host mirror, and only a RyzenAI
  // encoder feeding a CPU embedding still stages its features through a copy.
  std::unique_ptr<OrtMemoryInfo> GetMemoryInfo() const override {
    return OrtMemoryInfo::Create("Cpu",
                                 OrtAllocatorType::OrtDeviceAllocator,
                                 0,
                                 OrtMemType::OrtMemTypeDefault);
  }

  std::string GetExecutionProviderName() const override { return "RyzenAI"; }

  virtual void SetupProvider(OrtSessionOptions&, const ProviderOptions&) = 0;
};

// Creates a fresh RyzenAI DeviceInterface instance. Ownership is taken by OrtGlobals.
// `env` is the OrtGlobals env this interface belongs to (created before the interface and
// destroyed after it, per the reverse-order teardown).
std::unique_ptr<DeviceInterface> CreateRyzenAIInterface(OrtEnv& env);

}  // namespace Generators
