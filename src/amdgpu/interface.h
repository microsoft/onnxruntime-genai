// Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../smartptrs.h"

namespace Generators {

// Note: memory allocated through the AMD GPU interface is both host and device accessible, because
// the AMD APU iGPU this provider targets shares physical memory with the host.
struct AMDGPUInterface : DeviceInterface {
  using ProviderOptions = std::vector<std::pair<std::string, std::string>>;

  std::unique_ptr<OrtMemoryInfo> GetMemoryInfo() const override;

  virtual void SetupProvider(OrtSessionOptions&, const ProviderOptions&) = 0;
};

// Creates a fresh AMD GPU DeviceInterface instance. Ownership is taken by OrtGlobals.
// `env` is the OrtGlobals env this interface belongs to (created before the interface and
// destroyed after it, per the reverse-order teardown).
std::unique_ptr<DeviceInterface> CreateAMDGPUInterface(OrtEnv& env);

}  // namespace Generators
