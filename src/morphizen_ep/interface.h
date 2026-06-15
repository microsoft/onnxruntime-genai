// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "../smartptrs.h"  // DeviceInterface

namespace Generators {

// Memory allocated through the MorphiZen EP interface is treated as both
// host- and device-accessible because the underlying AMD APU iGPU shares
// physical memory with the host. Mirrors the RyzenAIInterface pattern; the
// difference is that this provider targets OrtHardwareDeviceType_GPU + AMD
// GPU vendor id (0x1002) instead of the NPU.
struct MorphiZenEPInterface : DeviceInterface {
  using ProviderOptions = std::vector<std::pair<std::string, std::string>>;

  virtual void SetupProvider(OrtSessionOptions&, const ProviderOptions&) = 0;

  static void Shutdown();
};

MorphiZenEPInterface* GetMorphiZenEPInterface();

}  // namespace Generators
