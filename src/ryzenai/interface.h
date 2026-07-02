// Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

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

  static void Shutdown();
};

RyzenAIInterface* GetRyzenAIInterface();

}  // namespace Generators
