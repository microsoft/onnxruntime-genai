// Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace Generators {

// Note: memory allocated through RyzenAI interface is host/cpu accessible
struct RyzenAIInterface : DeviceInterface {
  using ProviderOptions = std::vector<std::pair<std::string, std::string>>;

  virtual void SetupProvider(OrtSessionOptions&, const ProviderOptions&) = 0;
};

// Creates a fresh RyzenAI DeviceInterface instance. Ownership is taken by OrtGlobals.
// `env` is the OrtGlobals env this interface belongs to (created before the interface and
// destroyed after it, per the reverse-order teardown).
std::unique_ptr<DeviceInterface> CreateRyzenAIInterface(OrtEnv& env);

}  // namespace Generators
