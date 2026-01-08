#pragma once

namespace Generators {

// Note: memory allocated through RyzenAI interface is host/cpu accessible
struct RyzenAIInterface : DeviceInterface {
  using ProviderOptions = std::vector<std::pair<std::string, std::string>>;

  virtual void SetupProvider(OrtSessionOptions&, const ProviderOptions&) = 0;

  static void Shutdown();
};

RyzenAIInterface* GetRyzenAIInterface();

}  // namespace Generators
