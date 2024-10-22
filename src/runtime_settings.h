#pragma once

#include <string>
#include <memory>
#include <unordered_map>

namespace Generators {

// This struct should only be used for runtime settings that are not able to be put into config.
struct RuntimeSettings {
  RuntimeSettings() = default;

  std::string GenerateConfigOverlay() const;

  std::unordered_map<std::string, void*> handles_;
};

std::unique_ptr<RuntimeSettings> CreateRuntimeSettings();

}  // namespace Generators