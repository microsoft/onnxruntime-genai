#pragma once

#include <string>
#include <memory>
#include <unordered_map>

namespace Generators {

// Holder for runtime values that cannot be expressed in genai_config.json
// (e.g. live host-process pointers such as the WebGPU dawnProcTable). Recognised
// handles are projected into the layer-2 config-overlay channel by
// GenerateConfigOverlay() and consumed by Model::Create.
struct RuntimeSettings {
  RuntimeSettings() = default;

  // Layer-2 override channel (handle-driven).
  //
  // Returns a JSON document targeting `model.<component>.session_options.*` for
  // each recognised handle in handles_. Today only "dawnProcTable" is handled,
  // and it is projected to:
  //   model.decoder.session_options.provider_options[*].WebGPU.dawnProcTable
  //
  // Returns an empty string if no recognised handles are set, so callers can
  // unconditionally feed the result into the overlay merge without guarding.
  // Any unrelated handle key is silently ignored to keep the surface small;
  // adding a new recognised handle should only ever add a new branch here.
  //
  // The contract that the produced JSON must remain parsable by ParseConfig /
  // OverlayConfig is pinned by tests in test/runtime_settings_test.cpp.
  std::string GenerateConfigOverlay() const;

  std::unordered_map<std::string, void*> handles_;
};

std::unique_ptr<RuntimeSettings> CreateRuntimeSettings();

}  // namespace Generators