// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Layer-2 override channel regression suite.
//
// genai_config.json is the layer-1 baseline. Two runtime channels write into
// the same `model.<component>.session_options.*` namespace as overrides:
//   1. RuntimeSettings::GenerateConfigOverlay() — emits JSON for embedded
//      handles such as the WebGPU dawnProcTable (consumed at Model::Create).
//   2. OgaConfigOverlay() / OverlayConfig() — caller-supplied JSON applied
//      to a Config instance (consumed by language bindings: C#, Java,
//      Python via OgaConfig.overlay).
//
// Both feed through the same JSON parser registered for Config, so the contract
// is that any key writable from genai_config.json is also writable via these
// runtime channels. These tests pin that contract so future refactors of the
// overlay merge logic don't silently break it.

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

#include "generators.h"
#include "config.h"
#include "runtime_settings.h"

#include "test_utils.h"

namespace Generators::test {

namespace {

constexpr const char* kTinyGpt2RelativePath = "hf-internal-testing/tiny-random-gpt2-fp32";

std::filesystem::path TinyGpt2Path() {
  return std::filesystem::path(MODEL_PATH) / kTinyGpt2RelativePath;
}

}  // namespace

// --- RuntimeSettings::GenerateConfigOverlay --------------------------------

TEST(RuntimeSettingsTest, GenerateConfigOverlay_NoHandles_ReturnsEmpty) {
  RuntimeSettings settings;
  EXPECT_EQ(settings.GenerateConfigOverlay(), std::string{});
}

TEST(RuntimeSettingsTest, GenerateConfigOverlay_UnrelatedHandle_ReturnsEmpty) {
  // Only "dawnProcTable" is recognised today; any other handle key must be a no-op
  // so callers can stash unrelated state on RuntimeSettings without leaking it
  // into the config overlay.
  RuntimeSettings settings;
  settings.handles_["someUnrelatedKey"] = reinterpret_cast<void*>(static_cast<uintptr_t>(0xDEADBEEFU));
  EXPECT_EQ(settings.GenerateConfigOverlay(), std::string{});
}

TEST(RuntimeSettingsTest, GenerateConfigOverlay_DawnProcTable_TargetsWebGpuProviderOption) {
  RuntimeSettings settings;
  void* fake_handle = reinterpret_cast<void*>(static_cast<uintptr_t>(0x12345678U));
  settings.handles_["dawnProcTable"] = fake_handle;

  const std::string overlay = settings.GenerateConfigOverlay();
  ASSERT_FALSE(overlay.empty());

  // Layer-2 contract: the overlay must target
  // model.decoder.session_options.provider_options[*].WebGPU.dawnProcTable.
  // If any of these path components changes, every WebGPU consumer (Python,
  // C#, Java, ObjC) silently loses its dawn handle plumbing.
  EXPECT_NE(overlay.find("\"model\""), std::string::npos);
  EXPECT_NE(overlay.find("\"decoder\""), std::string::npos);
  EXPECT_NE(overlay.find("\"session_options\""), std::string::npos);
  EXPECT_NE(overlay.find("\"provider_options\""), std::string::npos);
  EXPECT_NE(overlay.find("\"WebGPU\""), std::string::npos);
  EXPECT_NE(overlay.find("\"dawnProcTable\""), std::string::npos);

  // The handle is encoded as a decimal string of the pointer's numeric value.
  const std::string expected_value = std::to_string(reinterpret_cast<size_t>(fake_handle));
  EXPECT_NE(overlay.find(expected_value), std::string::npos);
}

// --- OverlayConfig: caller-supplied overlay --------------------------------

TEST(OverlayConfigTest, SearchTopKIsOverridable) {
  Config config{TinyGpt2Path(), std::string_view{}};
  const int original_top_k = config.search.top_k;

  OverlayConfig(config, R"({"search": {"top_k": 7}})");
  EXPECT_EQ(config.search.top_k, 7);

  // A second overlay must replace, not accumulate, so callers can re-apply
  // settings (e.g. on each request) without state leaking across calls.
  OverlayConfig(config, R"({"search": {"top_k": 13}})");
  EXPECT_EQ(config.search.top_k, 13);

  // Sanity: the original baseline shouldn't have been 13 — otherwise the
  // assertions above are degenerate.
  EXPECT_NE(original_top_k, 13);
}

TEST(OverlayConfigTest, SearchTopKLeavesUnrelatedFieldsAlone) {
  Config config{TinyGpt2Path(), std::string_view{}};
  const float original_temperature = config.search.temperature;
  const float original_top_p = config.search.top_p;

  OverlayConfig(config, R"({"search": {"top_k": 25}})");
  EXPECT_EQ(config.search.top_k, 25);
  EXPECT_FLOAT_EQ(config.search.temperature, original_temperature);
  EXPECT_FLOAT_EQ(config.search.top_p, original_top_p);
}

// --- End-to-end: RuntimeSettings -> OverlayConfig --------------------------

TEST(RuntimeSettingsOverlayTest, DawnProcTableSurvivesParseIntoConfig) {
  Config config{TinyGpt2Path(), std::string_view{}};

  RuntimeSettings settings;
  void* fake_handle = reinterpret_cast<void*>(static_cast<uintptr_t>(0xABCDEF01U));
  settings.handles_["dawnProcTable"] = fake_handle;

  const std::string overlay = settings.GenerateConfigOverlay();
  ASSERT_FALSE(overlay.empty());

  // Apply the runtime overlay through the same channel callers use.
  ASSERT_NO_THROW(OverlayConfig(config, overlay));

  // The overlay should have produced (or merged into) a WebGPU provider_options
  // entry on the decoder session_options.
  const auto& po_list = config.model.decoder.session_options.provider_options;
  const auto webgpu = std::find_if(po_list.begin(), po_list.end(),
                                   [](const Config::ProviderOptions& po) { return po.name == "WebGPU"; });
  ASSERT_NE(webgpu, po_list.end()) << "WebGPU provider_options entry was not created by the overlay";

  const auto dawn = std::find_if(webgpu->options.begin(), webgpu->options.end(),
                                 [](const Config::NamedString& kv) { return kv.first == "dawnProcTable"; });
  ASSERT_NE(dawn, webgpu->options.end()) << "dawnProcTable option was not preserved through the overlay";
  EXPECT_EQ(dawn->second, std::to_string(reinterpret_cast<size_t>(fake_handle)));
}

}  // namespace Generators::test
