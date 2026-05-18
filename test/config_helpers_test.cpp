// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for free helpers declared in src/config.h that are used by every
// non-decoder session-creation path. These helpers are part of the public
// internal C++ surface for v4 model-package preparation: the v4 path will
// extend them to merge layer-1 (variant.json) baselines on top of the
// today-only layer-2 (genai_config) view, and the call sites must keep
// working unchanged.

#include <gtest/gtest.h>

#include <filesystem>
#include <optional>

#include "generators.h"
#include "config.h"
#include "models/session_options.h"

#include "test_utils.h"

namespace Generators::test {

namespace {

constexpr const char* kTinyGpt2RelativePath = "hf-internal-testing/tiny-random-gpt2-fp32";

fs::path TinyGpt2Path() {
  return fs::path((std::filesystem::path(MODEL_PATH) / kTinyGpt2RelativePath).string());
}

}  // namespace

// --- EffectiveSessionOptions ----------------------------------------------

TEST(EffectiveSessionOptionsTest, FallsBackToDecoderWhenComponentIsNullopt) {
  Config config{TinyGpt2Path(), std::string_view{}};
  std::optional<Config::SessionOptions> empty;

  const Config::SessionOptions& result = EffectiveSessionOptions(config, empty);

  // Must alias the decoder's session_options storage, not a copy: the address
  // identity is what lets call sites pass the result straight through to
  // CreateSessionOptionsFromConfig without lifetime concerns.
  EXPECT_EQ(&result, &config.model.decoder.session_options);
}

TEST(EffectiveSessionOptionsTest, ReturnsComponentWhenSet) {
  Config config{TinyGpt2Path(), std::string_view{}};
  std::optional<Config::SessionOptions> component;
  component.emplace();
  component->log_id = "vision-session";

  const Config::SessionOptions& result = EffectiveSessionOptions(config, component);

  // Reference must alias the optional's stored value, not the decoder, and
  // not a copy of either.
  EXPECT_EQ(&result, &component.value());
  EXPECT_NE(&result, &config.model.decoder.session_options);
  ASSERT_TRUE(result.log_id.has_value());
  EXPECT_EQ(result.log_id.value(), "vision-session");
}

TEST(EffectiveSessionOptionsTest, FallbackPicksUpLiveDecoderState) {
  // The fallback returns a reference into the live Config, so subsequent
  // mutations of the decoder block must be observable through the returned
  // reference. The five v4-prep call sites in marian / whisper / multi_modal
  // rely on this — they read the result immediately after the call, but the
  // reference semantics are part of the contract so future refactors don't
  // accidentally start copying.
  Config config{TinyGpt2Path(), std::string_view{}};
  std::optional<Config::SessionOptions> empty;

  const Config::SessionOptions& result = EffectiveSessionOptions(config, empty);

  config.model.decoder.session_options.log_id = "decoder-after-call";
  ASSERT_TRUE(result.log_id.has_value());
  EXPECT_EQ(result.log_id.value(), "decoder-after-call");
}

// --- EnsurePackageProvider ------------------------------------------------

TEST(EnsurePackageProviderTest, MapsCanonicalEpToInternalTag) {
  EXPECT_EQ(EpNameToProviderTag("CUDAExecutionProvider"), "cuda");
  EXPECT_EQ(EpNameToProviderTag("DmlExecutionProvider"), "DML");
  EXPECT_EQ(EpNameToProviderTag("NvTensorRtRtxExecutionProvider"), "NvTensorRtRtx");
  EXPECT_EQ(EpNameToProviderTag("OpenVINOExecutionProvider"), "OpenVINO");
  EXPECT_EQ(EpNameToProviderTag("QNNExecutionProvider"), "QNN");
  EXPECT_EQ(EpNameToProviderTag("RyzenAIExecutionProvider"), "RyzenAI");
  EXPECT_EQ(EpNameToProviderTag("VitisAIExecutionProvider"), "VitisAI");
  EXPECT_EQ(EpNameToProviderTag("WebGpuExecutionProvider"), "WebGPU");
}

TEST(EnsurePackageProviderTest, CpuAndUnknownAreEmptyTag) {
  // CPU is the implicit fallback path and never appears in `providers`.
  EXPECT_EQ(EpNameToProviderTag("CPUExecutionProvider"), "");
  // Unrecognised EPs surface as empty so callers no-op rather than
  // injecting a tag the dispatch table can't honour.
  EXPECT_EQ(EpNameToProviderTag("MadeUpExecutionProvider"), "");
  EXPECT_EQ(EpNameToProviderTag(""), "");
}

TEST(EnsurePackageProviderTest, InsertsEpAtFrontOfEmptyProviderList) {
  Config::SessionOptions so;

  EnsurePackageProvider(so, "CUDAExecutionProvider");

  ASSERT_EQ(so.providers.size(), 1u);
  EXPECT_EQ(so.providers.front(), "cuda");
  ASSERT_EQ(so.provider_options.size(), 1u);
  EXPECT_EQ(so.provider_options.front().name, "cuda");
}

TEST(EnsurePackageProviderTest, RotatesEpToFrontWhenAlreadyPresent) {
  // User-overlay added "DML" before us; package mode says "cuda" wins.
  // The package EP must rotate to the front; user's "DML" entry stays in
  // the list at a later position so layer-2 SO/PO overrides for it
  // remain visible (EnsurePackageProvider semantics).
  Config::SessionOptions so;
  so.providers = {"DML", "cuda"};
  Config::ProviderOptions dml;
  dml.name = "DML";
  Config::ProviderOptions cuda;
  cuda.name = "cuda";
  so.provider_options = {dml, cuda};

  EnsurePackageProvider(so, "CUDAExecutionProvider");

  ASSERT_EQ(so.providers.size(), 2u);
  EXPECT_EQ(so.providers[0], "cuda");
  EXPECT_EQ(so.providers[1], "DML");
  // ProviderOptions must NOT be duplicated — the existing "cuda" entry is reused.
  EXPECT_EQ(so.provider_options.size(), 2u);
}

TEST(EnsurePackageProviderTest, IsIdempotent) {
  Config::SessionOptions so;

  EnsurePackageProvider(so, "DmlExecutionProvider");
  EnsurePackageProvider(so, "DmlExecutionProvider");

  ASSERT_EQ(so.providers.size(), 1u);
  EXPECT_EQ(so.providers.front(), "DML");
  ASSERT_EQ(so.provider_options.size(), 1u);
  EXPECT_EQ(so.provider_options.front().name, "DML");
}

TEST(EnsurePackageProviderTest, CpuIsNoOp) {
  Config::SessionOptions so;
  so.providers = {"cuda"};
  Config::ProviderOptions cuda;
  cuda.name = "cuda";
  so.provider_options = {cuda};

  EnsurePackageProvider(so, "CPUExecutionProvider");

  // CPU is the implicit fallback; it must not push itself into providers
  // and must not disturb existing entries.
  ASSERT_EQ(so.providers.size(), 1u);
  EXPECT_EQ(so.providers.front(), "cuda");
  EXPECT_EQ(so.provider_options.size(), 1u);
}

TEST(EnsurePackageProviderTest, UnknownEpIsNoOp) {
  Config::SessionOptions so;

  EnsurePackageProvider(so, "MadeUpExecutionProvider");

  EXPECT_TRUE(so.providers.empty());
  EXPECT_TRUE(so.provider_options.empty());
}

TEST(EnsurePackageProviderTest, AddsMissingProviderOptionsForExistingProviderEntry) {
  // User-supplied genai_config carries `providers: ["cuda"]` but no matching
  // `provider_options[].name == "cuda"` entry. SetProviderSessionOptions
  // would throw "Provider options not found" in that state. Helper must
  // backfill an empty entry so package mode stays robust against partial
  // overlays.
  Config::SessionOptions so;
  so.providers = {"cuda"};

  EnsurePackageProvider(so, "CUDAExecutionProvider");

  ASSERT_EQ(so.provider_options.size(), 1u);
  EXPECT_EQ(so.provider_options.front().name, "cuda");
  EXPECT_TRUE(so.provider_options.front().options.empty());
}

}  // namespace Generators::test
