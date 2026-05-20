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
#include "models/model_package.h"
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

// --- ApplyVariantFileDefaults --------------------------------------------

TEST(ApplyVariantFileDefaultsTest, FillsUnsetTypedFields) {
  // variant.json carries layer-1 defaults; layer-2 (genai_config) values
  // win where present, but unset typed fields should pick up the defaults
  // verbatim.
  Config::SessionOptions so;
  VariantFile vf;
  vf.session_options = {
      {"intra_op_num_threads", "4"},
      {"inter_op_num_threads", "2"},
      {"enable_cpu_mem_arena", "false"},
      {"enable_mem_pattern", "true"},
      {"log_id", "variant-log"},
      {"log_severity_level", "3"},
      {"enable_profiling", "/tmp/prof"},
      {"graph_optimization_level", "ORT_ENABLE_EXTENDED"},
  };

  ApplyVariantFileDefaults(so, vf, "");

  ASSERT_TRUE(so.intra_op_num_threads.has_value());
  EXPECT_EQ(*so.intra_op_num_threads, 4);
  ASSERT_TRUE(so.inter_op_num_threads.has_value());
  EXPECT_EQ(*so.inter_op_num_threads, 2);
  ASSERT_TRUE(so.enable_cpu_mem_arena.has_value());
  EXPECT_FALSE(*so.enable_cpu_mem_arena);
  ASSERT_TRUE(so.enable_mem_pattern.has_value());
  EXPECT_TRUE(*so.enable_mem_pattern);
  ASSERT_TRUE(so.log_id.has_value());
  EXPECT_EQ(*so.log_id, "variant-log");
  ASSERT_TRUE(so.log_severity_level.has_value());
  EXPECT_EQ(*so.log_severity_level, 3);
  ASSERT_TRUE(so.enable_profiling.has_value());
  EXPECT_EQ(*so.enable_profiling, "/tmp/prof");
  ASSERT_TRUE(so.graph_optimization_level.has_value());
  EXPECT_EQ(*so.graph_optimization_level, ORT_ENABLE_EXTENDED);
}

TEST(ApplyVariantFileDefaultsTest, ExistingValuesWin) {
  // Layer-2 (genai_config) precedes the variant overlay. Anything already
  // set on `session_options` must NOT be overwritten.
  Config::SessionOptions so;
  so.intra_op_num_threads = 8;
  so.log_id = "from-genai-config";
  so.enable_cpu_mem_arena = true;

  VariantFile vf;
  vf.session_options = {
      {"intra_op_num_threads", "4"},
      {"log_id", "from-variant"},
      {"enable_cpu_mem_arena", "false"},
      // Unset on `so`, should be filled.
      {"inter_op_num_threads", "2"},
  };

  ApplyVariantFileDefaults(so, vf, "");

  EXPECT_EQ(*so.intra_op_num_threads, 8);
  EXPECT_EQ(*so.log_id, "from-genai-config");
  EXPECT_TRUE(*so.enable_cpu_mem_arena);
  ASSERT_TRUE(so.inter_op_num_threads.has_value());
  EXPECT_EQ(*so.inter_op_num_threads, 2);
}

TEST(ApplyVariantFileDefaultsTest, UnknownKeysGoToConfigEntries) {
  // Anything we don't recognise as a typed field becomes a free-form
  // `session_config_entry`, but only when the same key isn't already set.
  Config::SessionOptions so;
  so.config_entries.emplace_back("session.use_xnnpack", "0");

  VariantFile vf;
  vf.session_options = {
      {"session.use_xnnpack", "1"},         // existing key -> ignored
      {"session.disable_prepacking", "1"},  // new key -> appended
  };

  ApplyVariantFileDefaults(so, vf, "");

  ASSERT_EQ(so.config_entries.size(), 2u);
  EXPECT_EQ(so.config_entries[0].first, "session.use_xnnpack");
  EXPECT_EQ(so.config_entries[0].second, "0");
  EXPECT_EQ(so.config_entries[1].first, "session.disable_prepacking");
  EXPECT_EQ(so.config_entries[1].second, "1");
}

TEST(ApplyVariantFileDefaultsTest, MergesProviderOptionsIntoMatchingEntry) {
  // provider_options are merged into the entry whose `name` matches the
  // GenAI tag derived from the package's selected EP. Existing keys win,
  // new keys are appended.
  Config::SessionOptions so;
  so.providers = {"cuda"};
  Config::ProviderOptions cuda_po;
  cuda_po.name = "cuda";
  cuda_po.options.emplace_back("device_id", "0");
  so.provider_options.push_back(std::move(cuda_po));

  VariantFile vf;
  vf.provider_options = {
      {"device_id", "1"},                             // existing -> kept as 0
      {"arena_extend_strategy", "kSameAsRequested"},  // new -> appended
  };

  ApplyVariantFileDefaults(so, vf, "CUDAExecutionProvider");

  ASSERT_EQ(so.provider_options.size(), 1u);
  const auto& po = so.provider_options.front();
  ASSERT_EQ(po.options.size(), 2u);
  EXPECT_EQ(po.options[0].first, "device_id");
  EXPECT_EQ(po.options[0].second, "0");
  EXPECT_EQ(po.options[1].first, "arena_extend_strategy");
  EXPECT_EQ(po.options[1].second, "kSameAsRequested");
}

TEST(ApplyVariantFileDefaultsTest, ProviderOptionsSkippedWhenEpIsEmpty) {
  // `run_on_cpu` stages call with an empty EP so variant provider_options
  // don't get merged into a non-CPU ProviderOptions entry that doesn't
  // exist.
  Config::SessionOptions so;
  Config::ProviderOptions cuda_po;
  cuda_po.name = "cuda";
  so.provider_options.push_back(std::move(cuda_po));

  VariantFile vf;
  vf.provider_options = {{"device_id", "1"}};

  ApplyVariantFileDefaults(so, vf, "");

  ASSERT_EQ(so.provider_options.size(), 1u);
  EXPECT_TRUE(so.provider_options.front().options.empty());
}

TEST(ApplyVariantFileDefaultsTest, ProviderOptionsSkippedWhenEpIsCpu) {
  // CPU is the implicit fallback path and never carries provider_options;
  // skip the merge for "CPUExecutionProvider" as well.
  Config::SessionOptions so;
  Config::ProviderOptions cuda_po;
  cuda_po.name = "cuda";
  so.provider_options.push_back(std::move(cuda_po));

  VariantFile vf;
  vf.provider_options = {{"device_id", "1"}};

  ApplyVariantFileDefaults(so, vf, "CPUExecutionProvider");

  ASSERT_EQ(so.provider_options.size(), 1u);
  EXPECT_TRUE(so.provider_options.front().options.empty());
}

TEST(ApplyVariantFileDefaultsTest, NoMatchingProviderEntryDropsProviderOptions) {
  // Per contract: if no entry whose `name` matches the GenAI EP tag exists,
  // the variant provider_options are dropped. EnsurePackageProvider is the
  // helper that's expected to create the entry first.
  Config::SessionOptions so;  // no provider_options at all

  VariantFile vf;
  vf.provider_options = {{"device_id", "1"}};

  ApplyVariantFileDefaults(so, vf, "CUDAExecutionProvider");

  EXPECT_TRUE(so.provider_options.empty());
}

TEST(ApplyVariantFileDefaultsTest, InvalidGraphOptimizationLevelThrows) {
  Config::SessionOptions so;
  VariantFile vf;
  vf.session_options = {{"graph_optimization_level", "ORT_ENABLE_UNICORN"}};

  EXPECT_THROW(ApplyVariantFileDefaults(so, vf, ""), std::runtime_error);
}

}  // namespace Generators::test
