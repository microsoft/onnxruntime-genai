// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Unit tests for the v2.1 runtime-vs-build-time feature namespace under session_options
// (issue #2114, PR-F, design §7). These verify that:
//   * a `session_options.runtime` / `session_options.build_requires` block parses into the typed
//     Config::SessionOptions::{runtime,build_requires} structs with its keyed values preserved (schema),
//   * a config WITHOUT the new namespace parses byte-for-byte unchanged -- both blocks stay nullopt and
//     the existing session_options keys (config_entries) are untouched (back-compat),
//   * an unknown key OR a mis-namespaced / nonsensical feature value throws a clear error rather than
//     being silently ignored (validation).
//
// The features in the runtime namespace are DECLARED-ONLY at this PR: parsed + validated, but their
// per-feature session-time effects are deferred (see design §7, the "KV-quant/paging numeric parity
// out of scope" note). build_requires.* is "declared, never synthesized" by contract.

#include <string>
#include <gtest/gtest.h>

#include "generators.h"
#include "test_utils.h"

namespace {

Generators::Config LoadGpt2ConfigWithOverlay(const std::string& overlay) {
  return Generators::Config(
      fs::path(std::string(MODEL_PATH) + "hf-internal-testing/tiny-random-gpt2-fp32"), overlay);
}

}  // namespace

// (a) Schema: a full runtime + build_requires block parses into the typed structs with keyed values
// preserved.
TEST(RuntimeFeatureNamespaceTests, SchemaParsesRuntimeAndBuildNamespace) {
  const std::string overlay = R"({
    "model": { "decoder": { "session_options": {
      "runtime": {
        "kv_cache": { "dtype": "fp8", "quant": "per_token" },
        "paging": { "enabled": true, "block_size": 256 },
        "prefix_cache": { "enabled": true },
        "sliding_window": { "size": 4096, "sink_tokens": 4 },
        "chunked_prefill": { "max_batched_tokens": 2048 },
        "precision": "fp16"
      },
      "build_requires": {
        "attention": "gqa",
        "quantization": "awq",
        "extra_heads": "medusa"
      }
    } } }
  })";

  auto config = LoadGpt2ConfigWithOverlay(overlay);
  const auto& so = config.model.decoder.session_options;

  ASSERT_TRUE(so.runtime.has_value());
  const auto& rt = *so.runtime;

  ASSERT_TRUE(rt.kv_cache.has_value());
  ASSERT_TRUE(rt.kv_cache->dtype.has_value());
  EXPECT_EQ(*rt.kv_cache->dtype, "fp8");
  ASSERT_TRUE(rt.kv_cache->quant.has_value());
  EXPECT_EQ(*rt.kv_cache->quant, "per_token");

  ASSERT_TRUE(rt.paging.has_value());
  EXPECT_TRUE(rt.paging->enabled);
  ASSERT_TRUE(rt.paging->block_size.has_value());
  EXPECT_EQ(*rt.paging->block_size, 256);

  ASSERT_TRUE(rt.prefix_cache.has_value());
  EXPECT_TRUE(rt.prefix_cache->enabled);

  ASSERT_TRUE(rt.sliding_window.has_value());
  ASSERT_TRUE(rt.sliding_window->size.has_value());
  EXPECT_EQ(*rt.sliding_window->size, 4096);
  ASSERT_TRUE(rt.sliding_window->sink_tokens.has_value());
  EXPECT_EQ(*rt.sliding_window->sink_tokens, 4);

  ASSERT_TRUE(rt.chunked_prefill.has_value());
  ASSERT_TRUE(rt.chunked_prefill->max_batched_tokens.has_value());
  EXPECT_EQ(*rt.chunked_prefill->max_batched_tokens, 2048);

  ASSERT_TRUE(rt.precision.has_value());
  EXPECT_EQ(*rt.precision, "fp16");

  ASSERT_TRUE(so.build_requires.has_value());
  const auto& br = *so.build_requires;
  ASSERT_TRUE(br.attention.has_value());
  EXPECT_EQ(*br.attention, "gqa");
  ASSERT_TRUE(br.quantization.has_value());
  EXPECT_EQ(*br.quantization, "awq");
  ASSERT_TRUE(br.extra_heads.has_value());
  EXPECT_EQ(*br.extra_heads, "medusa");
}

// (a) Schema: a partial runtime block leaves the unspecified sub-features absent (block-presence
// gating is per sub-feature, not all-or-nothing).
TEST(RuntimeFeatureNamespaceTests, SchemaPartialRuntimeBlockLeavesOthersAbsent) {
  const std::string overlay = R"({
    "model": { "decoder": { "session_options": {
      "runtime": { "kv_cache": { "dtype": "int8" } }
    } } }
  })";

  auto config = LoadGpt2ConfigWithOverlay(overlay);
  const auto& so = config.model.decoder.session_options;

  ASSERT_TRUE(so.runtime.has_value());
  ASSERT_TRUE(so.runtime->kv_cache.has_value());
  ASSERT_TRUE(so.runtime->kv_cache->dtype.has_value());
  EXPECT_EQ(*so.runtime->kv_cache->dtype, "int8");
  EXPECT_FALSE(so.runtime->kv_cache->quant.has_value());
  EXPECT_FALSE(so.runtime->paging.has_value());
  EXPECT_FALSE(so.runtime->prefix_cache.has_value());
  EXPECT_FALSE(so.runtime->precision.has_value());
  EXPECT_FALSE(so.build_requires.has_value());
}

// (b) Back-compat: a config WITHOUT the new namespace leaves both blocks nullopt and parses the
// existing session_options keys exactly as before (no behavior change).
TEST(RuntimeFeatureNamespaceTests, BackCompatNoNamespaceUnchanged) {
  auto baseline = LoadGpt2ConfigWithOverlay("");
  const auto& so = baseline.model.decoder.session_options;

  EXPECT_FALSE(so.runtime.has_value());
  EXPECT_FALSE(so.build_requires.has_value());

  // The existing config_entries from the shipped genai_config.json are still present and untouched.
  bool found_device_allocator_entry = false;
  for (const auto& entry : so.config_entries) {
    if (entry.first == "session.use_device_allocator_for_initializers") {
      found_device_allocator_entry = true;
      EXPECT_EQ(entry.second, "0");
    }
  }
  EXPECT_TRUE(found_device_allocator_entry);

  // Adding ONLY the runtime namespace must not disturb the legacy config_entries -- the runtime keys
  // are routed to the typed struct, not appended to config_entries.
  auto with_runtime = LoadGpt2ConfigWithOverlay(R"({
    "model": { "decoder": { "session_options": { "runtime": { "precision": "fp16" } } } }
  })");
  EXPECT_EQ(with_runtime.model.decoder.session_options.config_entries.size(), so.config_entries.size());
  ASSERT_TRUE(with_runtime.model.decoder.session_options.runtime.has_value());
}

// (c) Validation: an unknown KEY inside the runtime namespace throws (not silently swallowed into
// config_entries the way unknown scalar session_options keys are).
TEST(RuntimeFeatureNamespaceTests, UnknownRuntimeKeyThrows) {
  const std::string overlay = R"({
    "model": { "decoder": { "session_options": {
      "runtime": { "not_a_real_feature": { "enabled": true } }
    } } }
  })";
  EXPECT_THROW(LoadGpt2ConfigWithOverlay(overlay), std::runtime_error);
}

// (c) Validation: a BUILD-TIME weight quantization scheme declared in the RUNTIME kv_cache.dtype slot
// is nonsensical ("declared, never synthesized") and throws a clear, namespaced error.
TEST(RuntimeFeatureNamespaceTests, MisNamespacedBuildFeatureInRuntimeThrows) {
  const std::string overlay = R"({
    "model": { "decoder": { "session_options": {
      "runtime": { "kv_cache": { "dtype": "awq" } }
    } } }
  })";
  EXPECT_THROW(
      {
        try {
          LoadGpt2ConfigWithOverlay(overlay);
        } catch (const std::runtime_error& e) {
          const std::string what = e.what();
          EXPECT_NE(what.find("runtime.kv_cache.dtype"), std::string::npos);
          EXPECT_NE(what.find("build_requires.quantization"), std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

// (c) Validation: an unknown enum value in build_requires throws a clear error.
TEST(RuntimeFeatureNamespaceTests, UnknownBuildEnumValueThrows) {
  const std::string overlay = R"({
    "model": { "decoder": { "session_options": {
      "build_requires": { "attention": "definitely_not_an_attention" }
    } } }
  })";
  EXPECT_THROW(LoadGpt2ConfigWithOverlay(overlay), std::runtime_error);
}

// (c) Validation: a runtime feature mistakenly declared as a build extra-head throws, enforcing the
// reverse direction of the runtime/build split.
TEST(RuntimeFeatureNamespaceTests, MisNamespacedRuntimeFeatureInBuildThrows) {
  const std::string overlay = R"({
    "model": { "decoder": { "session_options": {
      "build_requires": { "extra_heads": "paging" }
    } } }
  })";
  EXPECT_THROW(LoadGpt2ConfigWithOverlay(overlay), std::runtime_error);
}
