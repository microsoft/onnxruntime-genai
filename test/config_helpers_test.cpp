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

#include "test_utils.h"

namespace Generators::test {

namespace {

constexpr const char* kTinyGpt2RelativePath = "hf-internal-testing/tiny-random-gpt2-fp32";

std::filesystem::path TinyGpt2Path() {
  return std::filesystem::path(MODEL_PATH) / kTinyGpt2RelativePath;
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

}  // namespace Generators::test
