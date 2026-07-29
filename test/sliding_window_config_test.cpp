// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for the `model.decoder.sliding_window` config block, in particular the
// `cache_slack` field used to size a windowed KV cache on execution providers that evict
// entries themselves. Like validate_config_path_test.cpp these drive the parser through the
// public C API boundary (OgaConfig::Create) instead of reaching into the internal Config
// struct, which is not exported from the shared library.

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "ort_genai.h"

namespace Generators::test {
namespace {

namespace fs_std = std::filesystem;

fs_std::path MakeTempDir(const std::string& suffix) {
  static int counter = 0;
  ++counter;
  const auto dir = fs_std::temp_directory_path() /
                   ("ortgenai_sw_test_" + suffix + "_" + std::to_string(counter));
  std::error_code ec;
  fs_std::remove_all(dir, ec);
  fs_std::create_directories(dir);
  return dir;
}

// Builds a minimal genai_config.json whose decoder carries `sliding_window_json` as its
// `sliding_window` block, and returns the containing directory.
fs_std::path WriteConfigWithSlidingWindow(const std::string& suffix, const std::string& sliding_window_json) {
  const auto root = MakeTempDir(suffix);
  const std::string config =
      "{ \"model\": { \"type\": \"tiny-test-model\","
      " \"vocab_size\": 16, \"context_length\": 32,"
      " \"decoder\": { \"filename\": \"model.onnx\","
      " \"sliding_window\": " +
      sliding_window_json +
      " } },"
      " \"search\": {} }";
  std::ofstream out(root / "genai_config.json", std::ios::binary);
  out << config;
  return root;
}

template <typename Fn>
std::string CaptureThrowMessage(Fn&& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    return e.what();
  }
  return {};
}

}  // namespace

TEST(SlidingWindowConfigTest, AcceptsCacheSlack) {
  const auto root = WriteConfigWithSlidingWindow(
      "slack",
      "{ \"window_size\": 128, \"slide_key_value_cache\": false, \"slide_inputs\": false,"
      " \"layers\": [0, 2], \"cache_slack\": 256 }");
  EXPECT_NO_THROW(OgaConfig::Create(root.string().c_str()));
}

TEST(SlidingWindowConfigTest, AcceptsSlidingWindowWithoutCacheSlack) {
  // cache_slack is optional; older configs that predate it must keep loading.
  const auto root = WriteConfigWithSlidingWindow("no_slack", "{ \"window_size\": 128 }");
  EXPECT_NO_THROW(OgaConfig::Create(root.string().c_str()));
}

TEST(SlidingWindowConfigTest, RejectsMisspelledCacheSlack) {
  // Guards against a typo silently falling back to the default slack.
  const auto root = WriteConfigWithSlidingWindow("typo", "{ \"window_size\": 128, \"cache_slak\": 256 }");
  const std::string message = CaptureThrowMessage([&] { OgaConfig::Create(root.string().c_str()); });
  EXPECT_NE(message.find("cache_slak"), std::string::npos) << message;
}

TEST(SlidingWindowConfigTest, RejectsNonNumericCacheSlack) {
  const auto root = WriteConfigWithSlidingWindow("string", "{ \"window_size\": 128, \"cache_slack\": \"256\" }");
  const std::string message = CaptureThrowMessage([&] { OgaConfig::Create(root.string().c_str()); });
  EXPECT_NE(message.find("cache_slack"), std::string::npos) << message;
}

TEST(SlidingWindowConfigTest, RejectsOutOfRangeCacheSlack) {
  // cache_slack is stored as an int; SafeDoubleToInt must reject values that do not fit.
  const auto root = WriteConfigWithSlidingWindow("overflow", "{ \"window_size\": 128, \"cache_slack\": 1e20 }");
  const std::string message = CaptureThrowMessage([&] { OgaConfig::Create(root.string().c_str()); });
  EXPECT_NE(message.find("cache_slack"), std::string::npos) << message;
}

}  // namespace Generators::test
