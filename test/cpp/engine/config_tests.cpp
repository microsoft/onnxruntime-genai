// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "config.h"

namespace Generators::test {

TEST(ConfigTest, ParsesStaticBatching) {
  Config config;

  OverlayConfig(config, R"({"engine":{"static_batching":{"max_batch_size":8}}})");

  ASSERT_TRUE(config.engine.static_batching.has_value());
  EXPECT_EQ(config.engine.static_batching->max_batch_size, 8u);
}

TEST(ConfigTest, RejectsNonPositiveStaticBatchSize) {
  Config config;

  try {
    OverlayConfig(config, R"({"engine":{"static_batching":{"max_batch_size":0}}})");
    FAIL() << "Expected invalid max_batch_size to throw";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("engine:static_batching:max_batch_size:"), std::string::npos) << message;
  }
}

TEST(ConfigTest, FullProviderNameUpdatesCanonicalOptions) {
  Config config;
  SetProviderOption(config, "webgpu", "validationMode", "basic");
  SetProviderOption(config, "WebGpuExecutionProvider", "adapterIndex", "3");

  ASSERT_EQ(config.model.decoder.session_options.provider_options.size(), 1u);
  const auto& provider_options = config.model.decoder.session_options.provider_options.front();
  EXPECT_EQ(provider_options.name, "WebGPU");
  ASSERT_EQ(provider_options.options.size(), 2u);
  const auto find_option = [&provider_options](std::string_view name) {
    return std::find_if(provider_options.options.begin(), provider_options.options.end(),
                        [name](const auto& option) { return option.first == name; });
  };
  const auto validation_mode = find_option("validationMode");
  ASSERT_NE(validation_mode, provider_options.options.end());
  EXPECT_EQ(validation_mode->second, "basic");
  const auto adapter_index = find_option("adapterIndex");
  ASSERT_NE(adapter_index, provider_options.options.end());
  EXPECT_EQ(adapter_index->second, "3");
}

}  // namespace Generators::test