// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for the `model.<component>.cache` config block (Config::Model::Cache), which
// consolidates KV/conv/recurrent cache settings and is attached as an optional field to
// Decoder, Encoder, Vision, Embedding, and Mtp. Like config_tests.cpp these drive the parser
// directly through Config + OverlayConfig, which is possible here (unlike the public C API
// tests in sliding_window_config_test.cpp) because this test binary links against the internal
// Config struct directly.

#include <string>

#include <gtest/gtest.h>

#include "config.h"

namespace Generators::test {

TEST(CacheConfigTest, ParsesFullyPopulatedDecoderCache) {
  Config config;

  OverlayConfig(config, R"({
    "model": {
      "decoder": {
        "cache": {
          "global": {
            "eviction_policy": "lru",
            "enable_prefix_caching": true,
            "prefix_cache_max_entries": 32
          },
          "kv_cache": {
            "block_size": 128,
            "num_blocks": 64,
            "sliding_window": {
              "window_size": 512,
              "pad_value": -1,
              "alignment": "left",
              "slide_key_value_cache": false,
              "slide_inputs": false,
              "layers": [0, 2, 4],
              "cache_slack": 16
            }
          },
          "conv_cache": {"cache_size": 8},
          "recurrent_cache": {"state_size": 64}
        }
      }
    }
  })");

  const auto& decoder = config.model.decoder;
  ASSERT_TRUE(decoder.cache.has_value());
  const auto& cache = *decoder.cache;

  EXPECT_EQ(cache.global.eviction_policy, "lru");
  EXPECT_TRUE(cache.global.enable_prefix_caching);
  ASSERT_TRUE(cache.global.prefix_cache_max_entries.has_value());
  EXPECT_EQ(*cache.global.prefix_cache_max_entries, 32);

  ASSERT_TRUE(cache.kv_cache.has_value());
  EXPECT_EQ(cache.kv_cache->block_size, 128u);
  ASSERT_TRUE(cache.kv_cache->num_blocks.has_value());
  EXPECT_EQ(*cache.kv_cache->num_blocks, 64u);

  ASSERT_TRUE(cache.kv_cache->sliding_window.has_value());
  const auto& sw = *cache.kv_cache->sliding_window;
  EXPECT_EQ(sw.window_size, 512);
  EXPECT_EQ(sw.pad_value, -1);
  EXPECT_EQ(sw.alignment, "left");
  EXPECT_FALSE(sw.slide_key_value_cache);
  EXPECT_FALSE(sw.slide_inputs);
  ASSERT_EQ(sw.layers.size(), 3u);
  EXPECT_EQ(sw.layers[0], 0);
  EXPECT_EQ(sw.layers[1], 2);
  EXPECT_EQ(sw.layers[2], 4);
  EXPECT_EQ(sw.cache_slack, 16);

  ASSERT_TRUE(cache.conv_cache.has_value());
  EXPECT_EQ(cache.conv_cache->cache_size, 8);

  ASSERT_TRUE(cache.recurrent_cache.has_value());
  ASSERT_TRUE(cache.recurrent_cache->state_size.has_value());
  EXPECT_EQ(*cache.recurrent_cache->state_size, 64);
}

TEST(CacheConfigTest, LegacySlidingWindowAndConvCacheSizePopulateNewCacheFields) {
  Config config;

  // Only the legacy top-level fields are set; the new decoder.cache.* fields are absent.
  OverlayConfig(config, R"({
    "model": {
      "decoder": {
        "sliding_window": {
          "window_size": 256,
          "pad_value": 0,
          "layers": [1, 3],
          "cache_slack": 4
        },
        "conv_cache_size": 12
      }
    }
  })");

  const auto& decoder = config.model.decoder;

  // Legacy fields remain fully functional, unchanged.
  ASSERT_TRUE(decoder.sliding_window.has_value());
  EXPECT_EQ(decoder.sliding_window->window_size, 256);
  EXPECT_EQ(decoder.conv_cache_size, 12);

  // The new cache.* fields are populated to mirror the legacy values for consistency.
  ASSERT_TRUE(decoder.cache.has_value());
  ASSERT_TRUE(decoder.cache->kv_cache.has_value());
  ASSERT_TRUE(decoder.cache->kv_cache->sliding_window.has_value());
  EXPECT_EQ(decoder.cache->kv_cache->sliding_window->window_size, 256);
  EXPECT_EQ(decoder.cache->kv_cache->sliding_window->cache_slack, 4);
  ASSERT_EQ(decoder.cache->kv_cache->sliding_window->layers.size(), 2u);

  ASSERT_TRUE(decoder.cache->conv_cache.has_value());
  EXPECT_EQ(decoder.cache->conv_cache->cache_size, 12);
}

TEST(CacheConfigTest, NewCacheFieldsTakePrecedenceOverLegacyWhenBothSet) {
  Config config;

  // Both the legacy fields and the new decoder.cache.* equivalents are set with different values.
  OverlayConfig(config, R"({
    "model": {
      "decoder": {
        "sliding_window": {"window_size": 111},
        "conv_cache_size": 22,
        "cache": {
          "kv_cache": {"sliding_window": {"window_size": 222}},
          "conv_cache": {"cache_size": 33}
        }
      }
    }
  })");

  const auto& decoder = config.model.decoder;
  ASSERT_TRUE(decoder.sliding_window.has_value());
  EXPECT_EQ(decoder.sliding_window->window_size, 111);
  EXPECT_EQ(decoder.conv_cache_size, 22);

  ASSERT_TRUE(decoder.cache.has_value());
  ASSERT_TRUE(decoder.cache->kv_cache.has_value());
  ASSERT_TRUE(decoder.cache->kv_cache->sliding_window.has_value());
  // The new cache.* fields take precedence and are not overwritten by the legacy ones.
  EXPECT_EQ(decoder.cache->kv_cache->sliding_window->window_size, 222);
  ASSERT_TRUE(decoder.cache->conv_cache.has_value());
  EXPECT_EQ(decoder.cache->conv_cache->cache_size, 33);
}

TEST(CacheConfigTest, DefaultsWhenCacheAbsent) {
  Config config;

  OverlayConfig(config, R"({"model": {"decoder": {}}})");

  EXPECT_FALSE(config.model.decoder.cache.has_value());
  EXPECT_FALSE(config.model.decoder.sliding_window.has_value());
  EXPECT_EQ(config.model.decoder.conv_cache_size, 0);
  EXPECT_FALSE(config.model.encoder.cache.has_value());
  EXPECT_FALSE(config.model.vision.cache.has_value());
  EXPECT_FALSE(config.model.embedding.cache.has_value());
  EXPECT_FALSE(config.model.mtp.cache.has_value());
}

TEST(CacheConfigTest, ParsesEncoderCacheIndependentlyOfDecoderCache) {
  Config config;

  OverlayConfig(config, R"({
    "model": {
      "encoder": {
        "cache": {
          "global": {"eviction_policy": "fifo"},
          "kv_cache": {"block_size": 64}
        }
      }
    }
  })");

  ASSERT_TRUE(config.model.encoder.cache.has_value());
  EXPECT_EQ(config.model.encoder.cache->global.eviction_policy, "fifo");
  ASSERT_TRUE(config.model.encoder.cache->kv_cache.has_value());
  EXPECT_EQ(config.model.encoder.cache->kv_cache->block_size, 64u);

  // Decoder's cache is untouched.
  EXPECT_FALSE(config.model.decoder.cache.has_value());
}

TEST(CacheConfigTest, RejectsInvalidEvictionPolicy) {
  Config config;

  try {
    OverlayConfig(config, R"({"model": {"decoder": {"cache": {"global": {"eviction_policy": "bogus"}}}}})");
    FAIL() << "Expected invalid eviction_policy to throw";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("eviction_policy"), std::string::npos) << message;
  }
}

}  // namespace Generators::test
