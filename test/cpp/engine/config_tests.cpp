// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

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

TEST(ConfigTest, ParsesAndAppliesOneMatchingRuntimeProfile) {
  Config config;
  OverlayConfig(config, R"({
    "engine":{"dynamic_batching":{
      "num_blocks":32,"max_batch_size":4,"max_scheduled_tokens":1024}},
    "runtime_profiles":[{
      "id":"larger-gpu",
      "eligibility":{"minimum_total_device_memory_bytes":34359738368},
      "overlay":{
        "engine":{"dynamic_batching":{"num_blocks":64,"max_batch_size":8}},
        "search":{"chunk_size":512,"max_length":262144}
      }
    }]
  })");

  ApplyRuntimeProfile(config, 34359738368ULL);

  ASSERT_TRUE(config.engine.dynamic_batching);
  EXPECT_EQ(config.engine.dynamic_batching->num_blocks, 64u);
  EXPECT_EQ(config.engine.dynamic_batching->max_batch_size, 8u);
  EXPECT_EQ(config.engine.dynamic_batching->max_scheduled_tokens, 1024u);
  EXPECT_EQ(config.search.chunk_size, 512u);
  EXPECT_EQ(config.search.max_length, 262144);
}

TEST(ConfigTest, RuntimeProfileUsesBaseWhenNoRangeMatches) {
  Config config;
  OverlayConfig(config, R"({
    "engine":{"dynamic_batching":{"num_blocks":32}},
    "runtime_profiles":[{
      "id":"larger-gpu",
      "eligibility":{"minimum_total_device_memory_bytes":34359738368},
      "overlay":{"engine":{"dynamic_batching":{"num_blocks":64}}}
    }]
  })");

  ApplyRuntimeProfile(config, 25769803776ULL);

  ASSERT_TRUE(config.engine.dynamic_batching->num_blocks);
  EXPECT_EQ(*config.engine.dynamic_batching->num_blocks, 32u);
}

TEST(ConfigTest, RuntimeProfilePreservesOmittedSearchFields) {
  Config config;
  config.search.chunk_size = 128;
  config.search.max_length = 4096;
  OverlayConfig(config, R"({
    "engine":{"dynamic_batching":{"num_blocks":32}},
    "runtime_profiles":[{
      "id":"engine-only",
      "eligibility":{"minimum_total_device_memory_bytes":1},
      "overlay":{"engine":{"dynamic_batching":{"num_blocks":64}}}
    }]
  })");

  ApplyRuntimeProfile(config, 1);

  EXPECT_EQ(config.search.chunk_size, 128u);
  EXPECT_EQ(config.search.max_length, 4096);
}

TEST(ConfigTest, AdjacentInclusiveRangesSelectAtTheirBoundaries) {
  Config config;
  config.engine.dynamic_batching = Config::Engine::DynamicBatching{};
  OverlayConfig(config, R"({"runtime_profiles":[
    {"id":"lower","eligibility":{"minimum_total_device_memory_bytes":1,
                                    "maximum_total_device_memory_bytes":10},
     "overlay":{"engine":{"dynamic_batching":{"num_blocks":10}}}},
    {"id":"upper","eligibility":{"minimum_total_device_memory_bytes":11},
     "overlay":{"engine":{"dynamic_batching":{"num_blocks":20}}}}
  ]})");

  auto lower = config;
  ApplyRuntimeProfile(lower, 10);
  EXPECT_EQ(*lower.engine.dynamic_batching->num_blocks, 10u);

  auto upper = config;
  ApplyRuntimeProfile(upper, 11);
  EXPECT_EQ(*upper.engine.dynamic_batching->num_blocks, 20u);
}

TEST(ConfigTest, RejectsDuplicateRuntimeProfileIds) {
  Config config;
  EXPECT_THROW(OverlayConfig(config, R"({"runtime_profiles":[
    {"id":"same","eligibility":{"minimum_total_device_memory_bytes":1},
     "overlay":{"engine":{"dynamic_batching":{"num_blocks":1}}}},
    {"id":"same","eligibility":{"minimum_total_device_memory_bytes":2},
     "overlay":{"engine":{"dynamic_batching":{"num_blocks":2}}}}
  ]})"), std::runtime_error);
}

TEST(ConfigTest, RejectsInvalidRuntimeProfileRange) {
  Config config;
  EXPECT_THROW(OverlayConfig(config, R"({"runtime_profiles":[{
    "id":"invalid",
    "eligibility":{"minimum_total_device_memory_bytes":2,
                   "maximum_total_device_memory_bytes":1},
    "overlay":{"engine":{"dynamic_batching":{"num_blocks":1}}}
  }]})"), std::runtime_error);
}

TEST(ConfigTest, RejectsMultipleMatchingRuntimeProfiles) {
  Config config;
  EXPECT_THROW(OverlayConfig(config, R"({
    "engine":{"dynamic_batching":{"num_blocks":32}},
    "runtime_profiles":[
      {"id":"first","eligibility":{"minimum_total_device_memory_bytes":1,
                                      "maximum_total_device_memory_bytes":10},
      "overlay":{"engine":{"dynamic_batching":{"num_blocks":64}}}},
      {"id":"second","eligibility":{"minimum_total_device_memory_bytes":5,
                                       "maximum_total_device_memory_bytes":15},
       "overlay":{"engine":{"dynamic_batching":{"num_blocks":96}}}}
    ]
  })"), std::runtime_error);
}

TEST(ConfigTest, RejectsUnapprovedSearchOverlayField) {
  Config config;
  EXPECT_THROW(OverlayConfig(config, R"({"runtime_profiles":[{
    "id":"unapproved-search",
    "eligibility":{"minimum_total_device_memory_bytes":1},
    "overlay":{"search":{"temperature":0.5}}
  }]})"), std::runtime_error);
}

TEST(ConfigTest, RejectsModelOverlayField) {
  Config config;
  EXPECT_THROW(OverlayConfig(config, R"({"runtime_profiles":[{
    "id":"filename",
    "eligibility":{"minimum_total_device_memory_bytes":1},
    "overlay":{"model":{"decoder":{"filename":"decoder-large.onnx"}}}
  }]})"), std::runtime_error);
}

TEST(ConfigTest, AppliesSearchOnlyRuntimeProfileWithoutDynamicBatching) {
  Config config;
  OverlayConfig(config, R"({"runtime_profiles":[{
    "id":"search-only",
    "eligibility":{"minimum_total_device_memory_bytes":1},
    "overlay":{"search":{"chunk_size":256,"max_length":4096}}
  }]})");

  ApplyRuntimeProfile(config, 1);

  EXPECT_EQ(config.search.chunk_size, 256u);
  EXPECT_EQ(config.search.max_length, 4096);
}

TEST(ConfigTest, RejectsRuntimeProfileContextLengthChange) {
  Config config;
  config.model.context_length = 4096;
  EXPECT_THROW(OverlayConfig(config, R"({"runtime_profiles":[{
    "id":"context-change",
    "eligibility":{"minimum_total_device_memory_bytes":1},
    "overlay":{"model":{"context_length":8192}}
  }]})"), std::runtime_error);
}

TEST(ConfigTest, RejectsRecursiveRuntimeProfilesOverlay) {
  Config config;
  EXPECT_THROW(OverlayConfig(config, R"({"runtime_profiles":[{
    "id":"recursive",
    "eligibility":{"minimum_total_device_memory_bytes":1},
    "overlay":{"runtime_profiles":[]}
  }]})"), std::runtime_error);
}

TEST(ConfigTest, ParsesPagedScaleBindings) {
  Config config;
  EXPECT_TRUE(config.model.decoder.inputs.past_key_scale_names.empty());
  EXPECT_TRUE(config.model.dflash2.inputs.past_key_scale_names.empty());
  OverlayConfig(config, R"({"model":{
    "decoder":{"inputs":{"past_key_scale_names":"past.%d.ks","past_value_scale_names":"past.%d.vs"},
               "outputs":{"present_key_scale_names":"present.%d.ks","present_value_scale_names":"present.%d.vs"}},
    "dflash2":{"inputs":{"past_key_scale_names":"draft.%d.ks","past_value_scale_names":"draft.%d.vs"},
               "outputs":{"present_key_scale_names":"out.%d.ks","present_value_scale_names":"out.%d.vs"}}
  }})");
  EXPECT_EQ(config.model.decoder.inputs.past_key_scale_names, "past.%d.ks");
  EXPECT_EQ(config.model.decoder.inputs.past_value_scale_names, "past.%d.vs");
  EXPECT_EQ(config.model.decoder.outputs.present_key_scale_names, "present.%d.ks");
  EXPECT_EQ(config.model.decoder.outputs.present_value_scale_names, "present.%d.vs");
  EXPECT_EQ(config.model.dflash2.inputs.past_key_scale_names, "draft.%d.ks");
  EXPECT_EQ(config.model.dflash2.inputs.past_value_scale_names, "draft.%d.vs");
  EXPECT_EQ(config.model.dflash2.outputs.present_key_scale_names, "out.%d.ks");
  EXPECT_EQ(config.model.dflash2.outputs.present_value_scale_names, "out.%d.vs");
}

// The MTP head is always an unquantized full-attention layer, so it has no scale-name configuration
// surface at all. A config that tries to declare one is rejected rather than silently ignored.
TEST(ConfigTest, RejectsMtpScaleBindings) {
  for (const char* json : {R"({"model":{"mtp":{"inputs":{"past_key_scale_names":"mtp.%d.ks"}}}})",
                           R"({"model":{"mtp":{"inputs":{"past_value_scale_names":"mtp.%d.vs"}}}})",
                           R"({"model":{"mtp":{"outputs":{"present_key_scale_names":"mtp.%d.ks"}}}})",
                           R"({"model":{"mtp":{"outputs":{"present_value_scale_names":"mtp.%d.vs"}}}})"}) {
    Config config;
    EXPECT_THROW(OverlayConfig(config, json), std::runtime_error) << json;
  }
}

}  // namespace Generators::test