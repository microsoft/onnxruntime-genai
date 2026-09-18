// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include <gtest/gtest.h>

#include "config.h"

namespace Generators::test {

TEST(ConfigTest, ParsesTimestampConfiguration) {
  Config config;

  OverlayConfig(config, R"({"model":{
    "timestamp_level":"all",
    "segment_separators":[".","!?"] ,
    "segment_gap_threshold_seconds":1.25
  }})");

  EXPECT_EQ(config.model.timestamp_level, Config::TimestampLevel::All);
  EXPECT_EQ(config.model.segment_separators, (std::vector<std::string>{".", "!?"}));
  EXPECT_EQ(config.model.segment_gap_threshold_seconds, 1.25);
}

TEST(ConfigTest, ParsesDisabledTimestampGap) {
  Config config;
  config.model.segment_gap_threshold_seconds = 1.0;

  OverlayConfig(config, R"({"model":{"segment_gap_threshold_seconds":null}})");

  EXPECT_FALSE(config.model.segment_gap_threshold_seconds.has_value());
}

TEST(ConfigTest, RejectsInvalidTimestampConfiguration) {
  for (const char* json : {
           R"({"model":{"timestamp_level":"token"}})",
           R"({"model":{"segment_gap_threshold_frames":12}})",
           R"({"model":{"segment_gap_threshold_seconds":0}})",
           R"({"model":{"segment_gap_threshold_seconds":-1}})"}) {
    Config config;
    EXPECT_THROW(OverlayConfig(config, json), std::runtime_error);
  }
}

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