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

template <typename PipelineModels>
void ExpectPipelineOverlayMerged(const PipelineModels& pipeline) {
  ASSERT_EQ(pipeline.size(), 3u);

  const auto& untouched_model = pipeline[0];
  EXPECT_EQ(untouched_model.model_id, "untouched_model");
  EXPECT_EQ(untouched_model.filename, "untouched.onnx");
  ASSERT_TRUE(untouched_model.session_options);
  ASSERT_TRUE(untouched_model.session_options->log_id);
  EXPECT_EQ(*untouched_model.session_options->log_id, "untouched");
  EXPECT_FALSE(untouched_model.session_options->enable_profiling);

  const auto& existing_model = pipeline[1];
  EXPECT_EQ(existing_model.model_id, "existing_model");
  EXPECT_EQ(existing_model.filename, "existing.onnx");
  ASSERT_TRUE(existing_model.session_options);
  ASSERT_TRUE(existing_model.session_options->log_id);
  EXPECT_EQ(*existing_model.session_options->log_id, "original");
  ASSERT_TRUE(existing_model.session_options->enable_profiling);
  EXPECT_EQ(*existing_model.session_options->enable_profiling, "profile");
  ASSERT_EQ(existing_model.session_options->provider_options.size(), 1u);
  EXPECT_EQ(existing_model.session_options->provider_options[0].name, "CPU");

  const auto& new_model = pipeline[2];
  EXPECT_EQ(new_model.model_id, "new_model");
  EXPECT_EQ(new_model.filename, "new.onnx");
}

TEST(ConfigTest, PipelineOverlayMergesByModelId) {
  Config config;
  OverlayConfig(config, R"({
    "model": {
      "decoder": {
        "pipeline": [{
          "untouched_model": {
            "filename": "untouched.onnx",
            "session_options": {"log_id": "untouched"}
          },
          "existing_model": {
            "filename": "existing.onnx",
            "session_options": {
              "log_id": "original",
              "provider_options": [{"CPU": {}}]
            }
          }
        }]
      },
      "vision": {
        "pipeline": [{
          "untouched_model": {
            "filename": "untouched.onnx",
            "session_options": {"log_id": "untouched"}
          },
          "existing_model": {
            "filename": "existing.onnx",
            "session_options": {
              "log_id": "original",
              "provider_options": [{"CPU": {}}]
            }
          }
        }]
      }
    }
  })");

  OverlayConfig(config, R"({
    "model": {
      "decoder": {
        "pipeline": [{
          "existing_model": {
            "session_options": {"enable_profiling": "profile"}
          },
          "new_model": {"filename": "new.onnx"}
        }]
      },
      "vision": {
        "pipeline": [{
          "existing_model": {
            "session_options": {"enable_profiling": "profile"}
          },
          "new_model": {"filename": "new.onnx"}
        }]
      }
    }
  })");

  ExpectPipelineOverlayMerged(config.model.decoder.pipeline);
  ExpectPipelineOverlayMerged(config.model.vision.pipeline);
}

}  // namespace Generators::test
