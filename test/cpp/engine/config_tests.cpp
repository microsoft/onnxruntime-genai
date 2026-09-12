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
