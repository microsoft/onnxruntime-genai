// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Unit tests for Pipeline-as-Config v2 schema, presets, and v1 translator.

#include "generators.h"
#include "pipeline_config_schema.h"
#include "pipeline_presets.h"
#include "v1_translator.h"
#include "config.h"

#include <gtest/gtest.h>
#include <string>

namespace Generators::test {

// ============================================================================
// Preset tests
// ============================================================================

TEST(PipelinePresets, AutoRegressiveDecoder) {
  auto config = GetPreset("autoregressive-decoder");
  ASSERT_EQ(config.sessions.size(), 1u);
  ASSERT_TRUE(config.sessions.count("decoder"));
  EXPECT_EQ(config.sessions.at("decoder").file, "model.onnx");
  EXPECT_EQ(config.sessions.at("decoder").role, "decoder");

  ASSERT_EQ(config.flow.size(), 1u);
  EXPECT_EQ(config.flow[0].run, "decoder");
  EXPECT_EQ(config.flow[0].when, "always");

  EXPECT_EQ(config.state.kv_cache.format, "auto");
  EXPECT_EQ(config.state.position_ids.strategy, "auto");
}

TEST(PipelinePresets, VisionLanguage) {
  auto config = GetPreset("vision-language");
  ASSERT_EQ(config.sessions.size(), 3u);
  EXPECT_TRUE(config.sessions.count("vision"));
  EXPECT_TRUE(config.sessions.count("embedding"));
  EXPECT_TRUE(config.sessions.count("decoder"));

  EXPECT_EQ(config.sessions.at("vision").role, "vision");
  EXPECT_EQ(config.sessions.at("embedding").role, "embedding");
  EXPECT_EQ(config.sessions.at("decoder").role, "decoder");

  ASSERT_EQ(config.flow.size(), 3u);
  EXPECT_EQ(config.flow[0].run, "vision");
  EXPECT_EQ(config.flow[0].when, "prompt");
  EXPECT_EQ(config.flow[0].loop, "per_image");
  EXPECT_EQ(config.flow[1].run, "embedding");
  EXPECT_EQ(config.flow[1].when, "prompt");
  EXPECT_EQ(config.flow[2].run, "decoder");
  EXPECT_EQ(config.flow[2].when, "always");

  ASSERT_GE(config.dataflow.size(), 2u);
  EXPECT_EQ(config.dataflow[0].from_session, "vision");
  EXPECT_EQ(config.dataflow[0].to_session, "embedding");
  EXPECT_EQ(config.dataflow[1].from_session, "embedding");
  EXPECT_EQ(config.dataflow[1].to_session, "decoder");
}

TEST(PipelinePresets, EncoderDecoder) {
  auto config = GetPreset("encoder-decoder");
  ASSERT_EQ(config.sessions.size(), 2u);
  EXPECT_TRUE(config.sessions.count("encoder"));
  EXPECT_TRUE(config.sessions.count("decoder"));

  ASSERT_EQ(config.flow.size(), 2u);
  EXPECT_EQ(config.flow[0].run, "encoder");
  EXPECT_EQ(config.flow[0].when, "once");
  EXPECT_EQ(config.flow[1].run, "decoder");
  EXPECT_EQ(config.flow[1].when, "always");

  ASSERT_EQ(config.dataflow.size(), 1u);
  EXPECT_EQ(config.dataflow[0].from_session, "encoder");
  EXPECT_EQ(config.dataflow[0].to_session, "decoder");
}

TEST(PipelinePresets, UnknownPresetThrows) {
  EXPECT_THROW(GetPreset("nonexistent-preset"), std::runtime_error);
}

// ============================================================================
// ApplyOverrides tests
// ============================================================================

TEST(PipelinePresets, OverrideSessionMerge) {
  auto base = GetPreset("autoregressive-decoder");
  PipelineConfig overrides;
  overrides.sessions["decoder"] = PipelineConfig::Session{
      .file = "custom_model.onnx",
      .role = "decoder",
  };

  ApplyOverrides(base, overrides);
  EXPECT_EQ(base.sessions.at("decoder").file, "custom_model.onnx");
  // Flow should be unchanged (override flow is empty)
  ASSERT_EQ(base.flow.size(), 1u);
}

TEST(PipelinePresets, OverrideFlowReplacesEntirely) {
  auto base = GetPreset("vision-language");
  ASSERT_EQ(base.flow.size(), 3u);

  PipelineConfig overrides;
  overrides.flow.push_back(PipelineConfig::FlowStep{
      .run = "decoder",
      .when = "always",
  });

  ApplyOverrides(base, overrides);
  ASSERT_EQ(base.flow.size(), 1u);  // Replaced, not appended
  EXPECT_EQ(base.flow[0].run, "decoder");
}

TEST(PipelinePresets, OverrideAddsNewSession) {
  auto base = GetPreset("autoregressive-decoder");
  ASSERT_EQ(base.sessions.size(), 1u);

  PipelineConfig overrides;
  overrides.sessions["custom"] = PipelineConfig::Session{
      .file = "custom.onnx",
      .role = "custom",
  };

  ApplyOverrides(base, overrides);
  ASSERT_EQ(base.sessions.size(), 2u);
  EXPECT_TRUE(base.sessions.count("custom"));
}

TEST(PipelinePresets, OverrideStateFields) {
  auto base = GetPreset("autoregressive-decoder");
  PipelineConfig overrides;
  overrides.state.kv_cache.format = "windowed";
  overrides.state.position_ids.strategy = "mrope_3d";

  ApplyOverrides(base, overrides);
  EXPECT_EQ(base.state.kv_cache.format, "windowed");
  EXPECT_EQ(base.state.position_ids.strategy, "mrope_3d");
}

// ============================================================================
// Validation tests
// ============================================================================

TEST(PipelineValidation, ValidConfigPasses) {
  auto config = GetPreset("autoregressive-decoder");
  EXPECT_NO_THROW(ValidatePipelineConfig(config));
}

TEST(PipelineValidation, UnknownSessionInFlowThrows) {
  PipelineConfig config;
  config.sessions["decoder"] = PipelineConfig::Session{.file = "model.onnx"};
  config.flow.push_back(PipelineConfig::FlowStep{.run = "nonexistent"});

  EXPECT_THROW(ValidatePipelineConfig(config), std::runtime_error);
}

TEST(PipelineValidation, InvalidWhenValueThrows) {
  PipelineConfig config;
  config.sessions["decoder"] = PipelineConfig::Session{.file = "model.onnx"};
  config.flow.push_back(PipelineConfig::FlowStep{
      .run = "decoder",
      .when = "invalid_when",
  });

  EXPECT_THROW(ValidatePipelineConfig(config), std::runtime_error);
}

TEST(PipelineValidation, InvalidLoopValueThrows) {
  PipelineConfig config;
  config.sessions["decoder"] = PipelineConfig::Session{.file = "model.onnx"};
  config.flow.push_back(PipelineConfig::FlowStep{
      .run = "decoder",
      .when = "always",
      .loop = "invalid_loop",
  });

  EXPECT_THROW(ValidatePipelineConfig(config), std::runtime_error);
}

TEST(PipelineValidation, DataflowUnknownFromSessionThrows) {
  PipelineConfig config;
  config.sessions["decoder"] = PipelineConfig::Session{.file = "model.onnx"};
  config.dataflow.push_back(PipelineConfig::DataflowWire{
      .from_session = "unknown",
      .from_output = "out",
      .to_session = "decoder",
      .to_input = "in",
  });

  EXPECT_THROW(ValidatePipelineConfig(config), std::runtime_error);
}

TEST(PipelineValidation, DataflowEmptyTensorNameThrows) {
  PipelineConfig config;
  config.sessions["a"] = PipelineConfig::Session{.file = "a.onnx"};
  config.sessions["b"] = PipelineConfig::Session{.file = "b.onnx"};
  config.dataflow.push_back(PipelineConfig::DataflowWire{
      .from_session = "a",
      .from_output = "",  // Empty — should fail
      .to_session = "b",
      .to_input = "input",
  });

  EXPECT_THROW(ValidatePipelineConfig(config), std::runtime_error);
}

TEST(PipelineValidation, EmptyConfigPasses) {
  PipelineConfig config;
  EXPECT_NO_THROW(ValidatePipelineConfig(config));
}

// ============================================================================
// V1 translator tests
// ============================================================================

// Helper: create a minimal v1 Config with a model type
namespace {
Config MakeMinimalV1Config(const std::string& model_type) {
  Config config;
  config.version = 1;
  config.model.type = model_type;
  config.model.decoder.filename = "decoder_model.onnx";
  return config;
}
}  // namespace

TEST(V1Translator, LLMTranslatesToAutoRegressiveDecoder) {
  auto v1 = MakeMinimalV1Config("llama");
  auto pipeline = TranslateV1Config(v1);

  ASSERT_TRUE(pipeline.sessions.count("decoder"));
  EXPECT_EQ(pipeline.sessions.at("decoder").file, "decoder_model.onnx");

  ASSERT_GE(pipeline.flow.size(), 1u);
  bool has_decoder_flow = false;
  for (const auto& step : pipeline.flow) {
    if (step.run == "decoder" && step.when == "always") {
      has_decoder_flow = true;
    }
  }
  EXPECT_TRUE(has_decoder_flow);
}

TEST(V1Translator, AllLLMTypesTranslate) {
  // Verify all known LLM types translate without throwing
  for (const auto& type : {"chatglm", "decoder", "gemma", "llama", "phi", "qwen2", "qwen3", "smollm3"}) {
    auto v1 = MakeMinimalV1Config(type);
    EXPECT_NO_THROW(TranslateV1Config(v1)) << "Failed for type: " << type;
  }
}

TEST(V1Translator, VLMTranslatesToVisionLanguage) {
  auto v1 = MakeMinimalV1Config("phi3v");
  v1.model.vision.filename = "vision.onnx";
  v1.model.embedding.filename = "embed.onnx";
  auto pipeline = TranslateV1Config(v1);

  ASSERT_TRUE(pipeline.sessions.count("vision"));
  ASSERT_TRUE(pipeline.sessions.count("embedding"));
  ASSERT_TRUE(pipeline.sessions.count("decoder"));

  EXPECT_EQ(pipeline.sessions.at("vision").file, "vision.onnx");
  EXPECT_EQ(pipeline.sessions.at("embedding").file, "embed.onnx");
  EXPECT_EQ(pipeline.sessions.at("decoder").file, "decoder_model.onnx");
}

TEST(V1Translator, QwenVLGetsSpecialPositionStrategy) {
  auto v1 = MakeMinimalV1Config("qwen2_5_vl");
  v1.model.vision.filename = "vision.onnx";
  v1.model.embedding.filename = "embed.onnx";
  auto pipeline = TranslateV1Config(v1);

  EXPECT_EQ(pipeline.state.position_ids.strategy, "mrope_3d");
}

TEST(V1Translator, WhisperTranslatesToEncoderDecoder) {
  auto v1 = MakeMinimalV1Config("whisper");
  v1.model.encoder.filename = "encoder.onnx";
  auto pipeline = TranslateV1Config(v1);

  ASSERT_TRUE(pipeline.sessions.count("encoder"));
  ASSERT_TRUE(pipeline.sessions.count("decoder"));
  EXPECT_EQ(pipeline.sessions.at("encoder").file, "encoder.onnx");
}

TEST(V1Translator, KVCachePatternsPropagate) {
  auto v1 = MakeMinimalV1Config("llama");
  v1.model.decoder.inputs.past_key_names = "past_key_values.%d.key";
  v1.model.decoder.inputs.past_value_names = "past_key_values.%d.value";
  v1.model.decoder.outputs.present_key_names = "present.%d.key";
  v1.model.decoder.outputs.present_value_names = "present.%d.value";
  auto pipeline = TranslateV1Config(v1);

  EXPECT_EQ(pipeline.state.kv_cache.past_key_pattern.value_or(""), "past_key_values.%d.key");
  EXPECT_EQ(pipeline.state.kv_cache.present_key_pattern.value_or(""), "present.%d.key");
}

TEST(V1Translator, UnknownTypeStillTranslates) {
  // Unknown model types should still translate (fallback to decoder preset)
  auto v1 = MakeMinimalV1Config("totally_unknown_type");
  EXPECT_NO_THROW(TranslateV1Config(v1));

  auto pipeline = TranslateV1Config(v1);
  EXPECT_TRUE(pipeline.sessions.count("decoder"));
}

}  // namespace Generators::test
