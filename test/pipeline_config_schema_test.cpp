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

  EXPECT_EQ(config.state.kv_cache.format.value_or(""), "auto");
  EXPECT_EQ(config.state.position_ids.strategy.value_or(""), "auto");
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
  EXPECT_EQ(base.state.kv_cache.format.value_or(""), "windowed");
  EXPECT_EQ(base.state.position_ids.strategy.value_or(""), "mrope_3d");
}

TEST(PipelinePresets, OverrideAutoValueApplies) {
  // Regression test: explicitly setting "auto" should override a non-auto preset value
  auto base = GetPreset("autoregressive-decoder");
  base.state.kv_cache.format = "windowed";  // Simulate a preset with non-auto default

  PipelineConfig overrides;
  overrides.state.kv_cache.format = "auto";  // Explicitly set back to auto

  ApplyOverrides(base, overrides);
  EXPECT_EQ(base.state.kv_cache.format.value_or(""), "auto");
}

TEST(PipelinePresets, UnsetOverrideDoesNotClobber) {
  // When an override field is not set (nullopt), the base value should be preserved
  auto base = GetPreset("autoregressive-decoder");
  base.state.kv_cache.format = "windowed";

  PipelineConfig overrides;
  // overrides.state.kv_cache.format is nullopt (not set)

  ApplyOverrides(base, overrides);
  EXPECT_EQ(base.state.kv_cache.format.value_or(""), "windowed");  // Preserved
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

  EXPECT_EQ(pipeline.state.position_ids.strategy.value_or(""), "mrope_3d");
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

TEST(V1Translator, Phi4MMTranslatesToVisionLanguage) {
  auto v1 = MakeMinimalV1Config("phi4mm");
  v1.model.vision.filename = "vision.onnx";
  v1.model.embedding.filename = "embed.onnx";
  auto pipeline = TranslateV1Config(v1);

  ASSERT_TRUE(pipeline.sessions.count("vision"));
  ASSERT_TRUE(pipeline.sessions.count("embedding"));
  ASSERT_TRUE(pipeline.sessions.count("decoder"));
  EXPECT_EQ(pipeline.sessions.at("vision").file, "vision.onnx");
}

TEST(V1Translator, MarianSSRUTranslatesToEncoderDecoder) {
  auto v1 = MakeMinimalV1Config("marian-ssru");
  v1.model.encoder.filename = "encoder.onnx";
  auto pipeline = TranslateV1Config(v1);

  ASSERT_TRUE(pipeline.sessions.count("encoder"));
  ASSERT_TRUE(pipeline.sessions.count("decoder"));
  EXPECT_EQ(pipeline.sessions.at("encoder").file, "encoder.onnx");

  // Should have encoder→decoder flow
  ASSERT_GE(pipeline.flow.size(), 2u);
  EXPECT_EQ(pipeline.flow[0].run, "encoder");
  EXPECT_EQ(pipeline.flow[0].when, "once");
}

TEST(V1Translator, AllLLMTypesTranslateComprehensive) {
  // All 21 LLM types from model_type.h should translate without throwing
  for (const auto& type : {"chatglm", "decoder", "ernie4_5", "gemma", "gemma2",
                            "gemma3_text", "gpt2", "gptoss", "granite", "internlm2",
                            "llama", "mistral", "nemotron", "olmo", "phi", "phimoe",
                            "phi3", "phi3small", "qwen2", "qwen3", "smollm3"}) {
    auto v1 = MakeMinimalV1Config(type);
    EXPECT_NO_THROW(TranslateV1Config(v1)) << "Failed for type: " << type;
    auto pipeline = TranslateV1Config(v1);
    EXPECT_TRUE(pipeline.sessions.count("decoder")) << "No decoder for type: " << type;
  }
}

TEST(V1Translator, AllVLMTypesTranslate) {
  for (const auto& type : {"fara", "gemma3", "phi3v", "qwen2_5_vl"}) {
    auto v1 = MakeMinimalV1Config(type);
    EXPECT_NO_THROW(TranslateV1Config(v1)) << "Failed for type: " << type;
    auto pipeline = TranslateV1Config(v1);
    EXPECT_TRUE(pipeline.sessions.count("vision")) << "No vision for type: " << type;
    EXPECT_TRUE(pipeline.sessions.count("embedding")) << "No embedding for type: " << type;
  }
}

// ============================================================================
// Edge case tests: complex dataflow, standalone sessions, flow variants
// ============================================================================

TEST(PipelineEdgeCases, ComplexMultiSessionDataflow) {
  // 5-session pipeline with chain: preprocessor → vision → projector → embedding → decoder
  PipelineConfig config;
  config.sessions["preprocessor"] = PipelineConfig::Session{.file = "preproc.onnx", .role = "preprocessor"};
  config.sessions["vision"] = PipelineConfig::Session{.file = "vision.onnx", .role = "vision"};
  config.sessions["projector"] = PipelineConfig::Session{.file = "projector.onnx", .role = "projector"};
  config.sessions["embedding"] = PipelineConfig::Session{.file = "embed.onnx", .role = "embedding"};
  config.sessions["decoder"] = PipelineConfig::Session{.file = "decoder.onnx", .role = "decoder"};

  config.flow.push_back(PipelineConfig::FlowStep{.run = "preprocessor", .when = "once"});
  config.flow.push_back(PipelineConfig::FlowStep{.run = "vision", .when = "prompt", .loop = "per_image"});
  config.flow.push_back(PipelineConfig::FlowStep{.run = "projector", .when = "prompt"});
  config.flow.push_back(PipelineConfig::FlowStep{.run = "embedding", .when = "prompt"});
  config.flow.push_back(PipelineConfig::FlowStep{.run = "decoder", .when = "always"});

  // Chain wiring: each session feeds the next
  config.dataflow.push_back(PipelineConfig::DataflowWire{
      .from_session = "preprocessor", .from_output = "processed",
      .to_session = "vision", .to_input = "pixel_values"});
  config.dataflow.push_back(PipelineConfig::DataflowWire{
      .from_session = "vision", .from_output = "features",
      .to_session = "projector", .to_input = "vision_features"});
  config.dataflow.push_back(PipelineConfig::DataflowWire{
      .from_session = "projector", .from_output = "projected",
      .to_session = "embedding", .to_input = "image_features"});
  config.dataflow.push_back(PipelineConfig::DataflowWire{
      .from_session = "embedding", .from_output = "inputs_embeds",
      .to_session = "decoder", .to_input = "inputs_embeds"});

  EXPECT_NO_THROW(ValidatePipelineConfig(config));
  EXPECT_EQ(config.sessions.size(), 5u);
  EXPECT_EQ(config.flow.size(), 5u);
  EXPECT_EQ(config.dataflow.size(), 4u);
}

TEST(PipelineEdgeCases, StandaloneSessionNoDataflow) {
  // A session with no dataflow wires (standalone, e.g. audio preprocessor)
  PipelineConfig config;
  config.sessions["standalone"] = PipelineConfig::Session{.file = "standalone.onnx", .role = "custom"};
  config.sessions["decoder"] = PipelineConfig::Session{.file = "decoder.onnx", .role = "decoder"};

  config.flow.push_back(PipelineConfig::FlowStep{.run = "standalone", .when = "once"});
  config.flow.push_back(PipelineConfig::FlowStep{.run = "decoder", .when = "always"});
  // No dataflow wires — standalone session has its own inputs/outputs

  EXPECT_NO_THROW(ValidatePipelineConfig(config));
  EXPECT_TRUE(config.dataflow.empty());
}

TEST(PipelineEdgeCases, FlowWithOnlyOnceSteps) {
  // All flow steps are 'once' — no 'always' or 'prompt' steps
  PipelineConfig config;
  config.sessions["init_a"] = PipelineConfig::Session{.file = "a.onnx"};
  config.sessions["init_b"] = PipelineConfig::Session{.file = "b.onnx"};

  config.flow.push_back(PipelineConfig::FlowStep{.run = "init_a", .when = "once"});
  config.flow.push_back(PipelineConfig::FlowStep{.run = "init_b", .when = "once"});

  EXPECT_NO_THROW(ValidatePipelineConfig(config));
}

TEST(PipelineEdgeCases, FlowWithOnlyPromptSteps) {
  // All flow steps are 'prompt' — no 'always' or 'once' steps
  PipelineConfig config;
  config.sessions["encoder"] = PipelineConfig::Session{.file = "enc.onnx"};
  config.sessions["projector"] = PipelineConfig::Session{.file = "proj.onnx"};

  config.flow.push_back(PipelineConfig::FlowStep{.run = "encoder", .when = "prompt"});
  config.flow.push_back(PipelineConfig::FlowStep{.run = "projector", .when = "prompt"});

  EXPECT_NO_THROW(ValidatePipelineConfig(config));
}

TEST(PipelineEdgeCases, FlowWithOnlyAlwaysSteps) {
  // Multiple sessions all running every step
  PipelineConfig config;
  config.sessions["decoder_a"] = PipelineConfig::Session{.file = "a.onnx"};
  config.sessions["decoder_b"] = PipelineConfig::Session{.file = "b.onnx"};

  config.flow.push_back(PipelineConfig::FlowStep{.run = "decoder_a", .when = "always"});
  config.flow.push_back(PipelineConfig::FlowStep{.run = "decoder_b", .when = "always"});

  EXPECT_NO_THROW(ValidatePipelineConfig(config));
}

TEST(PipelineEdgeCases, PresetOverrideConflictingValues) {
  // Override a preset with conflicting session roles and state values
  auto base = GetPreset("vision-language");

  PipelineConfig overrides;
  // Override decoder with a different file AND add a conflicting session
  overrides.sessions["decoder"] = PipelineConfig::Session{
      .file = "custom_decoder.onnx",
      .role = "custom_decoder",  // Different role than preset
  };
  overrides.sessions["vision"] = PipelineConfig::Session{
      .file = "custom_vision.onnx",
      .role = "custom_vision",
  };
  // Override state with specific patterns
  overrides.state.kv_cache.format = "cross";
  overrides.state.kv_cache.past_key_pattern = "custom_past.%d.key";
  overrides.state.position_ids.strategy = "mrope_3d";

  ApplyOverrides(base, overrides);

  // Sessions should be replaced by overrides
  EXPECT_EQ(base.sessions.at("decoder").file, "custom_decoder.onnx");
  EXPECT_EQ(base.sessions.at("decoder").role, "custom_decoder");
  EXPECT_EQ(base.sessions.at("vision").file, "custom_vision.onnx");
  // Embedding should be preserved from preset
  EXPECT_TRUE(base.sessions.count("embedding"));
  // State fields should reflect overrides
  EXPECT_EQ(base.state.kv_cache.format.value_or(""), "cross");
  EXPECT_EQ(base.state.kv_cache.past_key_pattern.value_or(""), "custom_past.%d.key");
  EXPECT_EQ(base.state.position_ids.strategy.value_or(""), "mrope_3d");
  // Non-overridden state fields from preset should be preserved
  EXPECT_FALSE(base.state.kv_cache.past_value_pattern.has_value());
}

TEST(PipelineEdgeCases, LargeFlowManySessions) {
  // Stress test: 10 sessions in a chain
  PipelineConfig config;
  for (int i = 0; i < 10; ++i) {
    std::string name = "session_" + std::to_string(i);
    config.sessions[name] = PipelineConfig::Session{
        .file = name + ".onnx",
        .role = i == 9 ? "decoder" : "processor",
    };
    config.flow.push_back(PipelineConfig::FlowStep{
        .run = name,
        .when = i == 9 ? "always" : "prompt",
    });
    // Wire each session to the next
    if (i > 0) {
      std::string prev = "session_" + std::to_string(i - 1);
      config.dataflow.push_back(PipelineConfig::DataflowWire{
          .from_session = prev,
          .from_output = "output_" + std::to_string(i - 1),
          .to_session = name,
          .to_input = "input_" + std::to_string(i),
      });
    }
  }

  EXPECT_NO_THROW(ValidatePipelineConfig(config));
  EXPECT_EQ(config.sessions.size(), 10u);
  EXPECT_EQ(config.flow.size(), 10u);
  EXPECT_EQ(config.dataflow.size(), 9u);
}

TEST(PipelineEdgeCases, DiamondDataflowTopology) {
  // Diamond: A → B, A → C, B → D, C → D
  PipelineConfig config;
  config.sessions["a"] = PipelineConfig::Session{.file = "a.onnx"};
  config.sessions["b"] = PipelineConfig::Session{.file = "b.onnx"};
  config.sessions["c"] = PipelineConfig::Session{.file = "c.onnx"};
  config.sessions["d"] = PipelineConfig::Session{.file = "d.onnx", .role = "decoder"};

  config.flow.push_back(PipelineConfig::FlowStep{.run = "a", .when = "once"});
  config.flow.push_back(PipelineConfig::FlowStep{.run = "b", .when = "prompt"});
  config.flow.push_back(PipelineConfig::FlowStep{.run = "c", .when = "prompt"});
  config.flow.push_back(PipelineConfig::FlowStep{.run = "d", .when = "always"});

  config.dataflow.push_back(PipelineConfig::DataflowWire{
      .from_session = "a", .from_output = "out1",
      .to_session = "b", .to_input = "in1"});
  config.dataflow.push_back(PipelineConfig::DataflowWire{
      .from_session = "a", .from_output = "out2",
      .to_session = "c", .to_input = "in1"});
  config.dataflow.push_back(PipelineConfig::DataflowWire{
      .from_session = "b", .from_output = "features_b",
      .to_session = "d", .to_input = "features_b"});
  config.dataflow.push_back(PipelineConfig::DataflowWire{
      .from_session = "c", .from_output = "features_c",
      .to_session = "d", .to_input = "features_c"});

  EXPECT_NO_THROW(ValidatePipelineConfig(config));
  EXPECT_EQ(config.dataflow.size(), 4u);
}

TEST(PipelineEdgeCases, DataflowUnknownToSessionThrows) {
  PipelineConfig config;
  config.sessions["a"] = PipelineConfig::Session{.file = "a.onnx"};
  config.dataflow.push_back(PipelineConfig::DataflowWire{
      .from_session = "a",
      .from_output = "out",
      .to_session = "nonexistent",
      .to_input = "in",
  });

  EXPECT_THROW(ValidatePipelineConfig(config), std::runtime_error);
}

TEST(PipelineEdgeCases, SessionWithEmptyFile) {
  // A session with an empty file string is valid at schema level
  // (runtime will fail when loading, but schema validation shouldn't reject it)
  PipelineConfig config;
  config.sessions["decoder"] = PipelineConfig::Session{.file = "", .role = "decoder"};
  config.flow.push_back(PipelineConfig::FlowStep{.run = "decoder", .when = "always"});

  EXPECT_NO_THROW(ValidatePipelineConfig(config));
}

TEST(PipelineEdgeCases, DuplicateSessionInFlow) {
  // Same session referenced multiple times in flow (valid — runs multiple times)
  PipelineConfig config;
  config.sessions["decoder"] = PipelineConfig::Session{.file = "model.onnx"};
  config.flow.push_back(PipelineConfig::FlowStep{.run = "decoder", .when = "prompt"});
  config.flow.push_back(PipelineConfig::FlowStep{.run = "decoder", .when = "always"});

  EXPECT_NO_THROW(ValidatePipelineConfig(config));
}

TEST(PipelineEdgeCases, EmptyLoopStringIsValid) {
  // Empty loop string means "no looping" — should be valid
  PipelineConfig config;
  config.sessions["decoder"] = PipelineConfig::Session{.file = "model.onnx"};
  config.flow.push_back(PipelineConfig::FlowStep{
      .run = "decoder",
      .when = "always",
      .loop = "",
  });

  EXPECT_NO_THROW(ValidatePipelineConfig(config));
}

TEST(PipelineEdgeCases, OverrideDataflowReplacesEntirely) {
  auto base = GetPreset("vision-language");
  ASSERT_EQ(base.dataflow.size(), 2u);

  PipelineConfig overrides;
  overrides.dataflow.push_back(PipelineConfig::DataflowWire{
      .from_session = "vision",
      .from_output = "custom_features",
      .to_session = "decoder",
      .to_input = "custom_input",
  });

  ApplyOverrides(base, overrides);
  ASSERT_EQ(base.dataflow.size(), 1u);  // Replaced, not appended
  EXPECT_EQ(base.dataflow[0].from_output, "custom_features");
}

TEST(PipelineEdgeCases, OverrideKVCachePatterns) {
  auto base = GetPreset("autoregressive-decoder");
  EXPECT_FALSE(base.state.kv_cache.past_key_pattern.has_value());

  PipelineConfig overrides;
  overrides.state.kv_cache.past_key_pattern = "custom_past.%d.key";
  overrides.state.kv_cache.present_key_pattern = "custom_present.%d.key";
  overrides.state.kv_cache.past_value_pattern = "custom_past.%d.value";
  overrides.state.kv_cache.present_value_pattern = "custom_present.%d.value";

  ApplyOverrides(base, overrides);
  EXPECT_EQ(base.state.kv_cache.past_key_pattern.value_or(""), "custom_past.%d.key");
  EXPECT_EQ(base.state.kv_cache.present_key_pattern.value_or(""), "custom_present.%d.key");
  EXPECT_EQ(base.state.kv_cache.past_value_pattern.value_or(""), "custom_past.%d.value");
  EXPECT_EQ(base.state.kv_cache.present_value_pattern.value_or(""), "custom_present.%d.value");
}

TEST(V1Translator, VLMKVCachePatternsPropagate) {
  // VLM configs should also propagate KV cache patterns
  auto v1 = MakeMinimalV1Config("phi3v");
  v1.model.vision.filename = "vision.onnx";
  v1.model.embedding.filename = "embed.onnx";
  v1.model.decoder.inputs.past_key_names = "past_key_values.%d.key";
  v1.model.decoder.outputs.present_key_names = "present.%d.key";
  auto pipeline = TranslateV1Config(v1);

  EXPECT_EQ(pipeline.state.kv_cache.past_key_pattern.value_or(""), "past_key_values.%d.key");
  EXPECT_EQ(pipeline.state.kv_cache.present_key_pattern.value_or(""), "present.%d.key");
}

TEST(V1Translator, FaraGetsSpecialPositionStrategy) {
  // fara is also Qwen25VL family — should get mrope_3d
  auto v1 = MakeMinimalV1Config("fara");
  v1.model.vision.filename = "vision.onnx";
  v1.model.embedding.filename = "embed.onnx";
  auto pipeline = TranslateV1Config(v1);

  EXPECT_EQ(pipeline.state.position_ids.strategy.value_or(""), "mrope_3d");
}

}  // namespace Generators::test
