// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Portions of this file consist of AI generated content.

// Unit tests for the Pipeline-as-Config schema v2 parse surface and the v1->v2 translator
// (issue #2114, PR1). These verify that:
//   * v2 configs parse into Config::Pipeline with version == 2,
//   * `extends` presets are resolved (inherited when omitted, replaced when explicit),
//   * legacy v1 configs derive the expected Config::Pipeline WITHOUT changing config.model.*.

#include <string>
#include <gtest/gtest.h>

#include "generators.h"

#include "test_utils.h"

namespace {

Generators::Config LoadConfig(const std::string& relative_path) {
  return Generators::Config(fs::path(std::string(MODEL_PATH) + relative_path), /*json_overlay*/ "");
}

const Generators::Config::Pipeline::Session* FindSession(const Generators::Config::Pipeline& pipeline,
                                                         const std::string& name) {
  for (const auto& session : pipeline.sessions) {
    if (session.name == name) {
      return &session;
    }
  }
  return nullptr;
}

}  // namespace

// A minimal v2 decoder config (issue §4.1 shape) parses into Config::Pipeline with version == 2.
TEST(PipelineConfigTests, V2MinimalDecoderParses) {
  auto config = LoadConfig("pipeline-v2-decoder");

  EXPECT_EQ(config.version, 2);
  EXPECT_TRUE(config.pipeline.present);
  ASSERT_TRUE(config.pipeline.extends.has_value());
  EXPECT_EQ(*config.pipeline.extends, "autoregressive-decoder");

  ASSERT_EQ(config.pipeline.sessions.size(), 1u);
  EXPECT_EQ(config.pipeline.sessions[0].name, "decoder");
  EXPECT_EQ(config.pipeline.sessions[0].file, "model.onnx");

  // v2 lowering fills the legacy decoder filename so existing consumers keep working.
  EXPECT_EQ(config.model.decoder.filename, "model.onnx");
}

// PURE v2 (issue #2114 §4.1): no legacy `model` block. The top-level tokens/generation/metadata
// sections lower into config.model.* / config.search.*, and context_length is derived from
// generation.max_length so the config constructs past Config::Config validation.
TEST(PipelineConfigTests, V2PureDecoderLowersTokensAndGeneration) {
  auto config = LoadConfig("pipeline-v2-decoder");

  // metadata.model_type seeds the legacy model.type (used by CreatePipeline's fallback dispatch).
  EXPECT_EQ(config.model.type, "qwen2");

  // tokens.* lower into model token ids.
  ASSERT_EQ(config.model.eos_token_id.size(), 1u);
  EXPECT_EQ(config.model.eos_token_id[0], 151645);
  EXPECT_EQ(config.model.pad_token_id, 0);

  // generation.* lower into search.*; context_length is derived from generation.max_length.
  EXPECT_EQ(config.search.max_length, 4096);
  EXPECT_EQ(config.model.context_length, 4096);
  EXPECT_FLOAT_EQ(config.search.temperature, 0.7f);
}

// A config with only `extends` (no explicit flow) inherits the preset's flow wholesale.
TEST(PipelineConfigTests, V2PresetExpansionInheritsFlow) {
  auto config = LoadConfig("pipeline-v2-decoder");

  ASSERT_EQ(config.pipeline.flow.size(), 1u);
  EXPECT_EQ(config.pipeline.flow[0].run, "decoder");
  EXPECT_EQ(config.pipeline.flow[0].when, "step");
  EXPECT_EQ(config.pipeline.flow[0].loop, "batched");
}

// Explicit `flow`/`state` in the config replaces the preset defaults; omitted arrays (dataflow) are
// inherited from the preset.
TEST(PipelineConfigTests, V2ExtendsOverride) {
  auto config = LoadConfig("pipeline-v2-vlm-override");

  EXPECT_EQ(config.version, 2);
  ASSERT_TRUE(config.pipeline.extends.has_value());
  EXPECT_EQ(*config.pipeline.extends, "vision-language");

  // Explicit flow replaces the preset default (preset's vision loop is "batched").
  ASSERT_EQ(config.pipeline.flow.size(), 3u);
  EXPECT_EQ(config.pipeline.flow[0].run, "vision");
  EXPECT_EQ(config.pipeline.flow[0].loop, "per_image");

  // Explicit position_ids strategy overrides the preset default.
  EXPECT_EQ(config.pipeline.state.position_ids.strategy, "mrope_3d");
  ASSERT_TRUE(config.pipeline.state.position_ids.grid_source.has_value());
  EXPECT_EQ(*config.pipeline.state.position_ids.grid_source, "vision.image_grid_thw");

  // dataflow was omitted in the config, so it is inherited from the vision-language preset.
  ASSERT_EQ(config.pipeline.dataflow.size(), 2u);
  EXPECT_EQ(config.pipeline.dataflow[0].from, "vision.image_features");
  EXPECT_EQ(config.pipeline.dataflow[0].to, "embedding.image_features");
  EXPECT_EQ(config.pipeline.dataflow[1].from, "embedding.inputs_embeds");
  EXPECT_EQ(config.pipeline.dataflow[1].to, "decoder.inputs_embeds");
}

// GOLDEN backward-compat: a legacy v1 gpt2 fixture derives the expected autoregressive-decoder
// pipeline AND leaves config.model.* untouched.
TEST(PipelineConfigTests, V1GoldenGpt2) {
  auto config = LoadConfig("hf-internal-testing/tiny-random-gpt2-fp32");

  // config.model.* must be exactly as parsed (no behavior change from PR1).
  EXPECT_EQ(config.version, 1);
  EXPECT_EQ(config.model.type, "gpt2");
  EXPECT_EQ(config.model.decoder.filename, "past.onnx");
  EXPECT_EQ(config.model.decoder.inputs.past_names, "past_%d");
  EXPECT_EQ(config.model.decoder.outputs.present_names, "present_%d");

  // Derived introspective pipeline.
  EXPECT_TRUE(config.pipeline.present);
  ASSERT_TRUE(config.pipeline.extends.has_value());
  EXPECT_EQ(*config.pipeline.extends, "autoregressive-decoder");

  ASSERT_EQ(config.pipeline.sessions.size(), 1u);
  EXPECT_EQ(config.pipeline.sessions[0].name, "decoder");
  EXPECT_EQ(config.pipeline.sessions[0].file, "past.onnx");

  ASSERT_EQ(config.pipeline.flow.size(), 1u);
  EXPECT_EQ(config.pipeline.flow[0].run, "decoder");
  EXPECT_EQ(config.pipeline.flow[0].when, "step");

  // gpt2 uses the combined KV cache format.
  EXPECT_EQ(config.pipeline.state.kv_cache.format, "combined");
  EXPECT_EQ(config.pipeline.state.kv_cache.past_key_pattern, "past_%d");
  EXPECT_EQ(config.pipeline.state.kv_cache.present_key_pattern, "present_%d");

  EXPECT_EQ(config.pipeline.state.position_ids.strategy, "default");
}

// GOLDEN backward-compat: a legacy v1 phi3-v (VLM) fixture derives a vision-language pipeline with
// vision/embedding/decoder sessions and the right dataflow wiring, without changing config.model.*.
TEST(PipelineConfigTests, V1GoldenPhi3V) {
  auto config = LoadConfig("phi3-v");

  EXPECT_EQ(config.version, 1);
  EXPECT_EQ(config.model.type, "phi3v");
  // config.model.* unchanged.
  EXPECT_EQ(config.model.decoder.filename, "dummy_text.onnx");
  EXPECT_EQ(config.model.vision.filename, "dummy_vision.onnx");
  EXPECT_EQ(config.model.embedding.filename, "dummy_embedding.onnx");

  ASSERT_TRUE(config.pipeline.extends.has_value());
  EXPECT_EQ(*config.pipeline.extends, "vision-language");

  // Sessions enumerated in order: vision, embedding, decoder (no speech for phi3v).
  ASSERT_EQ(config.pipeline.sessions.size(), 3u);
  EXPECT_EQ(config.pipeline.sessions[0].name, "vision");
  EXPECT_EQ(config.pipeline.sessions[1].name, "embedding");
  EXPECT_EQ(config.pipeline.sessions[2].name, "decoder");

  const auto* vision = FindSession(config.pipeline, "vision");
  ASSERT_NE(vision, nullptr);
  EXPECT_EQ(vision->file, "dummy_vision.onnx");

  // Flow: vision/embedding run at init, decoder runs each step. phi3v is not a Qwen-VL family
  // model, so the vision loop is "batched".
  ASSERT_EQ(config.pipeline.flow.size(), 3u);
  EXPECT_EQ(config.pipeline.flow[0].run, "vision");
  EXPECT_EQ(config.pipeline.flow[0].when, "init");
  EXPECT_EQ(config.pipeline.flow[0].loop, "batched");
  EXPECT_EQ(config.pipeline.flow[2].run, "decoder");
  EXPECT_EQ(config.pipeline.flow[2].when, "step");

  // Dataflow wires vision features into the embedding and embeddings into the decoder.
  ASSERT_EQ(config.pipeline.dataflow.size(), 2u);
  EXPECT_EQ(config.pipeline.dataflow[0].from, "vision.image_features");
  EXPECT_EQ(config.pipeline.dataflow[0].to, "embedding.image_features");
  EXPECT_EQ(config.pipeline.dataflow[1].from, "embedding.inputs_embeds");
  EXPECT_EQ(config.pipeline.dataflow[1].to, "decoder.inputs_embeds");

  // phi3v uses the standard separate KV cache and default (1D) position ids.
  EXPECT_EQ(config.pipeline.state.kv_cache.format, "separate");
  EXPECT_EQ(config.pipeline.state.position_ids.strategy, "default");
}
