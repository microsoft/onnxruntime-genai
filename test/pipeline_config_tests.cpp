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

// Loads a shipped example config from examples/pipeline-config/ (outside test/models/ so it is not
// gitignored). Used by the ExamplePipelineConfigs tests below to prove every shipped example actually
// parses, lowers and routes with the current build.
Generators::Config LoadExample(const std::string& relative_path) {
  return Generators::Config(fs::path(std::string(EXAMPLES_PATH) + "pipeline-config/" + relative_path),
                            /*json_overlay*/ "");
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

// ---------------------------------------------------------------------------------------------------
// Shipped examples (examples/pipeline-config/). These prove that every documented example config under
// examples/pipeline-config/ actually parses + lowers with THIS build and routes to the model class the
// README claims. The plugin escape-hatch example (04) is intentionally NOT exercised here: its ON-path
// (USE_GENAI_PLUGINS=ON) is not linked into this build, so it is a documented syntax example only.
// ---------------------------------------------------------------------------------------------------

// Example 1 -- preset usage: the smallest possible v2 decoder, selecting a built-in preset by name.
TEST(ExamplePipelineConfigs, PresetDecoder) {
  auto config = LoadExample("01-preset-decoder");

  EXPECT_EQ(config.version, 2);
  EXPECT_TRUE(config.pipeline.present);
  ASSERT_TRUE(config.pipeline.extends.has_value());
  EXPECT_EQ(*config.pipeline.extends, "autoregressive-decoder");

  // The preset supplies the flow; the config only named the session file.
  ASSERT_EQ(config.pipeline.flow.size(), 1u);
  EXPECT_EQ(config.pipeline.flow[0].run, "decoder");
  EXPECT_EQ(config.pipeline.flow[0].when, "step");

  // Pure-v2 lowering: decoder file + context_length (from generation.max_length).
  EXPECT_EQ(config.model.decoder.filename, "model.onnx");
  EXPECT_EQ(config.model.context_length, 4096);

  EXPECT_EQ(Generators::ClassifyStructuralRoute(config), Generators::ModelRoute::DecoderOnly);
}

// Example 2 -- explicit multi-stage dataflow: an encoder/decoder graph written out by hand (no
// preset), demonstrating the init/step phases, explicit dataflow wiring and a frozen cross cache.
TEST(ExamplePipelineConfigs, ExplicitEncoderDecoder) {
  auto config = LoadExample("02-explicit-encoder-decoder");

  EXPECT_EQ(config.version, 2);
  EXPECT_FALSE(config.pipeline.extends.has_value());

  // Two stages, acyclic, well under the 10-stage guard.
  ASSERT_EQ(config.pipeline.flow.size(), 2u);
  EXPECT_EQ(config.pipeline.flow[0].run, "encoder");
  EXPECT_EQ(config.pipeline.flow[0].when, "init");
  EXPECT_EQ(config.pipeline.flow[1].run, "decoder");
  EXPECT_EQ(config.pipeline.flow[1].when, "step");
  ASSERT_TRUE(config.pipeline.flow[1].cross_attention_from.has_value());
  EXPECT_EQ(*config.pipeline.flow[1].cross_attention_from, "encoder");

  // Explicit dataflow wiring.
  ASSERT_EQ(config.pipeline.dataflow.size(), 1u);
  EXPECT_EQ(config.pipeline.dataflow[0].from, "encoder.encoder_hidden_states");
  EXPECT_EQ(config.pipeline.dataflow[0].to, "decoder.encoder_hidden_states");

  // Frozen cross-attention cache.
  ASSERT_TRUE(config.pipeline.state.cross_cache.has_value());
  ASSERT_TRUE(config.pipeline.state.cross_cache->source.has_value());
  EXPECT_EQ(*config.pipeline.state.cross_cache->source, "encoder");
  EXPECT_TRUE(config.pipeline.state.cross_cache->frozen);

  // Lowering populated both session filenames; the encoder session makes this an encoder-decoder route.
  EXPECT_EQ(config.model.encoder.filename, "encoder.onnx");
  EXPECT_EQ(config.model.decoder.filename, "decoder.onnx");
  EXPECT_EQ(Generators::ClassifyStructuralRoute(config), Generators::ModelRoute::Whisper);
}

// Example 3 -- VLM per-image flow: the Qwen-VL style vision pipeline (loop "per_image" + 3D mRoPE).
TEST(ExamplePipelineConfigs, VlmPerImage) {
  auto config = LoadExample("03-vlm-per-image");

  EXPECT_EQ(config.version, 2);
  ASSERT_TRUE(config.pipeline.extends.has_value());
  EXPECT_EQ(*config.pipeline.extends, "vision-language");

  ASSERT_EQ(config.pipeline.flow.size(), 3u);
  EXPECT_EQ(config.pipeline.flow[0].run, "vision");
  EXPECT_EQ(config.pipeline.flow[0].when, "init");
  EXPECT_EQ(config.pipeline.flow[0].loop, "per_image");

  EXPECT_EQ(config.pipeline.state.position_ids.strategy, "mrope_3d");
  ASSERT_TRUE(config.pipeline.state.position_ids.grid_source.has_value());
  EXPECT_EQ(*config.pipeline.state.position_ids.grid_source, "vision.image_grid_thw");

  ASSERT_EQ(config.pipeline.dataflow.size(), 2u);
  EXPECT_EQ(config.pipeline.dataflow[0].from, "vision.image_features");
  EXPECT_EQ(config.pipeline.dataflow[1].to, "decoder.inputs_embeds");

  // A vision session present (without a decoder.pipeline) routes to the general multimodal class.
  EXPECT_EQ(config.model.vision.filename, "vision.onnx");
  EXPECT_EQ(Generators::ClassifyStructuralRoute(config), Generators::ModelRoute::MultiModal);
}

// Example 5 -- v1 -> v2 side by side: the legacy v1 gpt2 config and its hand-written v2 equivalent must
// both load, and both must derive/produce the same structural route (Gpt, via the combined KV cache).
TEST(ExamplePipelineConfigs, V1ToV2Equivalence) {
  auto v1 = LoadExample("05-v1-to-v2/v1");
  EXPECT_EQ(v1.version, 1);
  EXPECT_EQ(v1.model.type, "gpt2");
  // The translator derives a pipeline view from v1 without altering model.* ...
  EXPECT_TRUE(v1.pipeline.present);
  EXPECT_EQ(v1.pipeline.state.kv_cache.format, "combined");
  EXPECT_EQ(Generators::ClassifyStructuralRoute(v1), Generators::ModelRoute::Gpt);

  auto v2 = LoadExample("05-v1-to-v2/v2");
  EXPECT_EQ(v2.version, 2);
  ASSERT_TRUE(v2.pipeline.extends.has_value());
  EXPECT_EQ(*v2.pipeline.extends, "autoregressive-decoder");
  EXPECT_EQ(v2.pipeline.state.kv_cache.format, "combined");
  EXPECT_EQ(v2.pipeline.state.kv_cache.past_key_pattern, "past_%d");
  EXPECT_EQ(v2.pipeline.state.kv_cache.present_key_pattern, "present_%d");

  // Same structural decision from both schema versions: the equivalence the README claims.
  EXPECT_EQ(Generators::ClassifyStructuralRoute(v2), Generators::ClassifyStructuralRoute(v1));
  EXPECT_EQ(Generators::ClassifyStructuralRoute(v2), Generators::ModelRoute::Gpt);
}
