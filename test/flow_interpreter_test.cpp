// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Unit tests for PR 3: FlowInterpreter and multi-session pipeline support.

#include "generators.h"
#include "pipeline_config_schema.h"
#include "pipeline_presets.h"
#include "models/flow_interpreter.h"

#include <gtest/gtest.h>
#include <string>

namespace Generators::test {

// ============================================================================
// FlowInterpreter — flow partitioning tests
// ============================================================================

TEST(FlowInterpreter, DecoderOnlyIsNotMultiSession) {
  auto config = GetPreset("autoregressive-decoder");
  FlowInterpreter interpreter(config);

  EXPECT_FALSE(interpreter.IsMultiSession());
  EXPECT_TRUE(interpreter.prompt_steps().empty());
  ASSERT_EQ(interpreter.always_steps().size(), 1u);
  EXPECT_EQ(interpreter.always_steps()[0].run, "decoder");
}

TEST(FlowInterpreter, VLMIsMultiSession) {
  auto config = GetPreset("vision-language");
  FlowInterpreter interpreter(config);

  EXPECT_TRUE(interpreter.IsMultiSession());
}

TEST(FlowInterpreter, VLMPromptStepsPartitioned) {
  auto config = GetPreset("vision-language");
  FlowInterpreter interpreter(config);

  // Vision runs on prompt only, embedding runs on prompt only
  ASSERT_EQ(interpreter.prompt_steps().size(), 2u);
  EXPECT_EQ(interpreter.prompt_steps()[0].run, "vision");
  EXPECT_EQ(interpreter.prompt_steps()[0].when, "prompt");
  EXPECT_EQ(interpreter.prompt_steps()[1].run, "embedding");
  EXPECT_EQ(interpreter.prompt_steps()[1].when, "prompt");
}

TEST(FlowInterpreter, VLMAlwaysStepsPartitioned) {
  auto config = GetPreset("vision-language");
  FlowInterpreter interpreter(config);

  // Decoder runs always
  ASSERT_EQ(interpreter.always_steps().size(), 1u);
  EXPECT_EQ(interpreter.always_steps()[0].run, "decoder");
  EXPECT_EQ(interpreter.always_steps()[0].when, "always");
}

TEST(FlowInterpreter, EncoderDecoderPartitioning) {
  auto config = GetPreset("encoder-decoder");
  FlowInterpreter interpreter(config);

  EXPECT_TRUE(interpreter.IsMultiSession());

  // Encoder runs once (goes to prompt_steps)
  ASSERT_EQ(interpreter.prompt_steps().size(), 1u);
  EXPECT_EQ(interpreter.prompt_steps()[0].run, "encoder");
  EXPECT_EQ(interpreter.prompt_steps()[0].when, "once");

  // Decoder runs always
  ASSERT_EQ(interpreter.always_steps().size(), 1u);
  EXPECT_EQ(interpreter.always_steps()[0].run, "decoder");
}

// ============================================================================
// FlowInterpreter — intermediate tensor storage
// ============================================================================

TEST(FlowInterpreter, StoreAndRetrieveIntermediate) {
  auto config = GetPreset("vision-language");
  FlowInterpreter interpreter(config);

  // Create a dummy OrtValue pointer (we just need address tracking)
  int dummy_value = 42;
  auto* fake_ort_value = reinterpret_cast<OrtValue*>(&dummy_value);

  interpreter.StoreIntermediate("vision", "image_features", fake_ort_value);

  EXPECT_EQ(interpreter.GetIntermediate("vision", "image_features"),
            fake_ort_value);
  EXPECT_EQ(interpreter.GetIntermediate("vision", "nonexistent"), nullptr);
  EXPECT_EQ(interpreter.GetIntermediate("other", "image_features"), nullptr);
}

TEST(FlowInterpreter, ClearAllRemovesEverything) {
  auto config = GetPreset("vision-language");
  FlowInterpreter interpreter(config);

  int dummy1 = 1, dummy2 = 2;
  interpreter.StoreIntermediate("vision", "image_features",
                                reinterpret_cast<OrtValue*>(&dummy1));
  interpreter.StoreIntermediate("embedding", "inputs_embeds",
                                reinterpret_cast<OrtValue*>(&dummy2));

  interpreter.ClearAll();

  EXPECT_EQ(interpreter.GetIntermediate("vision", "image_features"), nullptr);
  EXPECT_EQ(interpreter.GetIntermediate("embedding", "inputs_embeds"), nullptr);
}

TEST(FlowInterpreter, ClearPromptIntermediatesSelectiveClean) {
  auto config = GetPreset("vision-language");
  FlowInterpreter interpreter(config);

  int dummy_vision = 1, dummy_decoder = 2;
  interpreter.StoreIntermediate("vision", "image_features",
                                reinterpret_cast<OrtValue*>(&dummy_vision));
  interpreter.StoreIntermediate("decoder", "some_output",
                                reinterpret_cast<OrtValue*>(&dummy_decoder));

  // Vision is prompt-only (appears in prompt_steps but not always_steps),
  // so its intermediates should be cleared.  Decoder is always-run, so
  // its intermediates should survive.
  interpreter.ClearPromptIntermediates();

  EXPECT_EQ(interpreter.GetIntermediate("vision", "image_features"), nullptr);
  EXPECT_EQ(interpreter.GetIntermediate("decoder", "some_output"),
            reinterpret_cast<OrtValue*>(&dummy_decoder));
}

// ============================================================================
// FlowInterpreter — dataflow wiring
// ============================================================================

TEST(FlowInterpreter, GetWiredInputsFromDataflow) {
  auto config = GetPreset("vision-language");
  FlowInterpreter interpreter(config);

  // Store a vision output
  int dummy = 42;
  auto* fake_value = reinterpret_cast<OrtValue*>(&dummy);
  interpreter.StoreIntermediate("vision", "image_features", fake_value);

  // The VLM preset has a wire: vision.image_features -> embedding.image_features
  auto wired = interpreter.GetWiredInputs("embedding");
  ASSERT_GE(wired.size(), 1u);

  bool found = false;
  for (const auto& [name, value] : wired) {
    if (name == "image_features" && value == fake_value) {
      found = true;
    }
  }
  EXPECT_TRUE(found) << "Expected wired input 'image_features' for embedding session";
}

TEST(FlowInterpreter, GetWiredInputsForDecoderFromEmbedding) {
  auto config = GetPreset("vision-language");
  FlowInterpreter interpreter(config);

  // Store an embedding output
  int dummy = 99;
  auto* fake_value = reinterpret_cast<OrtValue*>(&dummy);
  interpreter.StoreIntermediate("embedding", "inputs_embeds", fake_value);

  // The VLM preset has a wire: embedding.inputs_embeds -> decoder.inputs_embeds
  auto wired = interpreter.GetWiredInputs("decoder");
  ASSERT_GE(wired.size(), 1u);

  bool found = false;
  for (const auto& [name, value] : wired) {
    if (name == "inputs_embeds" && value == fake_value) {
      found = true;
    }
  }
  EXPECT_TRUE(found) << "Expected wired input 'inputs_embeds' for decoder session";
}

TEST(FlowInterpreter, GetWiredInputsReturnsEmptyWhenNoIntermediates) {
  auto config = GetPreset("vision-language");
  FlowInterpreter interpreter(config);

  // No intermediates stored yet — wired inputs should be empty
  auto wired = interpreter.GetWiredInputs("embedding");
  EXPECT_TRUE(wired.empty());
}

TEST(FlowInterpreter, GetWiredInputsForSessionWithNoWires) {
  auto config = GetPreset("vision-language");
  FlowInterpreter interpreter(config);

  // Vision has no incoming wires in the VLM preset
  auto wired = interpreter.GetWiredInputs("vision");
  EXPECT_TRUE(wired.empty());
}

// ============================================================================
// FlowInterpreter — custom pipeline config
// ============================================================================

TEST(FlowInterpreter, CustomFlowWithMixedWhenValues) {
  PipelineConfig config;
  config.sessions["encoder"] = {.file = "encoder.onnx", .role = "encoder"};
  config.sessions["projector"] = {.file = "proj.onnx", .role = "embedding"};
  config.sessions["decoder"] = {.file = "model.onnx", .role = "decoder"};

  config.flow.push_back({.run = "encoder", .when = "once"});
  config.flow.push_back({.run = "projector", .when = "prompt"});
  config.flow.push_back({.run = "decoder", .when = "always"});

  FlowInterpreter interpreter(config);

  EXPECT_TRUE(interpreter.IsMultiSession());

  // "once" and "prompt" go to prompt_steps
  ASSERT_EQ(interpreter.prompt_steps().size(), 2u);
  EXPECT_EQ(interpreter.prompt_steps()[0].run, "encoder");
  EXPECT_EQ(interpreter.prompt_steps()[1].run, "projector");

  // "always" goes to always_steps
  ASSERT_EQ(interpreter.always_steps().size(), 1u);
  EXPECT_EQ(interpreter.always_steps()[0].run, "decoder");
}

TEST(FlowInterpreter, DataflowAccessors) {
  PipelineConfig config;
  config.sessions["a"] = {.file = "a.onnx", .role = "encoder"};
  config.sessions["b"] = {.file = "b.onnx", .role = "decoder"};
  config.flow.push_back({.run = "a", .when = "once"});
  config.flow.push_back({.run = "b", .when = "always"});
  config.dataflow.push_back({
      .from_session = "a",
      .from_output = "hidden",
      .to_session = "b",
      .to_input = "encoder_out",
  });

  FlowInterpreter interpreter(config);

  ASSERT_EQ(interpreter.dataflow().size(), 1u);
  EXPECT_EQ(interpreter.dataflow()[0].from_session, "a");
  EXPECT_EQ(interpreter.dataflow()[0].from_output, "hidden");
  EXPECT_EQ(interpreter.dataflow()[0].to_session, "b");
  EXPECT_EQ(interpreter.dataflow()[0].to_input, "encoder_out");
}

}  // namespace Generators::test
