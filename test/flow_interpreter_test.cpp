// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Unit tests for PR 3: FlowInterpreter and multi-session pipeline support.

#include "generators.h"
#include "pipeline_config_schema.h"
#include "pipeline_presets.h"
#include "models/flow_interpreter.h"

#include <gtest/gtest.h>
#include <map>
#include <string>

namespace Generators::test {

// ============================================================================
// FlowInterpreter — flow partitioning tests
// ============================================================================

TEST(FlowInterpreter, DecoderOnlyIsNotMultiSession) {
  auto config = GetPreset("autoregressive-decoder");
  FlowInterpreter interpreter(config);

  EXPECT_FALSE(interpreter.IsMultiSession());
  EXPECT_TRUE(interpreter.init_steps().empty());
  ASSERT_EQ(interpreter.step_steps().size(), 1u);
  EXPECT_EQ(interpreter.step_steps()[0].run, "decoder");
}

TEST(FlowInterpreter, VLMIsMultiSession) {
  auto config = GetPreset("vision-language");
  FlowInterpreter interpreter(config);

  EXPECT_TRUE(interpreter.IsMultiSession());
}

TEST(FlowInterpreter, VLMInitStepsPartitioned) {
  auto config = GetPreset("vision-language");
  FlowInterpreter interpreter(config);

  // Vision runs on init only, embedding runs on init only
  ASSERT_EQ(interpreter.init_steps().size(), 2u);
  EXPECT_EQ(interpreter.init_steps()[0].run, "vision");
  EXPECT_EQ(interpreter.init_steps()[0].when, "init");
  EXPECT_EQ(interpreter.init_steps()[1].run, "embedding");
  EXPECT_EQ(interpreter.init_steps()[1].when, "init");
}

TEST(FlowInterpreter, VLMStepStepsPartitioned) {
  auto config = GetPreset("vision-language");
  FlowInterpreter interpreter(config);

  // Decoder runs every step
  ASSERT_EQ(interpreter.step_steps().size(), 1u);
  EXPECT_EQ(interpreter.step_steps()[0].run, "decoder");
  EXPECT_EQ(interpreter.step_steps()[0].when, "step");
}

TEST(FlowInterpreter, EncoderDecoderPartitioning) {
  auto config = GetPreset("encoder-decoder");
  FlowInterpreter interpreter(config);

  EXPECT_TRUE(interpreter.IsMultiSession());

  // Encoder runs once (goes to init_steps)
  ASSERT_EQ(interpreter.init_steps().size(), 1u);
  EXPECT_EQ(interpreter.init_steps()[0].run, "encoder");
  EXPECT_EQ(interpreter.init_steps()[0].when, "init");

  // Decoder runs every step
  ASSERT_EQ(interpreter.step_steps().size(), 1u);
  EXPECT_EQ(interpreter.step_steps()[0].run, "decoder");
}

// ============================================================================
// FlowInterpreter — dataflow wiring (stateless — intermediates passed in)
// ============================================================================

TEST(FlowInterpreter, StoreAndRetrieveViaMap) {
  auto config = GetPreset("vision-language");
  FlowInterpreter interpreter(config);

  // Intermediates are now owned by the State, not the interpreter.
  // Tests pass an external map to GetWiredInputs.
  std::map<std::string, OrtValue*> intermediates;
  int dummy_value = 42;
  auto* fake_ort_value = reinterpret_cast<OrtValue*>(&dummy_value);

  intermediates["vision.image_features"] = fake_ort_value;

  auto wired = interpreter.GetWiredInputs("embedding", intermediates);
  ASSERT_GE(wired.size(), 1u);
  EXPECT_EQ(wired[0].first, "image_features");
  EXPECT_EQ(wired[0].second, fake_ort_value);
}

TEST(FlowInterpreter, GetWiredInputsFromDataflow) {
  auto config = GetPreset("vision-language");
  FlowInterpreter interpreter(config);

  std::map<std::string, OrtValue*> intermediates;
  int dummy = 42;
  auto* fake_value = reinterpret_cast<OrtValue*>(&dummy);
  intermediates["vision.image_features"] = fake_value;

  // The VLM preset has a wire: vision.image_features -> embedding.image_features
  auto wired = interpreter.GetWiredInputs("embedding", intermediates);
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

  std::map<std::string, OrtValue*> intermediates;
  int dummy = 99;
  auto* fake_value = reinterpret_cast<OrtValue*>(&dummy);
  intermediates["embedding.inputs_embeds"] = fake_value;

  // The VLM preset has a wire: embedding.inputs_embeds -> decoder.inputs_embeds
  auto wired = interpreter.GetWiredInputs("decoder", intermediates);
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

  std::map<std::string, OrtValue*> intermediates;  // empty
  auto wired = interpreter.GetWiredInputs("embedding", intermediates);
  EXPECT_TRUE(wired.empty());
}

TEST(FlowInterpreter, GetWiredInputsForSessionWithNoWires) {
  auto config = GetPreset("vision-language");
  FlowInterpreter interpreter(config);

  std::map<std::string, OrtValue*> intermediates;
  intermediates["something.tensor"] = reinterpret_cast<OrtValue*>(0x1234);

  // Vision has no incoming wires in the VLM preset
  auto wired = interpreter.GetWiredInputs("vision", intermediates);
  EXPECT_TRUE(wired.empty());
}

TEST(FlowInterpreter, PromptOnlySessionsIdentified) {
  auto config = GetPreset("vision-language");
  FlowInterpreter interpreter(config);

  // Vision and embedding are init-only (not in step_steps)
  const auto& init_only = interpreter.init_only_sessions();
  EXPECT_TRUE(init_only.count("vision"));
  EXPECT_TRUE(init_only.count("embedding"));
  EXPECT_FALSE(init_only.count("decoder"));
}

// ============================================================================
// FlowInterpreter — custom pipeline config
// ============================================================================

TEST(FlowInterpreter, CustomFlowWithMixedWhenValues) {
  PipelineConfig config;
  config.sessions["encoder"] = {.file = "encoder.onnx", .role = "encoder"};
  config.sessions["projector"] = {.file = "proj.onnx", .role = "embedding"};
  config.sessions["decoder"] = {.file = "model.onnx", .role = "decoder"};

  config.flow.push_back({.run = "encoder", .when = "init"});
  config.flow.push_back({.run = "projector", .when = "init"});
  config.flow.push_back({.run = "decoder", .when = "step"});

  FlowInterpreter interpreter(config);

  EXPECT_TRUE(interpreter.IsMultiSession());

  // "init" goes to init_steps
  ASSERT_EQ(interpreter.init_steps().size(), 2u);
  EXPECT_EQ(interpreter.init_steps()[0].run, "encoder");
  EXPECT_EQ(interpreter.init_steps()[1].run, "projector");

  // "step" goes to step_steps
  ASSERT_EQ(interpreter.step_steps().size(), 1u);
  EXPECT_EQ(interpreter.step_steps()[0].run, "decoder");
}

TEST(FlowInterpreter, DataflowAccessors) {
  PipelineConfig config;
  config.sessions["a"] = {.file = "a.onnx", .role = "encoder"};
  config.sessions["b"] = {.file = "b.onnx", .role = "decoder"};
  config.flow.push_back({.run = "a", .when = "init"});
  config.flow.push_back({.run = "b", .when = "step"});
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
