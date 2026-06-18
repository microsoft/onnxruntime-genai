// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Portions of this file consist of AI generated content.

// Unit tests for the PipelineFlow interpreter (issue #2114, PR3). These exercise the flow/dataflow
// resolution logic directly on hand-built Config::Pipeline objects, so they do NOT require any model
// download or ONNX fixture (the runnable model fixtures here are single-decoder and never reach this
// multi-stage executor). They verify:
//   * stage classification into init/step/final phases,
//   * explicit dataflow[] override wiring,
//   * the load-time guardrails: dataflow cycle rejection and the max-10-stage limit.

#include <string>
#include <gtest/gtest.h>

#include "generators.h"
#include "models/decoder_only_pipeline.h"

namespace {

using Generators::Config;
using Generators::PipelineFlow;

// Adds a decoder.pipeline[] stage named `name` plus a matching flow step with the given `when`.
void AddStage(Config& config, const std::string& name, const std::string& when,
              std::vector<std::string> inputs = {}, std::vector<std::string> outputs = {}) {
  Config::Model::Decoder::PipelineModel stage;
  stage.model_id = name;
  stage.filename = name + ".onnx";
  stage.inputs = std::move(inputs);
  stage.outputs = std::move(outputs);
  config.model.decoder.pipeline.push_back(std::move(stage));

  Config::Pipeline::FlowStep step;
  step.run = name;
  step.when = when;
  config.pipeline.flow.push_back(std::move(step));
}

}  // namespace

// A flow with init/step/final stages resolves each stage into the expected phase, and only the
// "final" stage is reported as final.
TEST(PipelineFlowTests, ClassifiesInitStepFinalPhases) {
  Config config;
  AddStage(config, "vision", "init");
  AddStage(config, "decoder", "step");
  AddStage(config, "vocoder", "final");

  PipelineFlow flow{config};

  EXPECT_EQ(flow.PhaseForStage(0), PipelineFlow::Phase::Init);
  EXPECT_EQ(flow.PhaseForStage(1), PipelineFlow::Phase::Step);
  EXPECT_EQ(flow.PhaseForStage(2), PipelineFlow::Phase::Final);

  EXPECT_FALSE(flow.IsFinal(0));
  EXPECT_FALSE(flow.IsFinal(1));
  EXPECT_TRUE(flow.IsFinal(2));

  EXPECT_TRUE(flow.HasFinalStages());
}

// A flow with no "final" step reports no final stages and defaults unreferenced stages to Step.
TEST(PipelineFlowTests, NoFinalStagesByDefault) {
  Config config;
  AddStage(config, "decoder", "step");

  PipelineFlow flow{config};

  EXPECT_FALSE(flow.HasFinalStages());
  EXPECT_FALSE(flow.IsFinal(0));
  // Out-of-range stage ids default to Step (and are never final).
  EXPECT_EQ(flow.PhaseForStage(99), PipelineFlow::Phase::Step);
  EXPECT_FALSE(flow.IsFinal(99));
}

// Explicit dataflow[] wiring is exposed for (consuming session, input) and maps to the producing
// tensor (which can differ in name from the consuming input).
TEST(PipelineFlowTests, ExplicitDataflowOverrideWiring) {
  Config config;
  AddStage(config, "vision", "init", /*inputs*/ {}, /*outputs*/ {"image_features"});
  AddStage(config, "embedding", "init", /*inputs*/ {"image_embeds"}, /*outputs*/ {"inputs_embeds"});
  AddStage(config, "decoder", "step", /*inputs*/ {"inputs_embeds"}, /*outputs*/ {});

  // vision.image_features feeds embedding.image_embeds (names differ -> needs an explicit wire).
  config.pipeline.dataflow.push_back({"vision.image_features", "embedding.image_embeds"});

  PipelineFlow flow{config};

  const std::string* src = flow.ExplicitSource("embedding", "image_embeds");
  ASSERT_NE(src, nullptr);
  EXPECT_EQ(*src, "image_features");

  // No wire targets the decoder input, so auto-match (nullptr here) governs it.
  EXPECT_EQ(flow.ExplicitSource("decoder", "inputs_embeds"), nullptr);
  // A wire is keyed by both session and input; a wrong session must not match.
  EXPECT_EQ(flow.ExplicitSource("decoder", "image_embeds"), nullptr);
}

// A cyclic dataflow graph is rejected at construction (issue §3.2 guardrail).
TEST(PipelineFlowTests, RejectsDataflowCycle) {
  Config config;
  AddStage(config, "a", "step", /*inputs*/ {"b_out"}, /*outputs*/ {"a_out"});
  AddStage(config, "b", "step", /*inputs*/ {"a_out"}, /*outputs*/ {"b_out"});

  // a -> b -> a forms a cycle.
  config.pipeline.dataflow.push_back({"a.a_out", "b.a_out"});
  config.pipeline.dataflow.push_back({"b.b_out", "a.b_out"});

  EXPECT_THROW({ PipelineFlow flow{config}; }, std::runtime_error);
}

// An acyclic dataflow graph (a diamond) is accepted.
TEST(PipelineFlowTests, AcceptsAcyclicDataflowDiamond) {
  Config config;
  AddStage(config, "a", "init");
  AddStage(config, "b", "step");
  AddStage(config, "c", "step");
  AddStage(config, "d", "step");

  // a -> b, a -> c, b -> d, c -> d (diamond, acyclic).
  config.pipeline.dataflow.push_back({"a.x", "b.x"});
  config.pipeline.dataflow.push_back({"a.y", "c.y"});
  config.pipeline.dataflow.push_back({"b.z", "d.z"});
  config.pipeline.dataflow.push_back({"c.w", "d.w"});

  EXPECT_NO_THROW({ PipelineFlow flow{config}; });
}

// More than 10 stages is rejected at construction (issue §3.2 guardrail).
TEST(PipelineFlowTests, RejectsMoreThanTenStages) {
  Config config;
  for (int i = 0; i < 11; ++i) {
    AddStage(config, "stage" + std::to_string(i), "step");
  }

  EXPECT_THROW({ PipelineFlow flow{config}; }, std::runtime_error);
}

// Exactly 10 stages is accepted (boundary).
TEST(PipelineFlowTests, AcceptsExactlyTenStages) {
  Config config;
  for (int i = 0; i < 10; ++i) {
    AddStage(config, "stage" + std::to_string(i), "step");
  }

  EXPECT_NO_THROW({ PipelineFlow flow{config}; });
}
