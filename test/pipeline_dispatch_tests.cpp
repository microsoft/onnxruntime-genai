// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Portions of this file consist of AI generated content.

// Zero-regression gate for Pipeline-as-Config PR5 (issue #2114). PR5 converts the runtime model-class
// dispatch from model.type strings (ClassifyLegacyRoute, the pre-#2114 behavior) to config STRUCTURE
// (ClassifyStructuralRoute, used by CreatePipeline). These tests prove that, for EVERY fixture config
// under test/models/, the structural router selects the exact same concrete Model subclass as the
// legacy router. The Config constructor only parses genai_config.json, so this needs no ONNX download
// and asserts the dispatch DECISION directly -- the safest possible regression gate.

#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "generators.h"

#include "test_utils.h"

namespace {

using Generators::Config;
using Generators::ModelRoute;

Config LoadConfig(const std::string& relative_path) {
  return Config(fs::path(std::string(MODEL_PATH) + relative_path), /*json_overlay*/ "");
}

const char* RouteName(ModelRoute route) {
  switch (route) {
    case ModelRoute::Unsupported: return "Unsupported";
    case ModelRoute::DecoderOnly: return "DecoderOnly";
    case ModelRoute::Gpt: return "Gpt";
    case ModelRoute::LFM2: return "LFM2";
    case ModelRoute::DecoderOnlyPipeline: return "DecoderOnlyPipeline";
    case ModelRoute::Qwen2_5_VL_Pipeline: return "Qwen2_5_VL_Pipeline";
    case ModelRoute::MultiModal: return "MultiModal";
    case ModelRoute::Whisper: return "Whisper";
    case ModelRoute::Marian: return "Marian";
    case ModelRoute::NemotronSpeech: return "NemotronSpeech";
    case ModelRoute::ParakeetTdt: return "ParakeetTdt";
  }
  return "?";
}

struct Fixture {
  std::string path;
  ModelRoute expected;
};

// Every routable fixture currently under test/models/, with the concrete class each must dispatch to.
const std::vector<Fixture>& Fixtures() {
  static const std::vector<Fixture> fixtures = {
      {"hf-internal-testing/tiny-random-gpt2-fp32", ModelRoute::Gpt},
      {"hf-internal-testing/tiny-random-gpt2-fp32-cuda", ModelRoute::Gpt},
      {"hf-internal-testing/tiny-random-gpt2-fp16-cuda", ModelRoute::Gpt},
      {"hf-internal-testing/tiny-random-lfm2-fp32", ModelRoute::LFM2},
      {"hf-internal-testing/tiny-qwen35-cuda", ModelRoute::DecoderOnly},
      {"gemma4", ModelRoute::MultiModal},
      {"phi3-v", ModelRoute::MultiModal},
      {"multimodal-decoder-no-input-ids", ModelRoute::MultiModal},
      {"multimodal-decoder-with-input-ids", ModelRoute::MultiModal},
      {"qwen2-5-vl", ModelRoute::MultiModal},
      {"qwen2-5-vl-pipeline", ModelRoute::Qwen2_5_VL_Pipeline},
      {"qwen3-vl", ModelRoute::MultiModal},
      {"qwen3-5", ModelRoute::MultiModal},
      {"whisper", ModelRoute::Whisper},
      {"pipeline-model", ModelRoute::DecoderOnlyPipeline},
      {"pipeline-v2-decoder", ModelRoute::DecoderOnly},
      {"pipeline-v2-vlm-override", ModelRoute::MultiModal},
  };
  return fixtures;
}

}  // namespace

// THE GATE: for every fixture, the structural router and the legacy model.type router must agree.
// If this fails, PR5 has changed the concrete model class for an existing config -> a regression.
TEST(PipelineDispatchTests, StructuralMatchesLegacyForEveryFixture) {
  for (const auto& fixture : Fixtures()) {
    auto config = LoadConfig(fixture.path);
    const ModelRoute legacy = Generators::ClassifyLegacyRoute(config);
    const ModelRoute structural = Generators::ClassifyStructuralRoute(config);
    EXPECT_EQ(structural, legacy)
        << "Fixture '" << fixture.path << "' (model.type='" << config.model.type
        << "') routes structurally to " << RouteName(structural) << " but legacy routes to "
        << RouteName(legacy) << " -- PR5 regression.";
  }
}

// Pin the absolute expected class per fixture too, so a future change that drifts BOTH routers in the
// same wrong direction (which the equivalence test alone would not catch) is still flagged.
TEST(PipelineDispatchTests, StructuralRouteMatchesExpectedClass) {
  for (const auto& fixture : Fixtures()) {
    auto config = LoadConfig(fixture.path);
    const ModelRoute structural = Generators::ClassifyStructuralRoute(config);
    EXPECT_EQ(structural, fixture.expected)
        << "Fixture '" << fixture.path << "' expected " << RouteName(fixture.expected)
        << " but got " << RouteName(structural) << ".";
  }
}

// gpt2 is selected structurally by the combined-KV cache format (not by the "gpt2" string).
TEST(PipelineDispatchTests, Gpt2RoutesViaCombinedKvFormat) {
  auto config = LoadConfig("hf-internal-testing/tiny-random-gpt2-fp32");
  EXPECT_EQ(config.pipeline.state.kv_cache.format, "combined");
  EXPECT_EQ(Generators::ClassifyStructuralRoute(config), ModelRoute::Gpt);
}

// The Qwen-VL family carries the 3D mRoPE position strategy as a config signal (CP3), so CreatePositionInputs
// no longer needs model.type. (qwen2-5-vl has no decoder.pipeline, hence MultiModal not the pipeline variant.)
TEST(PipelineDispatchTests, QwenVlCarriesMropeStrategy) {
  auto config = LoadConfig("qwen2-5-vl");
  EXPECT_EQ(config.pipeline.state.position_ids.strategy, "mrope_3d");
}

// A plain VLM (phi3v) does NOT request the per-image vision loop, so CreateVisionState (CP4) keeps the
// batched base VisionState without consulting model.type.
TEST(PipelineDispatchTests, Phi3vUsesBatchedVisionLoop) {
  auto config = LoadConfig("phi3-v");
  bool found_vision = false;
  for (const auto& step : config.pipeline.flow) {
    if (step.run == "vision") {
      found_vision = true;
      EXPECT_EQ(step.loop, "batched");
      EXPECT_FALSE(step.variable_resolution);
    }
  }
  EXPECT_TRUE(found_vision);
}
