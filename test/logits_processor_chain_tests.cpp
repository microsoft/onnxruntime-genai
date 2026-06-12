// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Unit tests for the v2.1 ordered logit-processor / sampler chain (issue #2114, PR-D, design §6).
// These verify that:
//   * an ordered `generation.logits` chain parses into Config::Search::logits_processors in
//     declared order with its typed fields preserved (schema),
//   * the chain applies its ops in declared order so a masking op changes the deterministic greedy
//     argmax (order matters),
//   * with NO chain configured the output is byte-for-byte the legacy behavior (back-compat), and a
//     chain consisting of only a terminal `sample` op reproduces the legacy greedy result.

#include <array>
#include <cstring>
#include <string>
#include <gtest/gtest.h>

#include "generators.h"
#include "test_utils.h"

#include "span.h"
#define OGA_USE_SPAN 1
#include <ort_genai.h>

namespace {

Generators::Config LoadGpt2ConfigWithOverlay(const std::string& overlay) {
  return Generators::Config(
      fs::path(std::string(MODEL_PATH) + "hf-internal-testing/tiny-random-gpt2-fp32"), overlay);
}

}  // namespace

// (a) Schema: a multi-processor chain parses in declared order with typed fields preserved.
TEST(LogitsChainTests, SchemaParsesChainInOrder) {
  const std::string overlay = R"({
    "generation": {
      "logits": [
        { "op": "repetition_penalty", "value": 1.1 },
        { "op": "logit_bias", "map": { "50256": -100 } },
        { "op": "grammar", "backend": "llguidance" },
        { "op": "temperature", "value": 0.7 },
        { "op": "top_k", "value": 50 },
        { "op": "top_p", "value": 0.9 },
        { "op": "sample" }
      ]
    }
  })";

  auto config = LoadGpt2ConfigWithOverlay(overlay);
  const auto& chain = config.search.logits_processors;
  ASSERT_EQ(chain.size(), 7u);

  EXPECT_EQ(chain[0].op, "repetition_penalty");
  ASSERT_TRUE(chain[0].value.has_value());
  EXPECT_FLOAT_EQ(*chain[0].value, 1.1f);

  EXPECT_EQ(chain[1].op, "logit_bias");
  ASSERT_EQ(chain[1].bias.size(), 1u);
  EXPECT_EQ(chain[1].bias[0].first, 50256);
  EXPECT_FLOAT_EQ(chain[1].bias[0].second, -100.0f);

  EXPECT_EQ(chain[2].op, "grammar");
  ASSERT_TRUE(chain[2].backend.has_value());
  EXPECT_EQ(*chain[2].backend, "llguidance");

  EXPECT_EQ(chain[3].op, "temperature");
  ASSERT_TRUE(chain[3].value.has_value());
  EXPECT_FLOAT_EQ(*chain[3].value, 0.7f);

  EXPECT_EQ(chain[4].op, "top_k");
  ASSERT_TRUE(chain[4].int_value.has_value());
  EXPECT_EQ(*chain[4].int_value, 50);

  EXPECT_EQ(chain[5].op, "top_p");
  ASSERT_TRUE(chain[5].value.has_value());
  EXPECT_FLOAT_EQ(*chain[5].value, 0.9f);

  EXPECT_EQ(chain[6].op, "sample");

  // Back-compat: an absent chain leaves logits_processors empty.
  auto plain = LoadGpt2ConfigWithOverlay("");
  EXPECT_TRUE(plain.search.logits_processors.empty());
}

// (b) Order matters: a logit_bias op that masks the highest-scoring token, applied before the
// terminal greedy `sample`, deterministically shifts the argmax to the next-highest token.
TEST(LogitsChainTests, BiasMaskShiftsDeterministicArgmax) {
  // vocab_size 5, batch 1. Legacy greedy argmax is token 1 (score 0.6).
  std::vector<float> logits_cpu{0.1f, 0.6f, 0.3f, 0.1f, 0.1f};

  auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  config->Overlay(R"({ "model": { "vocab_size": 5 } })");
  // Chain masks token 1, then masks token 4, then greedily samples. Expected argmax -> token 2 (0.3).
  config->Overlay(R"({
    "generation": {
      "logits": [
        { "op": "logit_bias", "map": { "1": -100 } },
        { "op": "logit_bias", "map": { "4": -100 } },
        { "op": "sample" }
      ]
    }
  })");

  auto model = OgaModel::Create(*config);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 10);

  auto generator = OgaGenerator::Create(*model, *params);
  auto logits_tensor = OgaTensor::Create(logits_cpu.data(), std::array<int64_t, 2>{1LL, 5LL});
  generator->SetLogits(*logits_tensor);

  generator->GenerateNextToken();
  auto next_tokens = generator->GetNextTokens();
  ASSERT_EQ(next_tokens.size(), 1u);
  EXPECT_EQ(next_tokens[0], 2);
}

// (c) Back-compat: with NO chain configured, greedy decoding yields the legacy argmax; a chain made
// of only a terminal `sample` op reproduces that exact result (the default chain == today's behavior).
TEST(LogitsChainTests, BackCompatDefaultMatchesLegacy) {
  std::vector<float> logits_cpu{0.1f, 0.6f, 0.3f, 0.1f, 0.1f};

  auto run = [&](const char* chain_overlay) -> int32_t {
    auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
    config->Overlay(R"({ "model": { "vocab_size": 5 } })");
    if (chain_overlay)
      config->Overlay(chain_overlay);
    auto model = OgaModel::Create(*config);
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", 10);
    auto generator = OgaGenerator::Create(*model, *params);
    auto logits_tensor = OgaTensor::Create(logits_cpu.data(), std::array<int64_t, 2>{1LL, 5LL});
    generator->SetLogits(*logits_tensor);
    generator->GenerateNextToken();
    return generator->GetNextTokens()[0];
  };

  const int32_t legacy = run(nullptr);
  EXPECT_EQ(legacy, 1);  // argmax of the synthetic logits

  const int32_t chain_sample_only = run(R"({ "generation": { "logits": [ { "op": "sample" } ] } })");
  EXPECT_EQ(chain_sample_only, legacy);
}
