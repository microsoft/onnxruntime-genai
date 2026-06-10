// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Portions of this file consist of AI generated content.

// Tests for the speculative-decoding executor (issue #2114 v2.1, PR-B).
//
// The executor implements the vanilla draft-target `speculative` flow strategy: a draft model
// proposes K tokens, the target verifies all K in a single forward pass, the longest greedy prefix
// is accepted, and the per-role KV caches are rolled back via PR-A's RewindTo on the first mismatch.
//
// Correctness invariant (design §10): under greedy acceptance the committed output MUST equal plain
// greedy decoding on the target, token-for-token. These tests assert exactly that on tiny, real
// ONNX draft+target fixtures (test/models/speculative-tiny), covering both the reject+rewind path
// (distinct draft weights) and the all-accepted multi-token-advance path (draft == target).

#include <memory>
#include <span>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "span.h"
#include "generators.h"
#include "search.h"
#include "constrained_logits_processor.h"
#include "models/model.h"
#include "models/speculative_decoder.h"

#include "test_utils.h"

#if !USE_DML

namespace {

std::shared_ptr<Generators::Model> LoadModel(const std::string& relative_path) {
  return Generators::CreateModel(Generators::GetOrtEnv(), (MODEL_PATH + relative_path).c_str());
}

// Plain greedy decoding on the target model -- the reference the speculative loop must reproduce.
std::vector<int32_t> GreedyBaseline(const std::shared_ptr<Generators::Model>& target,
                                    const std::vector<int32_t>& prompt, int max_length) {
  auto params = Generators::CreateGeneratorParams(*target);
  params->search.max_length = max_length;
  auto generator = Generators::CreateGenerator(*target, *params);
  generator->AppendTokens(Generators::cpu_span<const int32_t>(prompt.data(), prompt.size()));
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }
  auto seq = generator->GetSequence(0).CopyDeviceToCpu();
  return std::vector<int32_t>(seq.begin(), seq.end());
}

}  // namespace

// Config-only: the v2.1 `strategy`/`roles` block parses into Config::Pipeline. No ONNX load.
TEST(SpeculativeDecodingTests, SchemaParsesStrategyAndRoles) {
  Generators::Config config(fs::path(MODEL_PATH "speculative-tiny/target"), "");

  ASSERT_TRUE(config.pipeline.present);
  ASSERT_TRUE(config.pipeline.strategy.has_value());
  const auto& strategy = *config.pipeline.strategy;
  EXPECT_EQ(strategy.kind, "speculative");
  EXPECT_EQ(strategy.acceptance, "greedy");
  EXPECT_EQ(strategy.num_speculative_tokens, 4);
  EXPECT_EQ(strategy.draft.producer, "draft_model");
  ASSERT_TRUE(strategy.draft.session.has_value());
  EXPECT_EQ(*strategy.draft.session, "draft");
  EXPECT_EQ(strategy.verify.session, "target");
  EXPECT_FALSE(strategy.tree.has_value());

  ASSERT_TRUE(config.pipeline.roles.has_value());
  ASSERT_TRUE(config.pipeline.roles->target.has_value());
  ASSERT_TRUE(config.pipeline.roles->draft.has_value());
  EXPECT_EQ(*config.pipeline.roles->target, "target");
  EXPECT_EQ(*config.pipeline.roles->draft, "draft");
}

// Distinct draft weights: the draft frequently disagrees with the target, so the loop must reject
// and roll back -- yet the committed output must still equal greedy decoding on the target.
TEST(SpeculativeDecodingTests, GreedyMatchesBaselineDistinctDraft) {
  const std::vector<int32_t> prompt{1, 2, 3, 4};
  const int max_length = 24;

  auto target = LoadModel("speculative-tiny/target");
  auto draft = LoadModel("speculative-tiny/draft");

  Generators::SpeculativeDecoder decoder(target, draft, max_length);
  decoder.AppendTokens(std::span<const int32_t>(prompt.data(), prompt.size()));
  auto speculative = decoder.Generate();

  auto baseline = GreedyBaseline(target, prompt, max_length);

  EXPECT_GT(speculative.size(), prompt.size()) << "speculative decoding produced no new tokens";
  EXPECT_EQ(speculative, baseline)
      << "speculative output must be token-for-token identical to greedy decoding on the target";
  EXPECT_GE(decoder.verify_passes(), 1);
}

// Draft == target: every proposal is accepted, so each step commits K+1 tokens from a single verify
// pass. This proves genuine variable-tokens-per-step advancement (not just a 1-token fallback) while
// the output remains identical to the greedy baseline.
TEST(SpeculativeDecodingTests, AllAcceptedWhenDraftEqualsTarget) {
  const std::vector<int32_t> prompt{1, 2, 3, 4};
  const int max_length = 24;

  auto target = LoadModel("speculative-tiny/target");

  Generators::SpeculativeDecoder decoder(target, target, max_length);
  decoder.AppendTokens(std::span<const int32_t>(prompt.data(), prompt.size()));
  auto speculative = decoder.Generate();

  auto baseline = GreedyBaseline(target, prompt, max_length);

  EXPECT_EQ(speculative, baseline);
  // With an identical draft, more tokens are committed than verify passes are run -- the defining
  // speedup property of speculative decoding.
  EXPECT_GT(decoder.committed_token_count(), decoder.verify_passes());
  EXPECT_GT(decoder.accepted_draft_count(), 0);
}

#endif  // !USE_DML
