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

#include <array>
#include <cmath>
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

// --- PR-C: intermediate hidden-state dataflow edges (design §5/§4e) ---------------------------

// The configured intermediate hidden-state activation is exposed with the right shape from the
// tiny fixture. This is the EAGLE/MTP prerequisite: a draft module coupled to the target's hidden
// state needs the executor to read that activation out of an intermediate pipeline stage.
TEST(SpeculativeDecodingTests, HiddenStateEdgeExposedWithShape) {
  const std::vector<int32_t> prompt{1, 2, 3, 4};

  auto target = LoadModel("speculative-tiny/target");
  auto params = Generators::CreateGeneratorParams(*target);
  params->search.max_length = 24;
  auto generator = Generators::CreateGenerator(*target, *params);
  generator->AppendTokens(Generators::cpu_span<const int32_t>(prompt.data(), prompt.size()));

  // After prefill, the hidden state covers all prompt positions: [batch, prompt_len, hidden_size].
  std::array<int64_t, 3> shape{};
  auto hidden = generator->GetHiddenStates(shape);
  ASSERT_GT(hidden.size(), 0u) << "hidden-state edge was not exposed";
  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], static_cast<int64_t>(prompt.size()));
  EXPECT_EQ(shape[2], 8);  // hidden_size from the fixture config
  EXPECT_EQ(static_cast<int64_t>(hidden.size()), shape[0] * shape[1] * shape[2]);

  // After a token-gen forward pass, the hidden state is for one position: [1, 1, hidden_size].
  // The first GenerateNextToken samples from the prefill logits; the second triggers a single-token
  // forward pass whose hidden state we read here.
  generator->GenerateNextToken();
  generator->GenerateNextToken();
  auto hidden_step = generator->GetHiddenStates(shape);
  ASSERT_GT(hidden_step.size(), 0u);
  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], 1);
  EXPECT_EQ(shape[2], 8);
  auto cpu = hidden_step.CopyDeviceToCpu();
  for (float v : cpu) EXPECT_TRUE(std::isfinite(v)) << "hidden state contains non-finite values";
}

// A model without a configured hidden-state output exposes nothing (empty span, additive default).
TEST(SpeculativeDecodingTests, HiddenStateEdgeAbsentWhenNotConfigured) {
  const std::vector<int32_t> prompt{1, 2, 3, 4};

  auto draft = LoadModel("speculative-tiny/draft");  // draft config declares no hidden_states output
  auto params = Generators::CreateGeneratorParams(*draft);
  params->search.max_length = 24;
  auto generator = Generators::CreateGenerator(*draft, *params);
  generator->AppendTokens(Generators::cpu_span<const int32_t>(prompt.data(), prompt.size()));

  std::array<int64_t, 3> shape{1, 1, 1};
  auto hidden = generator->GetHiddenStates(shape);
  EXPECT_EQ(hidden.size(), 0u);
  EXPECT_EQ(shape[0], 0);
  EXPECT_EQ(shape[1], 0);
  EXPECT_EQ(shape[2], 0);
}

// The EAGLE-style config declares a hidden-state dataflow edge (target.hidden_states ->
// eagle_draft.prev_hidden) plus the roles producer. Schema must parse the edge for a roles producer.
TEST(SpeculativeDecodingTests, HiddenStateEdgeSchemaParsesEagleDataflow) {
  Generators::Config config(fs::path(MODEL_PATH "speculative-tiny/target-eagle"), "");

  ASSERT_TRUE(config.pipeline.present);
  EXPECT_EQ(config.model.decoder.outputs.hidden_states, "hidden_states");

  ASSERT_EQ(config.pipeline.dataflow.size(), 1u);
  EXPECT_EQ(config.pipeline.dataflow[0].from, "target.hidden_states");
  EXPECT_EQ(config.pipeline.dataflow[0].to, "eagle_draft.prev_hidden");

  ASSERT_TRUE(config.pipeline.roles.has_value());
  ASSERT_TRUE(config.pipeline.roles->draft.has_value());
  EXPECT_EQ(*config.pipeline.roles->draft, "eagle_draft");
}

// An EAGLE-style config that declares the hidden-state edge and exposes the activation each step
// must still decode token-for-token identically to plain greedy on the target: reading the
// hidden-state edge is additive and never perturbs the committed output.
TEST(SpeculativeDecodingTests, HiddenStateEdgeGreedyMatchesBaseline) {
  const std::vector<int32_t> prompt{1, 2, 3, 4};
  const int max_length = 24;

  auto eagle = LoadModel("speculative-tiny/target-eagle");
  auto params = Generators::CreateGeneratorParams(*eagle);
  params->search.max_length = max_length;
  auto generator = Generators::CreateGenerator(*eagle, *params);
  generator->AppendTokens(Generators::cpu_span<const int32_t>(prompt.data(), prompt.size()));

  // Drive decoding while consuming the hidden-state edge each step (as an EAGLE draft would).
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
    std::array<int64_t, 3> shape{};
    auto hidden = generator->GetHiddenStates(shape);
    ASSERT_GT(hidden.size(), 0u) << "hidden-state edge must stay exposed throughout decoding";
    EXPECT_EQ(shape[2], 8);
  }
  auto seq = generator->GetSequence(0).CopyDeviceToCpu();
  std::vector<int32_t> out(seq.begin(), seq.end());

  auto target = LoadModel("speculative-tiny/target");  // identical weights
  auto baseline = GreedyBaseline(target, prompt, max_length);
  EXPECT_EQ(out, baseline) << "consuming the hidden-state edge must not change greedy output";
}

// --- PR-C: token tree -> linear-K fallback (design §11.2) -------------------------------------

// medusa_choices (the static token-tree topology) parses into the struct so the executor can see
// the requested tree even though it degrades to the linear-K fallback.
TEST(SpeculativeDecodingTests, TreeMedusaChoicesParsed) {
  Generators::Config config(fs::path(MODEL_PATH "speculative-tiny/target-tree"), "");

  ASSERT_TRUE(config.pipeline.strategy.has_value());
  ASSERT_TRUE(config.pipeline.strategy->tree.has_value());
  const auto& tree = *config.pipeline.strategy->tree;
  EXPECT_EQ(tree.topology, "medusa_choices");
  EXPECT_EQ(tree.max_nodes, 7);
  EXPECT_EQ(tree.max_depth, 3);
  ASSERT_EQ(tree.medusa_choices.size(), 6u);
  EXPECT_EQ(tree.medusa_choices[0], (std::vector<int>{0}));
  EXPECT_EQ(tree.medusa_choices[1], (std::vector<int>{0, 0}));
  EXPECT_EQ(tree.medusa_choices[2], (std::vector<int>{0, 1}));
  EXPECT_EQ(tree.medusa_choices[4], (std::vector<int>{1, 0}));
  EXPECT_EQ(tree.medusa_choices[5], (std::vector<int>{2}));
}

// When a token tree is requested but the decoder graph cannot express a tree-attention mask, the
// executor degrades to the verified best-linear-chain path (linear-K fallback). The §10 invariant
// must survive: output still equals plain greedy on the target, and the fallback is reported.
TEST(SpeculativeDecodingTests, TreeDegradesToLinearKGreedyMatchesBaseline) {
  const std::vector<int32_t> prompt{1, 2, 3, 4};
  const int max_length = 24;

  auto target_tree = LoadModel("speculative-tiny/target-tree");
  auto draft = LoadModel("speculative-tiny/draft");

  Generators::SpeculativeDecoder decoder(target_tree, draft, max_length);
  decoder.AppendTokens(std::span<const int32_t>(prompt.data(), prompt.size()));
  auto speculative = decoder.Generate();

  EXPECT_TRUE(decoder.tree_linear_k_fallback())
      << "a tree config must report that it degraded to the linear-K fallback";

  auto target = LoadModel("speculative-tiny/target");  // identical weights
  auto baseline = GreedyBaseline(target, prompt, max_length);
  EXPECT_EQ(speculative, baseline)
      << "tree -> linear-K fallback must still equal greedy decoding on the target";
  EXPECT_GE(decoder.verify_passes(), 1);
}

#endif  // !USE_DML
