// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Speculative decoding executor (issue #2114 v2.1, PR-B).
//
// Implements the vanilla draft-target `speculative` flow strategy described in
// docs/pipeline-config-v2.1-design.md (§3, §4): a small "draft" model proposes K tokens, the
// "target" model verifies all K in a single forward pass, the longest greedy-matching prefix is
// accepted, and the per-role KV caches are rolled back on the first mismatch using PR-A's
// State::RewindTo (via Generator::RewindToLength).
//
// Scope (PR-B, extended in PR-C):
//   * Linear-K drafting (no token tree). When the config requests a token tree (strategy.tree),
//     PR-C degrades it to the verified best-linear-chain path (linear-K fallback) rather than
//     faking tree-attention -- the in-tree decoder graph cannot express a tree-attention mask
//     without a model-side 2D additive mask input (design §11.2; see the constructor). This is the
//     design's stated degradation and preserves the greedy-equivalence invariant.
//   * Greedy acceptance (argmax). `rejection_sampling`/`typical` are parsed but not yet driven
//     here (see the deferred list in the PR-B decision note).
//   * Multi-session roles are realized as two independent Generators (target + draft), each owning
//     its own session + KV cache. Folding both roles into a single DecoderOnlyPipelineState with a
//     role->cache map (the design's preferred shape) is deferred; this composition is functionally
//     equivalent and reuses the existing, battle-tested per-model machinery.
//
// Correctness invariant (design §10): under greedy acceptance the committed sequence is exactly the
// target model's greedy argmax sequence, token-for-token identical to plain greedy decoding on the
// target. The accompanying gtest asserts this against a non-speculative baseline.

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "../generators.h"

namespace Generators {

struct SpeculativeDecoder {
  // `target_model` must carry a pipeline.strategy of kind "speculative" (K and the acceptance rule
  // are read from it). `draft_model` produces the draft tokens. `max_length` bounds the committed
  // sequence for both roles. Greedy decoding only.
  SpeculativeDecoder(std::shared_ptr<const Model> target_model,
                     std::shared_ptr<const Model> draft_model,
                     int max_length);

  // Seed both roles with the prompt.
  void AppendTokens(std::span<const int32_t> prompt);

  ~SpeculativeDecoder();

  // Run outer speculative steps until an EOS token is committed or max_length is reached. Returns
  // the full committed sequence (prompt + generated).
  std::vector<int32_t> Generate();

  // Diagnostics from the most recent Generate(): a genuine speculative loop commits strictly more
  // tokens than it runs target verify passes whenever any draft token is accepted.
  int verify_passes() const { return verify_passes_; }
  int committed_token_count() const { return committed_token_count_; }
  int accepted_draft_count() const { return accepted_draft_count_; }

  // True when the config requested a token tree (strategy.tree) but the executor degraded it to the
  // verified best-linear-chain path (linear-K fallback). See the constructor for why true
  // tree-attention is deferred (needs a model-side 2D mask input). Output remains greedy-equivalent.
  bool tree_linear_k_fallback() const { return tree_linear_k_fallback_; }

 private:
  int ArgMax(std::span<const float> logits_row) const;
  bool IsEos(int32_t token) const;

  std::shared_ptr<const Model> target_model_;
  std::shared_ptr<const Model> draft_model_;

  std::shared_ptr<GeneratorParams> target_params_;
  std::shared_ptr<GeneratorParams> draft_params_;

  std::unique_ptr<Generator> target_;
  std::unique_ptr<Generator> draft_;

  int k_{5};                 // num_speculative_tokens
  int vocab_size_{};
  int max_length_{};
  std::vector<int> eos_tokens_;

  std::vector<int32_t> sequence_;  // committed tokens (prompt + generated)

  int verify_passes_{0};
  int committed_token_count_{0};
  int accepted_draft_count_{0};
  bool tree_linear_k_fallback_{false};
};

}  // namespace Generators
