// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <cstdint>
#include <deque>
#include <optional>
#include <random>
#include <vector>
#include "decoding_strategy.h"
#include "smartptrs.h"
#include "speculative_sampling.h"

namespace Generators {

struct Generator;
struct SpeculativeStats;

// SpeculativeDecodingStrategy
// Base class for speculative decoding: a small draft model proposes K tokens, the big target
// model verifies them in one pass, matching tokens are accepted, and the target is re-anchored
// for the next round. Subclasses implement Propose (produce K tokens) and Advance (update the
// draft's state); all shared logic (RNG, target rewind, stats, vocab check) lives here.
struct SpeculativeDecodingStrategy : DecodingStrategy {
  // Draft model's output. If probs is empty, accept a token when it equals the target's argmax
  // (greedy). Otherwise probs[i] is the draft's probability distribution for token i, and the
  // token is accepted with probability min(1, p_target/p_draft).
  struct Proposal {
    std::vector<int32_t> tokens;
    std::vector<std::vector<float>> probs;
  };

  // Sampling settings are read directly from the canonical config (search params) and the
  // Generator's sampling method inside Propose/RunRound.

  void Step(Generator& g) final;
  SpeculativeStats GetStats() const final;
  void Reset() final;

 protected:
  // Produce K candidate tokens. seed_length = sequence length at start. Sampling settings are
  // read from the canonical config (g.search_->params_->search) and g.IsGreedySampling().
  virtual Proposal Propose(Generator& g, int K, int seed_length) = 0;

  // Update the draft model's own state (its KV cache, n-gram tables, ...) after the base class
  // commits the accepted tokens. n_direct = how many proposed tokens were accepted (excludes the
  // correction/bonus token).
  virtual void Advance(Generator& g,
                       const Proposal& proposal,
                       int n_direct,
                       int32_t final_token,
                       int seed_length) = 0;

  // Stats accumulators.
  std::size_t rounds_{};
  std::size_t draft_proposed_{};
  std::size_t draft_accepted_{};
  std::size_t corrections_{};
  std::size_t bonuses_{};
  float total_propose_ms_{};
  float total_target_ms_{};
  float total_reanchor_ms_{};
  std::size_t reanchor_runs_{};

  // Runtime vocab-size sanity check.
  bool vocab_check_done_{false};

  // Shared RNG for draft sampling + accept/correction/bonus draws.
  std::mt19937 rng_;
  bool rng_seeded_{false};

 private:
  // Each Step emits one token. RunRound does a whole round's compute up front and buffers the
  // committed tokens; DrainOne hands them out one per call; FinalizeRound runs the deferred
  // re-anchor once the buffer empties (so hitting EOS mid-round skips the wasted target pass).
  void RunRound(Generator& g);
  void DrainOne(Generator& g);
  void FinalizeRound(Generator& g);
  void EmitToken(Generator& g, int32_t tok);

  // Committed but not emitted tokens for the round.
  std::deque<int32_t> pending_;

  // Round context saved by RunRound.
  Proposal saved_proposal_;
  int32_t saved_final_token_{};
  int saved_n_direct_{};
  int saved_seed_length_{};
  int saved_K_{};
  bool reanchor_pending_{false};

  // Re-anchor fold: instead of giving the round's committed token its own target forward, 
  // we tack it onto the front of the next round's verify batch. Saves
  // one full target run per round.
  // pending_anchor_token_ = the token waiting to ride the next verify (empty = none waiting).
  // is_multiple_tokens_ = true if the target returns one logits row per token (required for the fold).
  bool is_multiple_tokens_{false};
  std::optional<int32_t> pending_anchor_token_{};

  // Reusable one-hot logit row used to commit a single token.
  DeviceSpan<float> onehot_buf_;

  // Reusable fp32 scratch for the verify-logits cast (fp16/bf16 -> fp32), mirroring Logits. 
  std::unique_ptr<OrtValue> verify_logits_fp32_;
};

}  // namespace Generators
