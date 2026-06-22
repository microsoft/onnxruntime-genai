// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <cstdint>
#include <deque>
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

  // Sampling settings worked out by the base class and passed to Propose. greedy -> accept by
  // matching the target's argmax (empty probs); else sample from the top-k / top-p distribution.
  struct SamplingConfig {
    bool greedy{true};
    int top_k{};
    float top_p{};
    float temperature{1.0f};
  };

  void Step(Generator& g) final;
  SpeculativeStats GetStats() const final;
  void Reset() final;

 protected:
  // Produce K candidate tokens. seed_length = sequence length at start.
  virtual Proposal Propose(Generator& g, int K, int seed_length,
                           const SamplingConfig& sampling) = 0;

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

  // Reusable one-hot logit row used to commit a single token.
  DeviceSpan<float> onehot_buf_;
};

}  // namespace Generators
