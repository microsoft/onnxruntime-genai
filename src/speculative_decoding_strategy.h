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
  void PrepareForAppend(Generator& g) final;
  bool TryGetExternalLogits(Generator& g, DeviceSpan<float>& logits) final;
  void PrepareForSetLogits(Generator& g) final;

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
  std::size_t target_runs_{};
  float total_propose_ms_{};
  float total_target_ms_{};
  float total_reanchor_ms_{};
  std::size_t reanchor_runs_{};

  // Runtime vocab-size sanity check.
  bool vocab_check_done_{false};

  // Shared RNG for draft sampling + accept/correction/bonus draws.
  std::mt19937 rng_;
  bool rng_seeded_{false};

  // Grammar-forced tokens carried from a prior guidance round, read by Propose so it can place them
  // first in the verify batch (the grammar is already advanced past them).
  const std::deque<int32_t>& GuidanceFFCarry() const { return ff_carry_; }

 private:
  // Each Step emits one token. RunRound does a whole round's compute up front and buffers the
  // committed tokens; DrainOne hands them out one per call; FinalizeRound runs the deferred
  // re-anchor once the buffer empties (so hitting EOS mid-round skips the wasted target pass).
  void RunRound(Generator& g);
  void DrainOne(Generator& g);
  void FinalizeRound(Generator& g);
  void EmitToken(Generator& g, int32_t tok);
  void ClearPendingExternalLogits();

  // Runs one guidance round - check the draft's tokens against the grammar-masked target, tell the
  // grammar about each committed token, and add any tokens the grammar forces. Handles greedy and
  // sampling. Runs instead of the normal batched path.
  void RunGuidanceRound(Generator& g, const Proposal& proposal, int seed_length, int K,
                        float propose_ms);

  // Cleans up after a guidance round and returns the target's logits for the next position. Keeps
  // the accepted tokens already in the cache and feeds the rest back one at a time; also refreshes
  // the draft's saved logits.
  DeviceSpan<float> FinalizeGuidanceRound(Generator& g);

  // Rewinds both inner caches to floor and replays the committed tokens from there to the current
  // length; lines both caches back up with the committed sequence. Used when AppendTokens resumes.
  DeviceSpan<float> ReplayCommittedTail(Generator& g, int floor);

  // Committed but not emitted tokens for the round.
  std::deque<int32_t> pending_;

  // Raw target rows corresponding to accepted tokens in pending_. These let get_logits inspect the
  // next-token logits without discarding speculative lookahead or advancing RNG/model state.
  DeviceSpan<float> pending_target_logits_;
  int pending_target_logits_row0_{};
  int current_target_logits_row_{-1};
  int emitted_direct_tokens_{};
  int cached_direct_tokens_{};

  // Round context saved by RunRound.
  Proposal saved_proposal_;
  int32_t saved_final_token_{};
  int saved_n_direct_{};
  int saved_seed_length_{};
  int saved_K_{};
  bool reanchor_pending_{false};

  // Guidance round context - the committed token set (accepted prefix + correction + fast-forward
  // tokens), saved so FinalizeGuidanceRound can re-anchor the caches.
  std::vector<int32_t> saved_committed_;

  // Grammar-forced tokens the round could not fit in its K budget, carried to the next round and
  // emitted first (the grammar is already advanced past them).
  std::deque<int32_t> ff_carry_;

  // How many leading committed tokens were part of the target verify batch (so their KV is already
  // built). FinalizeGuidanceRound only replays committed tokens past this point.
  int saved_verify_prefix_{};

  // The last verify row for the committed prefix, reused as the next round's pos0 when nothing needs
  // replaying (avoids an extra target pass).
  std::vector<float> saved_last_row_;

  // Set while a round has left the inner KV caches out of sync with the committed sequence (mid round/deferred fold).
  bool round_dirty_{false};

  // True when the round just run was a guidance round.
  bool guidance_round_{false};

  // Re-anchor fold: instead of giving the round's committed token its own target forward, 
  // we tack it onto the front of the next round's verify batch. Saves
  // one full target run per round.
  // pending_anchor_token_ = the token waiting to ride the next verify (empty = none waiting).
  // is_multiple_tokens_ = true if the target returns one logits row per token (required for the fold).
  bool is_multiple_tokens_{false};
  std::optional<int32_t> pending_anchor_token_{};

  // Reusable fp32 scratch for the verify-logits cast (fp16/bf16 -> fp32), mirroring Logits. 
  std::unique_ptr<OrtValue> verify_logits_fp32_;
};

}  // namespace Generators
