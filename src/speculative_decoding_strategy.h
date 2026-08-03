// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <cmath>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>
#include "decoding_strategy.h"
#include "smartptrs.h"
#include "speculative_sampling.h"
#include "speculative_stats.h"

namespace Generators {

struct Generator;
struct Model;
struct State;
struct SpeculativeStats;
struct ConstrainedLogitsProcessor;

// Learns committed-token throughput per K, probes only adjacent widths, and keeps a probe only
// when acceptance stays high and its throughput is not materially worse.
class AdaptiveKController {
 public:
  static constexpr int kAdaptiveMaxK = 16;

  AdaptiveKController(int fixed_k, int adaptive_min_k, bool enabled)
      : min_k_{enabled ? adaptive_min_k : fixed_k},
        max_k_{enabled ? kAdaptiveMaxK : fixed_k},
        enabled_{enabled},
        effective_k_{min_k_} {}

  int GetK() const {
    return effective_k_;
  }
  int MinimumK() const { return min_k_; }
  std::size_t Increases() const { return increases_; }
  std::size_t Decreases() const { return decreases_; }
  std::size_t Observations() const { return observations_; }
  std::size_t Probes() const { return probes_; }
  float CurrentThroughput() const {
    const Estimate& current = estimates_[static_cast<size_t>(effective_k_)];
    if (current.samples > 0)
      return current.Throughput();
    if (probe_origin_k_ != 0)
      return estimates_[static_cast<size_t>(probe_origin_k_)].Throughput();
    return 0.0f;
  }

  void RecordCompletedRound(int k, int evaluated, int accepted,
                            std::size_t committed_tokens, bool filled_proposal_budget,
                            float propose_ms, float target_ms) {
    const float total_ms = propose_ms + target_ms;
    if (!enabled_ || k != effective_k_ || evaluated <= 0 || committed_tokens == 0 ||
        !filled_proposal_budget || !std::isfinite(total_ms) || total_ms <= 0.0f)
      return;

    Estimate& estimate = estimates_[static_cast<size_t>(k)];
    estimate.Update(static_cast<float>(committed_tokens), total_ms,
                    static_cast<float>(accepted) / static_cast<float>(evaluated));
    observations_++;

    if (probe_origin_k_ != 0) {
      probe_observations_++;
      const Estimate& origin = estimates_[static_cast<size_t>(probe_origin_k_)];
      const bool probing_up = effective_k_ > probe_origin_k_;
      const bool severe_regression =
          origin.Throughput() > 0.0f &&
          estimate.Throughput() < origin.Throughput() * kSevereRegressionRatio;
      if (severe_regression) {
        FinishProbe(false);
      } else if (probe_observations_ >= kProbeSamples) {
        const bool throughput_safe =
            origin.Throughput() <= 0.0f ||
            estimate.Throughput() >= origin.Throughput() * kProbeRetentionRatio;
        const bool acceptance_safe =
            !probing_up || estimate.acceptance >= kHighAcceptanceThreshold;
        FinishProbe(throughput_safe && acceptance_safe);
      }
      return;
    }

    stable_observations_++;
    if (probe_cooldown_ > 0) {
      probe_cooldown_--;
      return;
    }
    if (estimate.samples < kMinSamples)
      return;

    if (estimate.acceptance < kLowAcceptanceThreshold && effective_k_ > min_k_) {
      StartProbe(effective_k_ - 1);
    } else if (estimate.acceptance >= kHighAcceptanceThreshold &&
               effective_k_ < max_k_ && stable_observations_ >= kMinSamples) {
      StartProbe(effective_k_ + 1);
    }
  }

  void Reset() {
    effective_k_ = min_k_;
    estimates_ = {};
    probe_origin_k_ = 0;
    probe_observations_ = 0;
    stable_observations_ = 0;
    probe_cooldown_ = 0;
  }

 private:
  static constexpr float kEwmaAlpha = 0.25f;
  static constexpr float kHighAcceptanceThreshold = 0.75f;
  static constexpr float kLowAcceptanceThreshold = 0.50f;
  static constexpr float kProbeRetentionRatio = 0.97f;
  static constexpr float kSevereRegressionRatio = 0.80f;
  static constexpr int kMinSamples = 2;
  static constexpr int kProbeSamples = 2;
  static constexpr int kSuccessfulProbeCooldown = 1;
  static constexpr int kRejectedProbeCooldown = 6;

  struct Estimate {
    float tokens{};
    float milliseconds{};
    float acceptance{};
    std::size_t samples{};

    void Update(float sample_tokens, float sample_ms, float sample_acceptance) {
      if (samples == 0) {
        tokens = sample_tokens;
        milliseconds = sample_ms;
        acceptance = sample_acceptance;
      } else {
        tokens += kEwmaAlpha * (sample_tokens - tokens);
        milliseconds += kEwmaAlpha * (sample_ms - milliseconds);
        acceptance += kEwmaAlpha * (sample_acceptance - acceptance);
      }
      samples++;
    }

    float Throughput() const {
      return milliseconds > 0.0f ? tokens / milliseconds : 0.0f;
    }
  };

  void MoveTo(int k) {
    if (k == effective_k_)
      return;
    if (k > effective_k_)
      increases_++;
    else
      decreases_++;
    effective_k_ = k;
  }

  void StartProbe(int candidate_k) {
    probe_origin_k_ = effective_k_;
    probe_observations_ = 0;
    stable_observations_ = 0;
    probes_++;
    MoveTo(candidate_k);
  }

  void FinishProbe(bool keep_candidate) {
    if (!keep_candidate)
      MoveTo(probe_origin_k_);
    probe_origin_k_ = 0;
    probe_observations_ = 0;
    stable_observations_ = 0;
    probe_cooldown_ =
        keep_candidate ? kSuccessfulProbeCooldown : kRejectedProbeCooldown;
  }

  int min_k_;
  int max_k_;
  bool enabled_;
  int effective_k_;
  std::array<Estimate, 17> estimates_{};
  int probe_origin_k_{};
  int probe_observations_{};
  int stable_observations_{};
  int probe_cooldown_{};
  std::size_t increases_{};
  std::size_t decreases_{};
  std::size_t observations_{};
  std::size_t probes_{};
};

class SpeculativeCooldownController {
 public:
  explicit SpeculativeCooldownController(bool enabled) : enabled_{enabled} {}

  void RecordCompletedRound(int k, int minimum_k, int evaluated, int accepted) {
    if (!enabled_ || evaluated <= 0)
      return;
    if (accepted > 0 || k != minimum_k) {
      consecutive_zero_accept_rounds_ = 0;
      return;
    }

    consecutive_zero_accept_rounds_++;
    if (consecutive_zero_accept_rounds_ == kFailureThreshold) {
      cooldown_remaining_ = 1;
      consecutive_zero_accept_rounds_ = 0;
      entries_++;
    }
  }

  bool ShouldRunStandardStep() const { return cooldown_remaining_ > 0; }

  void CompleteStandardStep() {
    if (cooldown_remaining_ == 0)
      throw std::runtime_error("Speculative cooldown completed without an active cooldown.");
    cooldown_remaining_--;
    steps_++;
  }

  void Reset() {
    consecutive_zero_accept_rounds_ = 0;
    cooldown_remaining_ = 0;
  }

  std::size_t Entries() const { return entries_; }
  std::size_t Steps() const { return steps_; }
  int Remaining() const { return cooldown_remaining_; }

 private:
  static constexpr int kFailureThreshold = 3;

  bool enabled_;
  int consecutive_zero_accept_rounds_{};
  int cooldown_remaining_{};
  std::size_t entries_{};
  std::size_t steps_{};
};

// SpeculativeDecodingStrategy
// Base class for speculative decoding: a small draft model proposes K tokens, the big target
// model verifies them in one pass, matching tokens are accepted, and the target is re-anchored
// for the next round. Subclasses implement Propose (produce K tokens) and Advance (update the
// draft's state); shared verification borrows the Generator-owned RNG.
struct SpeculativeDecodingStrategy : DecodingStrategy {
  enum class ProposalMode {
    kUnset,
    kGreedyMatch,
    kDraftSampling,
    kDeterministic,
  };

  // Proposer output. The mode defines how verification interprets the tokens; probability storage
  // is data for draft-model sampling, not an implicit behavior signal.
  struct Proposal {
    Proposal() = default;
    explicit Proposal(ProposalMode proposal_mode) : mode{proposal_mode} {}

    bool UsesDraftProbabilities() const { return mode == ProposalMode::kDraftSampling; }

    ProposalMode mode{ProposalMode::kUnset};
    std::vector<int32_t> tokens;
    std::vector<std::vector<float>> probs;
  };

  struct RoundState {
    enum class Phase {
      kIdle,
      kDraining,
      kFinalizing,
      kReconcilePending,
    };

    enum class Kind {
      kStandard,
      kGuidance,
    };

    // Normal: Idle/ReconcilePending -> Draining -> Finalizing -> Idle/ReconcilePending.
    // Interruption: Draining -> ReconcilePending -> Idle after committed tokens are replayed.
    bool IsActive() const { return phase == Phase::kDraining; }
    bool NeedsReconciliation() const { return phase != Phase::kIdle; }

    Phase phase{Phase::kIdle};
    Kind kind{Kind::kStandard};
    bool discarded{};

    std::deque<int32_t> pending;
    std::deque<std::mt19937> pending_rng_states;
    DeviceSpan<float> target_logits;
    int target_logits_row0{};
    int current_target_logits_row{-1};
    int emitted_direct_tokens{};
    int cached_direct_tokens{};

    Proposal proposal;
    int32_t final_token{};
    int n_direct{};
    int seed_length{};
    int k{};

    std::vector<int32_t> committed;
    int verify_prefix{};
    std::vector<float> last_row;

    int evaluated{};
    int accepted{};
    std::size_t emitted{};
    bool filled_proposal_budget{};
    float propose_ms{};
    float target_ms{};
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
  SpeculativeDecodingStrategy(State& target_state, const Model& target_model);

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

  // Keep proposer-specific state synchronized after the target cache is rewound and replayed.
  virtual void ReconcileProposer(Generator& g,
                                 int floor,
                                 std::span<const int32_t> committed,
                                 int committed_length,
                                 bool record_stats) = 0;

  // Guidance may commit correction/forced tokens that differ from the original proposal.
  virtual void FinalizeGuidanceProposer(Generator& g,
                                        int seed_length,
                                        int proposal_length,
                                        std::span<const int32_t> committed) = 0;

  virtual void ResetProposer() = 0;

  virtual void PopulateProposerStats(SpeculativeStats& stats) const {}

  virtual void RunRound(Generator& g);
  virtual RoundState::Phase FinalizeRound(Generator& g);

  RoundState& CurrentRound() { return round_; }
  void BeginRound(int K, int evaluated, int accepted, size_t queued,
                  bool formula_supported, bool filled_proposal_budget,
                  float propose_ms, float target_ms, RoundState::Kind kind);
  void ClearPendingExternalLogits();
  DeviceSpan<float> GetFloatVerifyLogits(OrtValue& logits, DeviceInterface& device);
  bool HasPendingAnchorToken() const { return pending_anchor_token_.has_value(); }
  int32_t TakePendingAnchorToken();
  void SetPendingAnchorToken(int32_t token) { pending_anchor_token_ = token; }

  State& target_state_;
  const Model& target_model_;

  // Stats accumulators.
  std::size_t rounds_{};
  std::size_t completed_rounds_{};
  std::size_t interrupted_rounds_{};
  std::size_t draft_proposed_{};
  std::size_t draft_evaluated_{};
  std::size_t draft_accepted_{};
  std::size_t corrections_{};
  std::size_t bonuses_{};
  std::size_t tokens_queued_{};
  std::size_t tokens_emitted_{};
  std::size_t tokens_discarded_{};
  std::size_t draft_runs_{};
  std::size_t target_runs_{};
  std::size_t target_verify_runs_{};
  std::size_t target_reconciliation_runs_{};
  std::size_t full_accept_rounds_{};
  std::size_t partial_accept_rounds_{};
  std::size_t zero_accept_rounds_{};
  float total_propose_ms_{};
  float total_target_ms_{};
  float total_reanchor_ms_{};
  float total_reconciliation_ms_{};
  std::size_t reanchor_runs_{};

  // Runtime vocab-size sanity check.
  bool vocab_check_done_{false};

  // Proposal-side fast-forward composition shared by draft-model and n-gram proposers.
  std::deque<int32_t> CreateGuidanceFFQueue() const;
  static void CommitGuidanceProposalToken(
      ConstrainedLogitsProcessor& grammar,
      int32_t token,
      std::deque<int32_t>& ff_queue);

 private:
  // Each Step emits one token. RunRound does a whole round's compute up front and buffers the
  // committed tokens; DrainOne hands them out one per call; FinalizeRound runs the deferred
  // re-anchor once the buffer empties (so hitting EOS mid-round skips the wasted target pass).
  void DrainOne(Generator& g);
  bool EmitToken(Generator& g, int32_t tok);
  void FinishRound(RoundState::Phase final_phase);
  void DiscardPendingTokens();
  void PrepareForCooldownStep(Generator& g);

  // Runs one guidance round - check the draft's tokens against the grammar-masked target, tell the
  // grammar about each committed token, and add any tokens the grammar forces. Handles greedy and
  // sampling. Runs instead of the normal batched path.
  void RunGuidanceRound(Generator& g, const Proposal& proposal, int seed_length, int K,
                        bool filled_proposal_budget, float propose_ms);

  // Cleans up after a guidance round and returns the target's logits for the next position. Keeps
  // the accepted tokens already in the cache and feeds the rest back one at a time; also refreshes
  // the draft's saved logits.
  DeviceSpan<float> FinalizeGuidanceRound(Generator& g);

  // Rewinds both inner caches to floor and replays the committed tokens from there to the current
  // length; lines both caches back up with the committed sequence. Used when AppendTokens resumes.
  DeviceSpan<float> ReplayCommittedTail(Generator& g, int floor, bool record_stats = true);
  void PrepareForSetLogits(Generator& g, bool record_stats);

  RoundState round_;
  AdaptiveKController adaptive_k_;
  SpeculativeCooldownController cooldown_;
  std::size_t standard_fallback_steps_{};

  // K histogram for rounds that follow the standard geometric acceptance formula.
  std::array<std::size_t, 17> formula_k_counts_{};
  std::size_t formula_rounds_{};

  // Grammar-forced tokens the round could not fit in its K budget, carried to the next round and
  // emitted first (the grammar is already advanced past them).
  std::deque<int32_t> ff_carry_;

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
