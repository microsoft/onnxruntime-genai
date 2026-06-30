// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "speculative_decoding_strategy.h"

#include "generators.h"
#include "search.h"
#include "speculative_sampling.h"
#include "models/speculative_decoding.h"

#include <algorithm>
#include <chrono>
#include <random>

namespace Generators {

// Turn a logits row into the probability distribution the accept step compares against.
static std::vector<float> LogitsToDistribution(std::span<const float> logits, bool greedy,
                                               const Config::Search& search, int vocab_size,
                                               SampledCategorical& sampled) {
  if (greedy) {
    // Greedy accept/correct/bonus only needs argmax.
    std::vector<float> onehot(static_cast<size_t>(vocab_size), 0.0f);
    const int32_t idx = static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    onehot[static_cast<size_t>(idx)] = 1.0f;
    return onehot;
  }
  ComputeSampledCategorical(logits, search.top_k, search.top_p, search.temperature, sampled);
  return ScatterToFullVocab(sampled, vocab_size);
}

// Each call emits exactly one token. The first call of a round runs the whole round (RunRound)
// and buffers its tokens; every call then hands out one buffered token (DrainOne).
void SpeculativeDecodingStrategy::Step(Generator& g) {
  if (pending_.empty()) {
    // A new round needs a starting point: either fresh logits or a token left over from the 
    // fold to feed into this round's verify.
    if (!g.computed_logits_ && !pending_anchor_token_.has_value())
      throw std::runtime_error(
          "Speculative decoding: GenerateNextToken called without fresh logits. Call AppendTokens "
          "before GenerateNextToken, and again after any RewindToLength. Continuous decoding via "
          "RewindToLength followed directly by GenerateNextToken is not supported in this release.");
    RunRound(g);
  }
  DrainOne(g);
}

void SpeculativeDecodingStrategy::Reset() {
  pending_.clear();
  reanchor_pending_ = false;
  pending_anchor_token_.reset();
}

// Runs one speculative round:
//   1. Propose: the draft produces K draft tokens.
//   2. Verify: the target scores all K tokens in a single pass.
//   3. Accept each token in order - greedy keeps it if it's the target's top choice, sampling
//      keeps it with probability min(1, p_target/p_draft). The first rejected token is swapped
//      for a "correction" token; if all K are accepted we append one extra bonus token.
//   4. Buffer the accepted (+ correction/bonus) tokens for DrainOne to emit one at a time.
// Re-anchoring the target and the subclass's Advance are deferred to FinalizeRound.
// Note: grab the target's distribution for the first token before the verify Run - afterwards
// GetLogits() holds the last token's row, not the first.
void SpeculativeDecodingStrategy::RunRound(Generator& g) {
  using clock = std::chrono::steady_clock;
  using ms_f = std::chrono::duration<float, std::milli>;

  auto* spec_state = dynamic_cast<SpeculativeDecodingState*>(g.state_.get());
  if (!spec_state)
    throw std::runtime_error(
        "SpeculativeDecodingStrategy::Step requires a SpeculativeDecodingState "
        "(model.type=\"speculative\").");

  const auto& params = *g.search_->params_;
  const int seed_length = g.search_->GetSequenceLength();
  const int vocab_size = params.config.model.vocab_size;
  const int max_length = params.search.max_length;

  int K = params.speculative.max_draft_tokens;
  if (K < 1 || K > 16)
    throw std::runtime_error(
        "Speculative decoding: max_draft_tokens (K) must be in [1, 16]. Got K=" +
        std::to_string(K) + ".");

  const int remaining = max_length - seed_length;
  if (remaining <= 0)
    throw std::runtime_error(
        "Speculative decoding: cannot generate because sequence_length (" +
        std::to_string(seed_length) + ") has reached max_length (" +
        std::to_string(max_length) + ").");

  // Don't look further ahead than max_length.
  K = std::min(K, remaining);
  // Seed the shared RNG.
  if (!rng_seeded_) {
    const uint32_t seed = (params.search.random_seed < 0)
                              ? std::random_device{}()
                              : static_cast<uint32_t>(params.search.random_seed);
    rng_.seed(seed);
    rng_seeded_ = true;
  }

  // Read sampling settings from the canonical config/method rather than a parallel struct.
  const auto& search = params.search;
  const bool greedy = g.IsGreedySampling();

  // Reused scratch buffer for the sampling path of LogitsToDistribution.
  SampledCategorical sampled;

  // Propose: draft produces K candidate tokens.
  auto t_propose_start = clock::now();
  Proposal proposal = Propose(g, K, seed_length);
  auto t_propose_end = clock::now();

  if (static_cast<int>(proposal.tokens.size()) != K)
    throw std::runtime_error(
        "Speculative draft returned " + std::to_string(proposal.tokens.size()) +
        " tokens, expected K=" + std::to_string(K) + ".");
  const bool greedy_accept = proposal.probs.empty();
  if (!greedy_accept && static_cast<int>(proposal.probs.size()) != K)
    throw std::runtime_error(
        "Speculative draft returned " + std::to_string(proposal.probs.size()) +
        " prob rows, expected K=" + std::to_string(K) + " (or 0 for greedy-match).");

  // If a token is waiting from the fold, it goes at the front of this verify batch (so the
  // batch is K+1 wide instead of K). It also fills the gap left in the target's cache.
  const bool use_anchor = pending_anchor_token_.has_value();
  int verify_width = K;
  int32_t anchor_token = 0;
  if (use_anchor) {
    verify_width = K + 1;
    anchor_token = *pending_anchor_token_;
  }
  pending_anchor_token_.reset();  // taken; FinalizeRound may set a new one for next round.

  // We need the target's prediction for proposal token 0 ("pos0"). Without an anchor we already
  // have it from the previous step's logits. With an anchor it comes out of the verify below
  // (it's the row produced right after the anchor token) so -> fill it in there.
  std::vector<float> target_probs_pos0;
  if (!use_anchor) {
    auto pending_cpu_pos0 = g.search_->GetLogits().CopyDeviceToCpu();
    target_probs_pos0 =
        LogitsToDistribution({pending_cpu_pos0.data(), static_cast<size_t>(vocab_size)},
                             greedy, search, vocab_size, sampled);
  }

  // Verify: score the anchor (when present) plus the K proposed tokens in one target pass.
  auto target_input = params.p_device->Allocate<int32_t>(verify_width);
  {
    auto sp = target_input.CpuSpan();
    int w = 0;
    if (use_anchor) sp[w++] = anchor_token;
    for (int i = 0; i < K; i++) sp[w++] = proposal.tokens[i];
  }
  target_input.CopyCpuToDevice();

  auto t_target_start = clock::now();
  spec_state->target_state().Run(seed_length + K, target_input, {});
  auto t_target_end = clock::now();

  // K target distributions.
  const std::string& logits_name =
      spec_state->spec_model().target_model().config_->model.decoder.outputs.logits;
  OrtValue* raw_ort = spec_state->target_state().GetOutput(logits_name.c_str());
  if (!raw_ort)
    throw std::runtime_error(
        "Speculative decoding: target state has no logits output named '" + logits_name + "'.");

  auto raw_shape = raw_ort->GetTensorTypeAndShapeInfo()->GetShape();
  const bool is_multi =
      (raw_shape.size() >= 2 && raw_shape[1] == static_cast<int64_t>(verify_width));
  is_multi_ = is_multi;

  // Runtime vocab-size safety net (once per generator lifetime).
  if (!vocab_check_done_) {
    if (!raw_shape.empty()) {
      int64_t tv = raw_shape.back();
      if (tv > 0 && tv != static_cast<int64_t>(vocab_size))
        throw std::runtime_error(
            "Speculative decoding runtime vocab mismatch: target vocab=" +
            std::to_string(tv) + ", config vocab_size=" + std::to_string(vocab_size));
    }
    vocab_check_done_ = true;
  }

  std::vector<std::vector<float>> target_dists(K);
  auto elem_type = raw_ort->GetTensorTypeAndShapeInfo()->GetElementType();

  if (is_multi) {
    // CPU-only path: the logits live in host memory as fp32, so read them directly. GPU
    // execution providers may hand back a device pointer and fp16 (v1 pending).
    if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      throw std::runtime_error(
          "Speculative decoding: unsupported logits element type (expected fp32).");
    const float* data = raw_ort->GetTensorData<float>();
    auto row = [&](int r) {
      return std::span<const float>{data + static_cast<ptrdiff_t>(r) * vocab_size,
                                    static_cast<size_t>(vocab_size)};
    };
    // The anchor (when present) takes row 0, so the proposal rows start at 1; otherwise they
    // start at 0 and pos0 came from earlier. target_dists[i] ends up as "the
    // target's prediction after proposal token i", so the accept loop below is unchanged.
    int prop_row0 = 0;
    if (use_anchor) {
      prop_row0 = 1;
      target_probs_pos0 = LogitsToDistribution(row(0), greedy, search, vocab_size, sampled);
    }
    for (int i = 0; i < K; i++)
      target_dists[i] = LogitsToDistribution(row(prop_row0 + i), greedy, search, vocab_size, sampled);
  } else {
    // Pruned model - one target pass per token. Guard against out of sync cache.
    if (use_anchor)
      throw std::runtime_error(
          "Speculative decoding: internal error - re-anchor fold reached the non-multi target "
          "path. The fold must only engage for targets that return one logits row per token.");
    spec_state->target_state().RewindTo(seed_length);
    auto single_buf = params.p_device->Allocate<int32_t>(1);
    for (int i = 0; i < K; i++) {
      single_buf.CpuSpan()[0] = proposal.tokens[i];
      single_buf.CopyCpuToDevice();
      auto lgt = spec_state->target_state().Run(seed_length + i + 1, single_buf, {});
      auto cpu = lgt.CopyDeviceToCpu();
      target_dists[i] = LogitsToDistribution({cpu.data(), static_cast<size_t>(vocab_size)},
                                             greedy, search, vocab_size, sampled);
    }
  }

  // Decide accept/reject for each token in order. The target distribution that judges tokens[0]
  // is the one we saved above; for later tokens[i] it's target_dists[i-1]. If every token is
  // accepted, the bonus token is drawn from target_dists[K-1] (the target's next prediction).
  std::uniform_real_distribution<float> uni(0.f, 1.f);

  int n_direct = 0;
  int32_t final_token = -1;
  std::vector<float> correction_buf(vocab_size);

  for (int i = 0; i < K; i++) {
    const auto& t_dist = (i == 0) ? target_probs_pos0 : target_dists[i - 1];
    bool accepted = false;
    if (greedy_accept) {
      int32_t target_argmax = static_cast<int32_t>(
          std::max_element(t_dist.begin(), t_dist.end()) - t_dist.begin());
      accepted = (target_argmax == proposal.tokens[i]);
    } else {
      const float p_t = t_dist[proposal.tokens[i]];
      const float p_d = proposal.probs[i][proposal.tokens[i]];
      accepted = (uni(rng_) < ComputeAcceptProb(p_t, p_d));
    }

    if (accepted) {
      n_direct++;
    } else {
      if (greedy_accept) {
        final_token = static_cast<int32_t>(
            std::max_element(t_dist.begin(), t_dist.end()) - t_dist.begin());
      } else {
        BuildCorrectionDistribution(
            {t_dist.data(), static_cast<size_t>(vocab_size)},
            {proposal.probs[i].data(), static_cast<size_t>(vocab_size)},
            {correction_buf.data(), static_cast<size_t>(vocab_size)});
        final_token = SampleFromDistribution(
            {correction_buf.data(), static_cast<size_t>(vocab_size)}, rng_);
      }
      corrections_++;
      break;
    }
  }

  if (n_direct == K) {
    if (greedy_accept) {
      final_token = static_cast<int32_t>(
          std::max_element(target_dists[K - 1].begin(), target_dists[K - 1].end()) -
          target_dists[K - 1].begin());
    } else {
      final_token = SampleFromDistribution(
          {target_dists[K - 1].data(), static_cast<size_t>(vocab_size)}, rng_);
    }
    bonuses_++;
  }

  // Queue the committed tokens (n_direct accepted + 1 correction/bonus) for DrainOne to emit one
  // per Step. Save the round's details so FinalizeRound can re-anchor target later.
  for (int i = 0; i < n_direct; i++)
    pending_.push_back(proposal.tokens[i]);
  pending_.push_back(final_token);

  saved_final_token_ = final_token;
  saved_n_direct_ = n_direct;
  saved_seed_length_ = seed_length;
  saved_K_ = K;
  saved_proposal_ = std::move(proposal);
  reanchor_pending_ = true;

  // Stats - once per round, not per drained token.
  rounds_++;
  draft_proposed_ += static_cast<size_t>(K);
  draft_accepted_ += static_cast<size_t>(n_direct);
  total_propose_ms_ += ms_f(t_propose_end - t_propose_start).count();
  total_target_ms_ += ms_f(t_target_end - t_target_start).count();
}

// Emits one buffered token (keeping in mind EOS / max_length). Once the last token of the round is
// emitted, run the deferred re-anchor (FinalizeRound) so the model's state matches exactly what
// was streamed to the user.
void SpeculativeDecodingStrategy::DrainOne(Generator& g) {
  const int max_length = g.search_->params_->search.max_length;

  // If we can't emit (done / at max_length), drop the rest of the round and skip the re-anchor.
  if (g.search_->IsDone() || g.search_->GetSequenceLength() >= max_length) {
    pending_.clear();
    reanchor_pending_ = false;
    g.computed_logits_ = false;
    return;
  }

  const int32_t tok = pending_.front();
  pending_.pop_front();
  EmitToken(g, tok);
  g.computed_logits_ = false;

  // Last token of the round just went out: re-anchor now. (Deferring it to here means an EOS
  // partway through the round skips the re-anchor and its wasted target pass.)
  if (pending_.empty() && reanchor_pending_)
    FinalizeRound(g);
}

// Tidy up after a round and get ready for the next one. Verify ran all the proposed tokens, but
// we only kept n_direct of them, so first drop the rejected ones from the target's cache. Then
// handle the one committed token (final_token) in one of two ways:
//   * fold path (is_multi && K>=2): leave it for the next round's verify batch (see RunRound).
//     This skips a whole extra target run each round.
//   * legacy path (K==1 / pruned target): run it through the target now. This keeps K==1
//     byte-for-byte identical to plain greedy decoding.
// Either way we then advance the draft model. Runs once the round's tokens have all been emitted.
void SpeculativeDecodingStrategy::FinalizeRound(Generator& g) {
  using clock = std::chrono::steady_clock;
  using ms_f = std::chrono::duration<float, std::milli>;

  reanchor_pending_ = false;

  if (g.search_->IsDone()) {
    g.computed_logits_ = false;
    return;
  }

  auto* spec_state = dynamic_cast<SpeculativeDecodingState*>(g.state_.get());
  const auto& params = *g.search_->params_;

  const int target_kv_len = saved_seed_length_ + saved_K_;
  const int rewind_to = saved_seed_length_ + saved_n_direct_;

  // Only rewind if there's actually something to drop.
  if (rewind_to < target_kv_len)
    spec_state->target_state().RewindTo(rewind_to);

  const bool fold = is_multi_ && saved_K_ >= 2;
  if (fold) {
    // Hand the committed token to the next round instead of running it now. Its verify pass
    // will both place it in the cache and give us its prediction - saving a full target run.
    pending_anchor_token_ = saved_final_token_;
  } else {
    auto single_buf = params.p_device->Allocate<int32_t>(1);
    single_buf.CpuSpan()[0] = saved_final_token_;
    single_buf.CopyCpuToDevice();

    const int next_len = saved_seed_length_ + saved_n_direct_ + 1;
    auto t_target_start = clock::now();
    auto target_lgt = spec_state->target_state().Run(next_len, single_buf, {});
    auto t_target_end = clock::now();
    // One single-token target run, which is exactly the baseline cost per token (T_target).
    // We track it separately from the K-token verify for the speedup formula in GetStats.
    // (The fold path has no such run, so it leaves this at 0.)
    total_reanchor_ms_ += ms_f(t_target_end - t_target_start).count();
    reanchor_runs_++;
    g.SetLogits(target_lgt);
  }

  // Update draft model (KV cache + probs); count it as propose time. Unchanged by the fold - the
  // draft always advances on final_token to seed the next round's first proposal.
  auto t_advance_start = clock::now();
  Advance(g, saved_proposal_, saved_n_direct_, saved_final_token_, saved_seed_length_);
  auto t_advance_end = clock::now();
  total_propose_ms_ += ms_f(t_advance_end - t_advance_start).count();
}

// Commit one already-decided token.
void SpeculativeDecodingStrategy::EmitToken(Generator& g, int32_t tok) {
  const auto& params = *g.search_->params_;
  const int vocab_size = params.config.model.vocab_size;

  if (onehot_buf_.empty())
    onehot_buf_ = params.p_device->Allocate<float>(vocab_size);

  auto logit_span = onehot_buf_.CpuSpan();
  std::fill(logit_span.begin(), logit_span.end(), -1e9f);
  logit_span[tok] = 1e9f;
  onehot_buf_.CopyCpuToDevice();
  g.search_->SetLogits(onehot_buf_);
  g.search_->SelectTop();
}

SpeculativeStats SpeculativeDecodingStrategy::GetStats() const {
  SpeculativeStats s{};
  s.rounds = rounds_;
  s.draft_tokens_proposed = draft_proposed_;
  s.draft_tokens_accepted = draft_accepted_;
  s.correction_tokens = corrections_;
  s.bonus_tokens = bonuses_;
  if (draft_proposed_ > 0) {
    s.avg_draft_ms_per_token = total_propose_ms_ / static_cast<float>(draft_proposed_);
    s.avg_target_ms_per_token = total_target_ms_ / static_cast<float>(draft_proposed_);
    s.acceptance_rate =
        static_cast<float>(draft_accepted_) / static_cast<float>(draft_proposed_);
  }
  const std::size_t committed = draft_accepted_ + corrections_ + bonuses_;
  if (rounds_ > 0) {
    s.mean_accepted_tokens =
        static_cast<float>(committed) / static_cast<float>(rounds_);
  }
  if (reanchor_runs_ > 0 && committed > 0) {
    // Speedup = E[tok/round] / (1 + k*(T_draft/T_target) + x), mapped to measured
    // per-round times: reanchor=1*T_target, verify=x*T_target, propose=k*T_draft.
    // Only the non-fold path runs a single-token target forward, so it is the only
    // path that measures T_target; under the re-anchor fold this stays 0
    const float t_target = total_reanchor_ms_ / static_cast<float>(reanchor_runs_);
    const float t_spec_total = total_propose_ms_ + total_target_ms_ + total_reanchor_ms_;
    if (t_spec_total > 0.0f)
      s.effective_speedup =
          static_cast<float>(committed) * t_target / t_spec_total;
  }
  return s;
}

}  // namespace Generators
