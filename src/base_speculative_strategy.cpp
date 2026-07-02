// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "base_speculative_strategy.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "generators.h"
#include "search.h"
#include "softmax.h"
#include "speculative_sampling.h"
#include "models/speculative_decoding.h"

namespace Generators {

// Validate the downcast before binding the non-null reference member
static SpeculativeDecodingState& RequireSpeculativeState(Generator& g) {
  auto* spec_state = dynamic_cast<SpeculativeDecodingState*>(g.state_.get());
  if (!spec_state) {
    throw std::runtime_error("BaseSpeculativeStrategy requires SpeculativeDecodingState");
  }
  return *spec_state;
}

BaseSpeculativeStrategy::BaseSpeculativeStrategy(Generator& g) : spec_state_{RequireSpeculativeState(g)} {}

// Propose K draft tokens.
// Greedy: argmax, probs empty.
// Sampling: token i drawn from draft's truncated dist q_i (saved in probs[i] for the skeleton's min(1, p_i/q_i) test). d_0 reuses
// draft_pending_logits_, so only d_1..d_{K-1} run -> ~N*(K-1) passes, not N*K.
// Each draft row is put through the same min-length / repetition penalties the target rows get
// (BaseSpeculativeStrategy shares the helpers in logits_penalty.h), so the draft approximates the
// penalized target: acceptance stays high and self-speculative greedy stays exact.
SpeculativeDecodingStrategy::Proposal BaseSpeculativeStrategy::Propose(
    Generator& g, int K, int seed_length) {
  if (!spec_state_.draft_pending_valid())
    throw std::runtime_error(
        "BaseSpeculativeStrategy::Propose: draft pending logits not initialized. "
        "AppendTokens must be called before GenerateNextToken.");

  const auto& params = *g.search_->params_;
  const int vocab_size = params.config.model.vocab_size;
  // Read sampling settings from the canonical config/method rather than a parallel struct.
  const auto& search = params.search;
  const bool greedy = g.IsGreedySampling();

  // Penalty context: apply exactly what the target rows will receive (see RunRound), so the
  // draft's proposal distribution matches the penalized target it is verified against.
  const float repetition_penalty = search.repetition_penalty;
  const int min_length = search.min_length;
  const bool penalties_active = (repetition_penalty != 1.0f) || (min_length > 0);
  std::span<const int32_t> eos_ids{params.config.model.eos_token_id};

  // The committed sequence so far (length seed_length); grows as we chain draft tokens. Only
  // materialized when penalties are active so the default path stays allocation-free.
  std::vector<int32_t> prefix;
  std::vector<bool> rep_visited;
  std::vector<float> row_buf;
  if (penalties_active) {
    auto seed = g.search_->GetSequence(0).CopyDeviceToCpu();
    prefix.assign(seed.begin(), seed.end());
    row_buf.resize(static_cast<size_t>(vocab_size));
  }

  Proposal proposal;
  proposal.tokens.resize(K);
  if (!greedy)
    // greedy-match leaves probs empty
    proposal.probs.resize(K);

  auto argmax = [](std::span<const float> v) {
    return static_cast<int32_t>(std::max_element(v.begin(), v.end()) - v.begin());
  };

  SampledCategorical sampled;

  // Turn one draft logits row (predicting sequence index current_length) into a token, applying
  // the shared penalties first. Fills proposal.probs[idx] with the truncated dist in the sampling path.
  auto emit_from_logits = [&](std::span<const float> logits, int idx, int current_length) {
    std::span<const float> row = logits;
    if (penalties_active) {
      std::copy(logits.begin(), logits.end(), row_buf.begin());
      ApplyMinLengthToLogits({row_buf.data(), row_buf.size()}, current_length, min_length, eos_ids);
      ApplyRepetitionPenaltyToLogits({row_buf.data(), row_buf.size()}, prefix, repetition_penalty, rep_visited);
      row = {row_buf.data(), row_buf.size()};
    }
    if (greedy) {
      proposal.tokens[idx] = argmax(row);
    } else {
      ComputeSampledCategorical(row, search.top_k, search.top_p, search.temperature, sampled);
      proposal.probs[idx] = ScatterToFullVocab(sampled, vocab_size);
      std::discrete_distribution<int> dist(proposal.probs[idx].begin(), proposal.probs[idx].end());
      proposal.tokens[idx] = static_cast<int32_t>(dist(rng_));
    }
  };

  // d_0 from the carried-over pending draft logits; context = committed sequence (len seed_length).
  emit_from_logits(spec_state_.draft_pending_logits(), 0, seed_length);

  // d_1..d_{K-1}: feed the previous draft token through the draft model; context grows by one.
  auto single_buf = params.p_device->Allocate<int32_t>(1);
  for (int i = 1; i < K; i++) {
    if (penalties_active)
      prefix.push_back(proposal.tokens[i - 1]);
    single_buf.CpuSpan()[0] = proposal.tokens[i - 1];
    single_buf.CopyCpuToDevice();
    auto lgt = spec_state_.draft_state().Run(seed_length + i, single_buf, {});
    auto cpu = lgt.CopyDeviceToCpu();
    emit_from_logits({cpu.data(), static_cast<size_t>(vocab_size)}, i, seed_length + i);
  }

  return proposal;
}

// Re-sync draft to the committed length, then advance on final_token (its logits = next round's
// d_0). n_direct == K: all accepted, advance once more (bonus), no rewind; else rewind draft to
// seed_length + n_direct.
void BaseSpeculativeStrategy::Advance(Generator& g,
                                      const Proposal& proposal,
                                      int n_direct,
                                      int32_t final_token,
                                      int seed_length) {
  const auto& params = *g.search_->params_;
  const int vocab_size = params.config.model.vocab_size;
  // Derive K from the proposal, not config: Step may have clamped K against max_length,
  // and the draft cache was advanced by that clamped K. Avoid wrong sync.
  const int K = static_cast<int>(proposal.tokens.size());

  auto single_buf = params.p_device->Allocate<int32_t>(1);

  int draft_kv_len = seed_length + K - 1;
  const int rewind_to = seed_length + n_direct;

  if (n_direct == K) {
    // All proposed tokens accepted: catch draft up to where the K-th token
    // would have advanced it (one extra step on last proposed token).
    single_buf.CpuSpan()[0] = proposal.tokens[K - 1];
    single_buf.CopyCpuToDevice();
    spec_state_.draft_state().Run(seed_length + K, single_buf, {});
    draft_kv_len = seed_length + K;
  }

  if (rewind_to < draft_kv_len)
    spec_state_.draft_state().RewindTo(rewind_to);

  // Advance one step on final_token; the resulting logits feed next round's d_0.
  single_buf.CpuSpan()[0] = final_token;
  single_buf.CopyCpuToDevice();
  auto draft_lgt = spec_state_.draft_state().Run(seed_length + n_direct + 1, single_buf, {});
  auto cpu_draft = draft_lgt.CopyDeviceToCpu();
  std::vector<float> draft_logits(cpu_draft.data(), cpu_draft.data() + vocab_size);
  spec_state_.set_draft_pending_logits(std::move(draft_logits));
}

}  // namespace Generators
