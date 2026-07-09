// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "base_speculative_strategy.h"

#include <algorithm>
#include <deque>
#include <stdexcept>
#include <vector>

#include "generators.h"
#include "search.h"
#include "softmax.h"
#include "speculative_sampling.h"
#include "constrained_logits_processor.h"
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
  }

  // Guidance - clone the grammar cursor so we can look ahead over K positions
  // without disturbing the verify cursor.
  const bool guidance_active = (g.guidance_logits_processor_ != nullptr);
  std::unique_ptr<ConstrainedLogitsProcessor> draft_grammar;
  DeviceSpan<float> guidance_buf;
  if (guidance_active) {
    draft_grammar = g.guidance_logits_processor_->Clone();
    guidance_buf = params.p_device->Allocate<float>(static_cast<size_t>(vocab_size));
  }
  if (penalties_active || guidance_active)
    row_buf.resize(static_cast<size_t>(vocab_size));

  auto is_eos = [&](int32_t t) {
    return std::find(eos_ids.begin(), eos_ids.end(), t) != eos_ids.end();
  };

  Proposal proposal;
  proposal.tokens.resize(K);
  if (!greedy)
    // greedy-match leaves probs empty
    proposal.probs.resize(K);

  auto argmax = [](std::span<const float> v) {
    return static_cast<int32_t>(std::max_element(v.begin(), v.end()) - v.begin());
  };

  SampledCategorical sampled;

  // Apply the shared penalties to a draft row in place.
  auto penalize = [&](std::span<const float> logits, int current_length) -> std::span<const float> {
    if (!penalties_active) return logits;
    std::copy(logits.begin(), logits.end(), row_buf.begin());
    ApplyMinLengthToLogits({row_buf.data(), row_buf.size()}, current_length, min_length, eos_ids);
    ApplyRepetitionPenaltyToLogits({row_buf.data(), row_buf.size()}, prefix, repetition_penalty, rep_visited);
    return {row_buf.data(), row_buf.size()};
  };

  // Turn one (penalized) draft row into a token, filling probs[idx] in the sampling path.
  auto pick = [&](std::span<const float> row, int idx) {
    if (greedy) {
      proposal.tokens[idx] = argmax(row);
    } else {
      ComputeSampledCategorical(row, search.top_k, search.top_p, search.temperature, sampled);
      proposal.probs[idx] = ScatterToFullVocab(sampled, vocab_size);
      std::discrete_distribution<int> dist(proposal.probs[idx].begin(), proposal.probs[idx].end());
      proposal.tokens[idx] = static_cast<int32_t>(dist(rng_));
    }
  };

  auto single_buf = params.p_device->Allocate<int32_t>(1);

  if (guidance_active) {
    // Walk the clone forward over K positions. At each free position the draft
    // predicts a grammar-masked token; committing it may force a span of fast-forward tokens, which
    // we inject straight into the proposal (counted in K) so the target verifies them in the same
    // batched pass. Anything past K carries to the next round. 
    std::deque<int32_t> ff_queue(GuidanceFFCarry().begin(), GuidanceFFCarry().end());
    // logits for position 0
    std::vector<float> pending(spec_state_.draft_pending_logits().begin(),
                               spec_state_.draft_pending_logits().end());
    for (int i = 0; i < K; i++) {
      if (!ff_queue.empty()) {
        // forced token: no draft prediction, no distribution
        proposal.tokens[i] = ff_queue.front();
        ff_queue.pop_front();
        if (!greedy) proposal.probs[i].clear();
      } else {
        auto row = penalize({pending.data(), pending.size()}, seed_length + i);
        auto gb = guidance_buf.CpuSpan();
        std::copy(row.begin(), row.end(), gb.begin());
        guidance_buf.CopyCpuToDevice();
        draft_grammar->ProcessLogits(guidance_buf);
        auto masked = guidance_buf.CopyDeviceToCpu();
        pick({masked.data(), static_cast<size_t>(vocab_size)}, i);
        const int32_t chosen = proposal.tokens[i];
        if (!is_eos(chosen)) {
          int32_t c = chosen;
          draft_grammar->CommitTokens({&c, 1});
          for (int32_t f : draft_grammar->GetFFTokens(0)) ff_queue.push_back(f);
        }
      }
      if (penalties_active) prefix.push_back(proposal.tokens[i]);
      // Advance the draft over this token so the next free position is predicted in context.
      if (i + 1 < K) {
        single_buf.CpuSpan()[0] = proposal.tokens[i];
        single_buf.CopyCpuToDevice();
        auto lgt = spec_state_.draft_state().Run(seed_length + i + 1, single_buf, {});
        auto cpu = lgt.CopyDeviceToCpu();
        pending.assign(cpu.data(), cpu.data() + vocab_size);
      }
    }
    return proposal;
  }

  // Non-guidance path - d_0 from the carried-over pending logits, then chain d_1..d_{K-1}.
  pick(penalize(spec_state_.draft_pending_logits(), seed_length), 0);
  for (int i = 1; i < K; i++) {
    if (penalties_active)
      prefix.push_back(proposal.tokens[i - 1]);
    single_buf.CpuSpan()[0] = proposal.tokens[i - 1];
    single_buf.CopyCpuToDevice();
    auto lgt = spec_state_.draft_state().Run(seed_length + i, single_buf, {});
    auto cpu = lgt.CopyDeviceToCpu();
    pick(penalize({cpu.data(), static_cast<size_t>(vocab_size)}, seed_length + i), i);
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
  // Reuse the pending-logits buffer instead of allocating a fresh vocab-sized vector each round.
  spec_state_.assign_draft_pending_logits(cpu_draft.data(), static_cast<size_t>(vocab_size));
}

}  // namespace Generators
