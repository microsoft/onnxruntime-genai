// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "base_speculative_strategy.h"

#include <algorithm>
#include <stdexcept>

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
// draft_pending_probs_, so only d_1..d_{K-1} run -> ~N*(K-1) passes, not N*K.
SpeculativeDecodingStrategy::Proposal BaseSpeculativeStrategy::Propose(
    Generator& g, int K, int seed_length) {
  if (!spec_state_.draft_pending_valid())
    throw std::runtime_error(
        "BaseSpeculativeStrategy::Propose: draft pending probs not initialized. "
        "AppendTokens must be called before GenerateNextToken.");

  const auto& params = *g.search_->params_;
  const int vocab_size = params.config.model.vocab_size;
  // Read sampling settings from the canonical config/method rather than a parallel struct.
  const auto& search = params.search;
  const bool greedy = g.IsGreedySampling();

  Proposal proposal;
  proposal.tokens.resize(K);
  if (!greedy)
    // greedy-match leaves probs empty
    proposal.probs.resize(K);  

  auto argmax = [](std::span<const float> v) {
    return static_cast<int32_t>(std::max_element(v.begin(), v.end()) - v.begin());
  };

  // d_0 from the carried-over pending probs.
  if (greedy) {
    proposal.tokens[0] = argmax(spec_state_.draft_pending_probs());
  } else {
    proposal.probs[0] = SamplingDistributionFromProbs(
        spec_state_.draft_pending_probs(), search.top_k, search.top_p,
        search.temperature);
    std::discrete_distribution<int> dist(proposal.probs[0].begin(), proposal.probs[0].end());
    proposal.tokens[0] = static_cast<int32_t>(dist(rng_));
  }

  // d_1..d_{K-1}: feed the previous draft token through the draft model.
  auto single_buf = params.p_device->Allocate<int32_t>(1);
  SampledCategorical sampled;
  for (int i = 1; i < K; i++) {
    single_buf.CpuSpan()[0] = proposal.tokens[i - 1];
    single_buf.CopyCpuToDevice();
    auto lgt = spec_state_.draft_state().Run(seed_length + i, single_buf, {});
    auto cpu = lgt.CopyDeviceToCpu();
    std::span<const float> logits{cpu.data(), static_cast<size_t>(vocab_size)};
    if (greedy) {
      proposal.tokens[i] = argmax(logits);
    } else {
      ComputeSampledCategorical(logits, search.top_k, search.top_p,
                                search.temperature, sampled);
      proposal.probs[i] = ScatterToFullVocab(sampled, vocab_size);
      std::discrete_distribution<int> dist(proposal.probs[i].begin(), proposal.probs[i].end());
      proposal.tokens[i] = static_cast<int32_t>(dist(rng_));
    }
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
  std::vector<float> draft_probs(cpu_draft.data(), cpu_draft.data() + vocab_size);
  Softmax(draft_probs, 1.0f);
  spec_state_.set_draft_pending_probs(std::move(draft_probs));
}

}  // namespace Generators
