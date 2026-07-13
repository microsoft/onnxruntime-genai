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

namespace {

int32_t RowArgmax(std::span<const float> logits) {
  return static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

void SelectProposalToken(std::span<const float> logits, int index, bool greedy,
                         int top_k, float top_p, float temperature, int vocab_size,
                         SampledCategorical& sampled, std::mt19937& rng,
                         SpeculativeDecodingStrategy::Proposal& proposal) {
  if (greedy) {
    proposal.tokens[index] = RowArgmax(logits);
    return;
  }

  ComputeSampledCategorical(logits, top_k, top_p, temperature, sampled);
  proposal.probs[index] = ScatterToFullVocab(sampled, vocab_size);
  std::discrete_distribution<int> distribution(proposal.probs[index].begin(),
                                               proposal.probs[index].end());
  proposal.tokens[index] = static_cast<int32_t>(distribution(rng));
}

}  // namespace

// Validate the downcast before binding the non-null reference member
static SpeculativeDecodingState& RequireSpeculativeState(Generator& g) {
  auto* spec_state = dynamic_cast<SpeculativeDecodingState*>(g.state_.get());
  if (!spec_state) {
    throw std::runtime_error("BaseSpeculativeStrategy requires SpeculativeDecodingState");
  }
  return *spec_state;
}

BaseSpeculativeStrategy::BaseSpeculativeStrategy(Generator& g)
    : SpeculativeDecodingStrategy{RequireSpeculativeState(g).target_state(),
                                  RequireSpeculativeState(g).spec_model().target_model()},
      spec_state_{RequireSpeculativeState(g)} {}

// Propose K draft tokens.
// Greedy: argmax, probs empty.
// Sampling: token i drawn from draft's truncated dist q_i (saved in probs[i] for the skeleton's min(1, p_i/q_i) test). d_0 reuses
// draft_pending_logits_, so only d_1..d_{K-1} run -> ~N*(K-1) passes, not N*K.
// Each draft row is put through the same min-length / repetition penalties the target rows get
// (BaseSpeculativeStrategy shares the helpers in sampling_distribution.h), so the draft approximates the
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
  std::span<const int32_t> eos_ids{params.config.model.eos_token_id};
  LogitsPenaltyProcessor penalty_processor{vocab_size, repetition_penalty, min_length, eos_ids};

  // The committed sequence so far (length seed_length); grows as we chain draft tokens. Only
  // materialized when penalties are active so the default path stays allocation-free.
  std::vector<int32_t> prefix;
  if (penalty_processor.IsActive()) {
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
  Proposal proposal;
  proposal.tokens.resize(K);
  if (!greedy)
    // greedy-match leaves probs empty
    proposal.probs.resize(K);

  SampledCategorical sampled;

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
        auto row = penalty_processor.Apply({pending.data(), pending.size()}, seed_length + i, prefix);
        auto gb = guidance_buf.CpuSpan();
        std::copy(row.begin(), row.end(), gb.begin());
        guidance_buf.CopyCpuToDevice();
        draft_grammar->ProcessLogits(guidance_buf);
        auto masked = guidance_buf.CopyDeviceToCpu();
        SelectProposalToken({masked.data(), static_cast<size_t>(vocab_size)}, i, greedy,
                            search.top_k, search.top_p, search.temperature, vocab_size,
                            sampled, rng_, proposal);
        const int32_t chosen = proposal.tokens[i];
        if (std::find(eos_ids.begin(), eos_ids.end(), chosen) == eos_ids.end()) {
          int32_t c = chosen;
          draft_grammar->CommitTokens({&c, 1});
          for (int32_t f : draft_grammar->GetFFTokens(0)) ff_queue.push_back(f);
        }
      }
      if (penalty_processor.IsActive()) prefix.push_back(proposal.tokens[i]);
      // Advance the draft over this token so the next free position is predicted in context.
      if (i + 1 < K) {
        single_buf.CpuSpan()[0] = proposal.tokens[i];
        single_buf.CopyCpuToDevice();
        auto lgt = spec_state_.draft_state().Run(seed_length + i + 1, single_buf, {});
        draft_runs_++;
        auto cpu = lgt.CopyDeviceToCpu();
        pending.assign(cpu.data(), cpu.data() + vocab_size);
      }
    }
    return proposal;
  }

  // Non-guidance path - d_0 from the carried-over pending logits, then chain d_1..d_{K-1}.
  SelectProposalToken(penalty_processor.Apply(spec_state_.draft_pending_logits(), seed_length, prefix),
                      0, greedy, search.top_k, search.top_p, search.temperature, vocab_size,
                      sampled, rng_, proposal);
  for (int i = 1; i < K; i++) {
    if (penalty_processor.IsActive())
      prefix.push_back(proposal.tokens[i - 1]);
    single_buf.CpuSpan()[0] = proposal.tokens[i - 1];
    single_buf.CopyCpuToDevice();
    auto lgt = spec_state_.draft_state().Run(seed_length + i, single_buf, {});
    draft_runs_++;
    auto cpu = lgt.CopyDeviceToCpu();
    SelectProposalToken(
        penalty_processor.Apply({cpu.data(), static_cast<size_t>(vocab_size)}, seed_length + i, prefix),
        i, greedy, search.top_k, search.top_p, search.temperature, vocab_size,
        sampled, rng_, proposal);
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
    draft_runs_++;
    draft_kv_len = seed_length + K;
  }

  if (rewind_to < draft_kv_len)
    spec_state_.draft_state().RewindTo(rewind_to);

  // Advance one step on final_token; the resulting logits feed next round's d_0.
  single_buf.CpuSpan()[0] = final_token;
  single_buf.CopyCpuToDevice();
  auto draft_lgt = spec_state_.draft_state().Run(seed_length + n_direct + 1, single_buf, {});
  draft_runs_++;
  auto cpu_draft = draft_lgt.CopyDeviceToCpu();
  // Reuse the pending-logits buffer instead of allocating a fresh vocab-sized vector each round.
  spec_state_.assign_draft_pending_logits(cpu_draft.data(), static_cast<size_t>(vocab_size));
}

void BaseSpeculativeStrategy::ReconcileProposer(Generator& g,
                                                int floor,
                                                std::span<const int32_t> committed,
                                                int committed_length,
                                                bool record_stats) {
  const int vocab_size = g.search_->params_->config.model.vocab_size;
  spec_state_.draft_state().RewindTo(static_cast<size_t>(floor));

  const int replay_count = committed_length - floor;
  auto replay = g.search_->params_->p_device->Allocate<int32_t>(static_cast<size_t>(replay_count));
  std::copy_n(committed.data() + floor, replay_count, replay.CpuSpan().data());
  replay.CopyCpuToDevice();
  auto draft_logits = spec_state_.draft_state().Run(committed_length, replay, {});
  if (record_stats)
    draft_runs_++;
  auto draft_cpu = draft_logits.CopyDeviceToCpu();
  spec_state_.assign_draft_pending_logits(draft_cpu.data(), static_cast<size_t>(vocab_size));
}

void BaseSpeculativeStrategy::FinalizeGuidanceProposer(
    Generator& g,
    int seed_length,
    int proposal_length,
    std::span<const int32_t> committed) {
  const int vocab_size = g.search_->params_->config.model.vocab_size;
  const int draft_kv_len = seed_length + proposal_length - 1;
  if (seed_length < draft_kv_len)
    spec_state_.draft_state().RewindTo(static_cast<size_t>(seed_length));

  auto single = g.search_->params_->p_device->Allocate<int32_t>(1);
  DeviceSpan<float> draft_logits;
  for (size_t p = 0; p < committed.size(); p++) {
    single.CpuSpan()[0] = committed[p];
    single.CopyCpuToDevice();
    draft_logits =
        spec_state_.draft_state().Run(seed_length + static_cast<int>(p) + 1, single, {});
    draft_runs_++;
  }
  auto draft_cpu = draft_logits.CopyDeviceToCpu();
  spec_state_.assign_draft_pending_logits(draft_cpu.data(), static_cast<size_t>(vocab_size));
}

}  // namespace Generators
