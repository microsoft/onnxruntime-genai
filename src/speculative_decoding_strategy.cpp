// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "speculative_decoding_strategy.h"

#include "generators.h"
#include "search.h"
#include "constrained_logits_processor.h"
#include "speculative_sampling.h"
#include "models/speculative_decoding.h"

#include <algorithm>
#include <chrono>
#include <random>

namespace Generators {

namespace {
// A target verify row kept cheaply (item #2): greedy needs only the argmax id; sampling keeps the
// truncated categorical (kept ids + renormalized probs) -- far smaller than a dense vocab-sized
// vector -- and densifies it on demand only for the single correction/bonus row.
struct SparseRow {
  std::vector<int32_t> indices;
  std::vector<float> probs;
};

int32_t RowArgmax(std::span<const float> logits) {
  return static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}
}  // namespace

// Each call emits exactly one token. The first call of a round runs the whole round (RunRound)
// and buffers its tokens; every call then hands out one buffered token (DrainOne).
void SpeculativeDecodingStrategy::Step(Generator& g) {
  if (pending_.empty()) {
    // Fresh logits already present after prefill/ComputeLogits or when fold left a token
    // to start verify (pending_anchor_token_). After RewindToLength -> stale, so replay
    // the boundary token like StandardDecodingStrategy - ComputeLogits -> Run refreshes both the
    // target logits and draft_pending_logits_ before RunRound.
    if (!g.computed_logits_ && !pending_anchor_token_.has_value()) {
      if (g.search_->GetSequenceLength() == 0)
        throw std::runtime_error(
            "Speculative decoding: GenerateNextToken called with no prior state. Please call "
            "AppendTokens, SetLogits, or SetInputs before calling GenerateNextToken.");
      auto next_tokens = g.search_->GetNextTokens();
      if (g.last_action_ == Generator::Action::rewound)
        g.search_->AppendTokens(next_tokens);
      g.ComputeLogits(next_tokens);
    }
    RunRound(g);
  }
  DrainOne(g);
}

void SpeculativeDecodingStrategy::Reset() {
  pending_.clear();
  ClearPendingExternalLogits();
  reanchor_pending_ = false;
  pending_anchor_token_.reset();
  round_dirty_ = false;
  guidance_round_ = false;
  ff_carry_.clear();
}

// Continuous decoding via a mid-generation AppendTokens. A buffered round can leave the inner
// caches ahead of the committed sequence (mid-round) or behind it (deferred fold). Both caches were 
// consistent at the round start (saved_seed_length_), so rewind both just below it and replay 
// the committed tail to land them on the committed length; drop interrupted round's buffered tokens.
void SpeculativeDecodingStrategy::PrepareForAppend(Generator& g) {
  if (!round_dirty_)
    return;

  auto* spec_state = dynamic_cast<SpeculativeDecodingState*>(g.state_.get());
  if (!spec_state)
    throw std::runtime_error(
        "SpeculativeDecodingStrategy::PrepareForAppend requires a SpeculativeDecodingState.");

  int floor = saved_seed_length_ - 1;
  if (floor < 0)
    floor = 0;

  // Drop the interrupted round's tokens.
  pending_.clear();
  ClearPendingExternalLogits();
  reanchor_pending_ = false;
  pending_anchor_token_.reset();
  round_dirty_ = false;
  guidance_round_ = false;
  ff_carry_.clear();

  if (floor >= g.search_->GetSequenceLength())
    return;

  ReplayCommittedTail(g, floor);
}

bool SpeculativeDecodingStrategy::TryGetExternalLogits(Generator& g, DeviceSpan<float>& logits) {
  if (g.search_->IsDone())
    throw std::runtime_error("Speculative decoding logits are unavailable after generation is complete.");

  if (round_dirty_ && !pending_.empty()) {
    if (g.guidance_logits_processor_)
      throw std::runtime_error("Speculative decoding logits cannot be accessed while a guided round has buffered tokens.");

    const int vocab_size = g.search_->params_->config.model.vocab_size;
    if (current_target_logits_row_ < 0 ||
        current_target_logits_row_ < pending_target_logits_row0_ ||
        current_target_logits_row_ >= pending_target_logits_row0_ + cached_direct_tokens_) {
      throw std::runtime_error("Speculative decoding has no target logits for the current committed sequence.");
    }

    const size_t offset = static_cast<size_t>(current_target_logits_row_) *
                          static_cast<size_t>(vocab_size);
    if (offset + static_cast<size_t>(vocab_size) > pending_target_logits_.size())
      throw std::runtime_error("Speculative decoding cached target logits are incomplete.");

    logits = pending_target_logits_.subspan(offset, static_cast<size_t>(vocab_size));
    return true;
  }

  // A deferred fold has no buffered tokens, so materializing its boundary logits cannot change
  // token or RNG state. Reconcile it once and use the regular computed-logits path below.
  if (round_dirty_ || !g.computed_logits_)
    PrepareForSetLogits(g);

  return false;
}

void SpeculativeDecodingStrategy::PrepareForSetLogits(Generator& g) {
  if (g.search_->IsDone())
    throw std::runtime_error("Speculative decoding logits are unavailable after generation is complete.");

  if (round_dirty_) {
    if (g.guidance_logits_processor_)
      throw std::runtime_error("Speculative decoding logits cannot be replaced while a guided round has buffered tokens.");

    int floor = std::max(saved_seed_length_ - 1, 0);
    if (floor >= g.search_->GetSequenceLength())
      throw std::runtime_error("Speculative decoding cannot reconcile logits with the committed sequence.");

    Reset();
    g.SetLogits(ReplayCommittedTail(g, floor));
    g.last_action_ = Generator::Action::standard;
    return;
  }

  if (g.computed_logits_)
    return;
  if (g.search_->GetSequenceLength() == 0)
    throw std::runtime_error("Speculative decoding logits require prior input tokens.");
  if (g.last_action_ != Generator::Action::rewound)
    throw std::runtime_error("Speculative decoding logits are not available at the current generation state.");

  auto next_tokens = g.search_->GetNextTokens();
  g.search_->AppendTokens(next_tokens);
  g.ComputeLogits(next_tokens);
}

void SpeculativeDecodingStrategy::ClearPendingExternalLogits() {
  pending_target_logits_ = {};
  pending_target_logits_row0_ = 0;
  current_target_logits_row_ = -1;
  emitted_direct_tokens_ = 0;
  cached_direct_tokens_ = 0;
}

// Rewinds both inner caches to floor and replays the committed tokens back to the current length,
// re-syncing both caches. Returns the target's logits at the end. Used when AppendTokens resumes.
DeviceSpan<float> SpeculativeDecodingStrategy::ReplayCommittedTail(Generator& g, int floor) {
  auto* spec_state = static_cast<SpeculativeDecodingState*>(g.state_.get());
  const auto& params = *g.search_->params_;
  const int committed_length = g.search_->GetSequenceLength();

  spec_state->RewindTo(static_cast<size_t>(floor));

  auto committed = g.search_->GetSequence(0).CopyDeviceToCpu();
  const int replay_count = committed_length - floor;
  auto replay = params.p_device->Allocate<int32_t>(static_cast<size_t>(replay_count));
  auto replay_cpu = replay.CpuSpan();
  for (int i = 0; i < replay_count; i++)
    replay_cpu[i] = committed[static_cast<size_t>(floor + i)];
  replay.CopyCpuToDevice();
  return spec_state->Run(committed_length, replay, {});
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
  ClearPendingExternalLogits();

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

  // Guidance - run a dedicated grammar-masked round instead of the batched path below (still uses
  // the draft proposal from above). Handles greedy and sampling.
  if (g.guidance_logits_processor_) {
    RunGuidanceRound(g, proposal, seed_length, K, ms_f(t_propose_end - t_propose_start).count());
    return;
  }

  // Penalty context (min-length + repetition penalty). Every target row is put through the same
  // helpers the standard search uses (logits_penalty.h), so the committed tokens come from the
  // penalized target distribution -- matching plain decoding token-for-token. seed_prefix is the
  // committed sequence at the start of the round; each row extends it by the proposed tokens up to
  // the position it predicts. Only materialized when a penalty is active, so the default path is
  // byte-for-byte unchanged and allocation-free.
  const float repetition_penalty = search.repetition_penalty;
  const int min_length = search.min_length;
  const bool penalties_active = (repetition_penalty != 1.0f) || (min_length > 0);
  std::span<const int32_t> eos_ids{params.config.model.eos_token_id};
  std::vector<int32_t> seed_prefix;
  std::vector<bool> rep_visited;
  std::vector<float> row_buf;
  if (penalties_active) {
    auto seed = g.search_->GetSequence(0).CopyDeviceToCpu();
    seed_prefix.assign(seed.begin(), seed.end());
    row_buf.resize(static_cast<size_t>(vocab_size));
  }

  // Apply penalties (when active) to a logits row in place and return a view of the (possibly
  // rewritten) row. No allocation/copy on the default no-penalty path.
  auto penalize = [&](std::span<const float> logits, int current_length,
                      std::span<const int32_t> prefix) -> std::span<const float> {
    if (!penalties_active)
      return logits;
    std::copy(logits.begin(), logits.end(), row_buf.begin());
    ApplyMinLengthToLogits({row_buf.data(), row_buf.size()}, current_length, min_length, eos_ids);
    ApplyRepetitionPenaltyToLogits({row_buf.data(), row_buf.size()}, prefix, repetition_penalty, rep_visited);
    return {row_buf.data(), row_buf.size()};
  };

  // Cheap per-row storage (item #2): greedy keeps just the argmax id; sampling keeps the truncated
  // categorical and densifies at most one row (the single correction/bonus) later. This replaces
  // building K dense vocab-sized rows every round. scratch_sc is a single reused vocab scratch.
  std::vector<int32_t> target_argmax;    // greedy: prediction after each proposal token
  int32_t pos0_argmax = -1;              // greedy: prediction for proposal token 0
  std::vector<SparseRow> target_sparse;  // sampling: same rows, as truncated categoricals
  SparseRow pos0_sparse;
  SampledCategorical scratch_sc;
  if (greedy_accept)
    target_argmax.assign(static_cast<size_t>(K), -1);
  else
    target_sparse.resize(static_cast<size_t>(K));

  // Store proposal-token-0's target prediction (pos0).
  auto store_pos0 = [&](std::span<const float> logits, int current_length,
                        std::span<const int32_t> prefix) {
    auto row = penalize(logits, current_length, prefix);
    if (greedy_accept) {
      pos0_argmax = RowArgmax(row);
    } else {
      ComputeSampledCategorical(row, search.top_k, search.top_p, search.temperature, scratch_sc);
      pos0_sparse = {scratch_sc.indices, scratch_sc.probs};
    }
  };
  // Store target prediction row i (the target's prediction after proposal token i).
  auto store_row = [&](int i, std::span<const float> logits, int current_length,
                       std::span<const int32_t> prefix) {
    auto row = penalize(logits, current_length, prefix);
    if (greedy_accept) {
      target_argmax[static_cast<size_t>(i)] = RowArgmax(row);
    } else {
      ComputeSampledCategorical(row, search.top_k, search.top_p, search.temperature, scratch_sc);
      target_sparse[static_cast<size_t>(i)] = {scratch_sc.indices, scratch_sc.probs};
    }
  };

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
  if (!use_anchor) {
    auto pending_cpu_pos0 = g.search_->GetLogits().CopyDeviceToCpu();
    store_pos0({pending_cpu_pos0.data(), static_cast<size_t>(vocab_size)}, seed_length, seed_prefix);
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
  target_runs_++;
  auto t_target_end = clock::now();

  // K target distributions.
  const std::string& logits_name =
      spec_state->spec_model().target_model().config_->model.decoder.outputs.logits;
  OrtValue* raw_ort = spec_state->target_state().GetOutput(logits_name.c_str());
  if (!raw_ort)
    throw std::runtime_error(
        "Speculative decoding: target state has no logits output named '" + logits_name + "'.");

  auto raw_shape = raw_ort->GetTensorTypeAndShapeInfo()->GetShape();
  const bool is_multiple_tokens =
      (raw_shape.size() >= 2 && raw_shape[1] == static_cast<int64_t>(verify_width));
  is_multiple_tokens_ = is_multiple_tokens;

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

  if (is_multiple_tokens) {
    // Device-agnostic verify read (mirrors Logits::Get()) - cast fp16/bf16 -> fp32 on the target's
    // device (Cast falls back to CPU when needed), wrap as a DeviceSpan<float>, and copy to host.
    // No-op on CPU; on GPU/NPU it handles device and fp16/bf16 logits.
    auto& target_device = *spec_state->spec_model().target_model().p_device_inputs_;
    const auto elem_type = raw_ort->GetTensorTypeAndShapeInfo()->GetElementType();

    DeviceSpan<float> verify_logits;
    if (elem_type == Ort::TypeToTensorType<float>) {
      verify_logits = WrapTensor<float>(target_device, *raw_ort);
    } else {
      Cast(*raw_ort, verify_logits_fp32_, target_device, Ort::TypeToTensorType<float>);
      verify_logits = WrapTensor<float>(target_device, *verify_logits_fp32_);
    }
    const float* data = verify_logits.CopyDeviceToCpu().data();
    auto row = [&](int r) {
      return std::span<const float>{data + static_cast<ptrdiff_t>(r) * vocab_size,
                                    static_cast<size_t>(vocab_size)};
    };
    // The anchor (when present) takes row 0, so the proposal rows start at 1; otherwise they
    // start at 0 and pos0 came from earlier. Row i ends up as "the target's prediction after
    // proposal token i", so the accept loop below is unchanged. Penalty context for row i is
    // seed + proposal.tokens[0..i] (it predicts the token at sequence index seed_length + i + 1).
    int prop_row0 = 0;
    if (use_anchor) {
      prop_row0 = 1;
      store_pos0(row(0), seed_length, seed_prefix);
    }
    pending_target_logits_ = verify_logits;
    pending_target_logits_row0_ = prop_row0;

    std::vector<int32_t> dist_prefix;
    if (penalties_active) dist_prefix = seed_prefix;
    for (int i = 0; i < K; i++) {
      if (penalties_active) dist_prefix.push_back(proposal.tokens[i]);
      store_row(i, row(prop_row0 + i), seed_length + i + 1, dist_prefix);
    }
  } else {
    // Pruned model - one target pass per token. Guard against out of sync cache.
    if (use_anchor)
      throw std::runtime_error(
          "Speculative decoding: internal error - re-anchor fold reached the non-multi target "
          "path. The fold must only engage for targets that return one logits row per token.");
    spec_state->target_state().RewindTo(seed_length);
    auto single_buf = params.p_device->Allocate<int32_t>(1);
    pending_target_logits_ =
        params.p_device->Allocate<float>(static_cast<size_t>(K) * static_cast<size_t>(vocab_size));
    auto cached_rows = pending_target_logits_.CpuSpan();
    pending_target_logits_row0_ = 0;
    std::vector<int32_t> dist_prefix;
    if (penalties_active) dist_prefix = seed_prefix;
    for (int i = 0; i < K; i++) {
      single_buf.CpuSpan()[0] = proposal.tokens[i];
      single_buf.CopyCpuToDevice();
      auto lgt = spec_state->target_state().Run(seed_length + i + 1, single_buf, {});
      target_runs_++;
      auto cpu = lgt.CopyDeviceToCpu();
      std::copy_n(cpu.data(), static_cast<size_t>(vocab_size),
                  cached_rows.data() + static_cast<ptrdiff_t>(i) * vocab_size);
      if (penalties_active) dist_prefix.push_back(proposal.tokens[i]);
      store_row(i, {cpu.data(), static_cast<size_t>(vocab_size)}, seed_length + i + 1, dist_prefix);
    }
    pending_target_logits_.CopyCpuToDevice();
  }

  // Decide accept/reject for each token in order. The target prediction that judges tokens[0] is
  // pos0 (pos0_argmax / pos0_sparse); for later tokens[i] it's row i-1. If every token is accepted,
  // the bonus token is drawn from row K-1 (the target's next prediction).
  std::uniform_real_distribution<float> uni(0.f, 1.f);

  int n_direct = 0;
  int32_t final_token = -1;

  // Sampling-only scratch, lazily sized. densify() expands one truncated row into a full vocab
  // vector (identical to the old dense row, so the discrete_distribution draw is unchanged);
  // correction_buf holds the built correction distribution. Greedy touches neither.
  std::vector<float> dense_row;
  std::vector<float> correction_buf;
  auto densify = [&](const SparseRow& sr) -> std::vector<float>& {
    dense_row.assign(static_cast<size_t>(vocab_size), 0.0f);
    for (size_t j = 0; j < sr.indices.size(); j++)
      dense_row[static_cast<size_t>(sr.indices[j])] = sr.probs[j];
    return dense_row;
  };
  auto sparse_prob = [](const SparseRow& sr, int32_t token) -> float {
    for (size_t j = 0; j < sr.indices.size(); j++)
      if (sr.indices[j] == token) return sr.probs[j];
    return 0.0f;
  };

  for (int i = 0; i < K; i++) {
    bool accepted = false;
    if (greedy_accept) {
      const int32_t ta = (i == 0) ? pos0_argmax : target_argmax[i - 1];
      accepted = (ta == proposal.tokens[i]);
    } else {
      const SparseRow& sr = (i == 0) ? pos0_sparse : target_sparse[i - 1];
      const float p_t = sparse_prob(sr, proposal.tokens[i]);
      const float p_d = proposal.probs[i][proposal.tokens[i]];
      accepted = (uni(rng_) < ComputeAcceptProb(p_t, p_d));
    }

    if (accepted) {
      n_direct++;
    } else {
      if (greedy_accept) {
        final_token = (i == 0) ? pos0_argmax : target_argmax[i - 1];
      } else {
        const SparseRow& sr = (i == 0) ? pos0_sparse : target_sparse[i - 1];
        std::vector<float>& dense_t = densify(sr);
        if (correction_buf.empty()) correction_buf.resize(static_cast<size_t>(vocab_size));
        BuildCorrectionDistribution(
            {dense_t.data(), static_cast<size_t>(vocab_size)},
            {proposal.probs[i].data(), static_cast<size_t>(vocab_size)},
            {correction_buf.data(), static_cast<size_t>(vocab_size)});
        std::discrete_distribution<int> dist(correction_buf.begin(), correction_buf.end());
        final_token = dist(rng_);
      }
      corrections_++;
      break;
    }
  }

  if (n_direct == K) {
    if (greedy_accept) {
      final_token = target_argmax[K - 1];
    } else {
      std::vector<float>& dense_last = densify(target_sparse[K - 1]);
      std::discrete_distribution<int> dist(dense_last.begin(), dense_last.end());
      final_token = dist(rng_);
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
  cached_direct_tokens_ = n_direct;
  reanchor_pending_ = true;
  // Verify moved the inner caches past the committed sequence; FinalizeRound/PrepareForAppend re-sync.
  round_dirty_ = true;

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
    ClearPendingExternalLogits();
    reanchor_pending_ = false;
    g.computed_logits_ = false;
    return;
  }

  const int32_t tok = pending_.front();
  pending_.pop_front();
  EmitToken(g, tok);
  g.computed_logits_ = false;

  if (!pending_.empty() && !guidance_round_) {
    if (emitted_direct_tokens_ >= cached_direct_tokens_)
      throw std::runtime_error("Speculative decoding has buffered tokens without matching target logits.");
    current_target_logits_row_ = pending_target_logits_row0_ + emitted_direct_tokens_;
    emitted_direct_tokens_++;
  }

  // Last token of the round just went out: re-anchor now. (Deferring it to here means an EOS
  // partway through the round skips the re-anchor and its wasted target pass.)
  if (pending_.empty() && reanchor_pending_) {
    ClearPendingExternalLogits();
    FinalizeRound(g);
  }
}

// Tidy up after a round and get ready for the next one. Verify ran all the proposed tokens, but
// we only kept n_direct of them, so first drop the rejected ones from the target's cache. Then
// handle the one committed token (final_token) in one of two ways:
//   * fold path (is_multiple_tokens && K>=2): leave it for the next round's verify batch (see RunRound).
//     This skips a whole extra target run each round.
//   * legacy path (K==1 / pruned target): run it through the target now. This keeps K==1
//     byte-for-byte identical to plain greedy decoding.
// Either way we then advance the draft model. Runs once the round's tokens have all been emitted.
void SpeculativeDecodingStrategy::FinalizeRound(Generator& g) {
  using clock = std::chrono::steady_clock;
  using ms_f = std::chrono::duration<float, std::milli>;

  reanchor_pending_ = false;

  // Guidance round - hand off to FinalizeGuidanceRound.
  if (guidance_round_) {
    guidance_round_ = false;
    if (g.search_->IsDone()) {
      g.computed_logits_ = false;
      return;  // leave round_dirty_ set so a later AppendTokens reconciles the caches
    }
    g.SetLogits(FinalizeGuidanceRound(g));
    round_dirty_ = false;
    return;
  }

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

  const bool fold = is_multiple_tokens_ && saved_K_ >= 2;
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
    target_runs_++;
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

  // Non-fold re-anchored the target to the committed length (caches match); the fold leaves the
  // target one token behind -> it stays dirty until the next round / PrepareForAppend.
  round_dirty_ = fold;
}

// Runs one guidance round over the K draft tokens - mask the target with the grammar, accept/verify
// each token, commit it, and add forced tokens. Greedy commits the draft's matching token; sampling
// uses speculative sampling on the masked distributions. Caches are re-anchored in FinalizeRound.
void SpeculativeDecodingStrategy::RunGuidanceRound(Generator& g, const Proposal& proposal,
                                                   int seed_length, int K, float propose_ms) {
  using clock = std::chrono::steady_clock;
  using ms_f = std::chrono::duration<float, std::milli>;

  auto* spec_state = static_cast<SpeculativeDecodingState*>(g.state_.get());
  const auto& params = *g.search_->params_;
  const int vocab_size = params.config.model.vocab_size;
  auto& proc = *g.guidance_logits_processor_;
  std::span<const int32_t> eos_ids{params.config.model.eos_token_id};
  auto is_eos = [&](int32_t t) {
    return std::find(eos_ids.begin(), eos_ids.end(), t) != eos_ids.end();
  };

  // Penalties (min-length + repetition) are applied after the grammar mask, in the regular order.
  // seed_prefix is the round-start sequence and grows as tokens are accepted (repetition context).
  // Built only when a penalty is active.
  const float repetition_penalty = params.search.repetition_penalty;
  const int min_length = params.search.min_length;
  const bool penalties_active = (repetition_penalty != 1.0f) || (min_length > 0);
  std::vector<int32_t> seed_prefix;
  std::vector<bool> rep_visited;
  if (penalties_active) {
    auto seed = g.search_->GetSequence(0).CopyDeviceToCpu();
    seed_prefix.assign(seed.begin(), seed.end());
  }

  // pos0 - the target's prediction for the first proposal position, read from the current logits
  // before the verify run overwrites them.
  std::vector<float> pos0;
  {
    auto cpu = g.search_->GetLogits().CopyDeviceToCpu();
    pos0.assign(cpu.data(), cpu.data() + vocab_size);
  }

  // Verify - run the target over the K proposed tokens in one pass (no anchor/fold under guidance).
  auto target_input = params.p_device->Allocate<int32_t>(static_cast<size_t>(K));
  {
    auto sp = target_input.CpuSpan();
    for (int i = 0; i < K; i++) sp[i] = proposal.tokens[i];
  }
  target_input.CopyCpuToDevice();

  auto t_target_start = clock::now();
  spec_state->target_state().Run(seed_length + K, target_input, {});
  target_runs_++;
  auto t_target_end = clock::now();

  // Read the K target rows (device-agnostic, mirrors the non-guidance verify read).
  std::vector<std::vector<float>> rows(static_cast<size_t>(K));
  const std::string& logits_name =
      spec_state->spec_model().target_model().config_->model.decoder.outputs.logits;
  OrtValue* raw_ort = spec_state->target_state().GetOutput(logits_name.c_str());
  if (!raw_ort)
    throw std::runtime_error("Speculative guidance: target state has no logits output named '" +
                             logits_name + "'.");
  auto raw_shape = raw_ort->GetTensorTypeAndShapeInfo()->GetShape();
  const bool is_multiple_tokens =
      (raw_shape.size() >= 2 && raw_shape[1] == static_cast<int64_t>(K));
  if (is_multiple_tokens) {
    auto& target_device = *spec_state->spec_model().target_model().p_device_inputs_;
    const auto elem_type = raw_ort->GetTensorTypeAndShapeInfo()->GetElementType();
    DeviceSpan<float> verify_logits;
    if (elem_type == Ort::TypeToTensorType<float>) {
      verify_logits = WrapTensor<float>(target_device, *raw_ort);
    } else {
      Cast(*raw_ort, verify_logits_fp32_, target_device, Ort::TypeToTensorType<float>);
      verify_logits = WrapTensor<float>(target_device, *verify_logits_fp32_);
    }
    const float* data = verify_logits.CopyDeviceToCpu().data();
    for (int i = 0; i < K; i++)
      rows[static_cast<size_t>(i)].assign(data + static_cast<ptrdiff_t>(i) * vocab_size,
                                          data + static_cast<ptrdiff_t>(i + 1) * vocab_size);
  } else {
    // Pruned target - one pass per token.
    spec_state->target_state().RewindTo(static_cast<size_t>(seed_length));
    auto single = params.p_device->Allocate<int32_t>(1);
    for (int i = 0; i < K; i++) {
      single.CpuSpan()[0] = proposal.tokens[i];
      single.CopyCpuToDevice();
      auto lgt = spec_state->target_state().Run(seed_length + i + 1, single, {});
      target_runs_++;
      auto cpu = lgt.CopyDeviceToCpu();
      rows[static_cast<size_t>(i)].assign(cpu.data(), cpu.data() + vocab_size);
    }
  }

  // Mask one logits row with the grammar (same ProcessLogits as the regular path), then apply
  // penalties. Returns the row so greedy can take the argmax and sampling can draw from it.
  auto mask_buf = params.p_device->Allocate<float>(static_cast<size_t>(vocab_size));
  auto masked_logits = [&](const std::vector<float>& row, int current_length,
                           const std::vector<int32_t>& prefix) -> std::vector<float> {
    auto cpu = mask_buf.CpuSpan();
    std::copy(row.begin(), row.end(), cpu.begin());
    mask_buf.CopyCpuToDevice();
    proc.ProcessLogits(mask_buf);
    auto out = mask_buf.CopyDeviceToCpu();
    std::vector<float> masked(out.data(), out.data() + static_cast<size_t>(vocab_size));
    if (penalties_active) {
      ApplyMinLengthToLogits({masked.data(), masked.size()}, current_length, min_length, eos_ids);
      ApplyRepetitionPenaltyToLogits({masked.data(), masked.size()}, prefix, repetition_penalty,
                                     rep_visited);
    }
    return masked;
  };
  auto commit_and_ff = [&](int32_t tok) {
    proc.CommitTokens({&tok, 1});
    return proc.GetFFTokens(0);
  };

  const bool sampling = !proposal.probs.empty();
  const auto& search = params.search;
  std::uniform_real_distribution<float> uni(0.f, 1.f);
  std::vector<float> correction_buf;  // sampling reject scratch

  // Forced tokens the grammar has already decided - prior round's overflow (ff_carry_) + each
  // accepted token's fast-forward span. A position filled from here is auto-accepted; it is in the
  // verify batch and the grammar is already advanced past it.
  std::deque<int32_t> pending_forced(ff_carry_.begin(), ff_carry_.end());

  std::vector<int32_t> committed;
  // accepted draft tokens (excludes forced tokens and corrections)
  int n_direct = 0;
  // leading committed tokens that came from the verify batch
  int verify_prefix = 0;  
  bool eos_hit = false, rejected = false;
  // repetition-penalty context, grows with committed tokens
  std::vector<int32_t> dist_prefix; 
  if (penalties_active) dist_prefix = seed_prefix;

  int i = 0;
  for (; i < K; i++) {
    if (!pending_forced.empty()) {
      // Forced position - auto-accept (verified in the batch, grammar already past it).
      const int32_t f = proposal.tokens[i];
      pending_forced.pop_front();
      committed.push_back(f);
      verify_prefix++;
      if (penalties_active) dist_prefix.push_back(f);
      if (is_eos(f)) { eos_hit = true; break; }
      continue;
    }

    // Free position - judge with pos0 (first token) or the batched verify row i-1.
    const std::vector<float>& judge = (i == 0) ? pos0 : rows[static_cast<size_t>(i - 1)];
    const std::vector<float> masked = masked_logits(judge, seed_length + i, dist_prefix);

    if (!sampling) {
      // Greedy - accept the draft token if it matches the target's masked argmax; commit the draft's
      // own token (not a batched row).
      const int32_t ttok = RowArgmax({masked.data(), static_cast<size_t>(vocab_size)});
      if (proposal.tokens[i] == ttok) {
        committed.push_back(proposal.tokens[i]);
        verify_prefix++;
        n_direct++;
        if (penalties_active) dist_prefix.push_back(proposal.tokens[i]);
        if (is_eos(proposal.tokens[i])) { eos_hit = true; break; }
        for (int32_t fwd : commit_and_ff(proposal.tokens[i])) pending_forced.push_back(fwd);
      } else {
        // Reject - only commit a correction at the first position, where pos0 is a single-token
        // result; later positions defer to the next round's per-token pos0.
        rejected = true;
        if (i == 0) {
          committed.push_back(ttok);
          corrections_++;
          if (!is_eos(ttok))
            for (int32_t fwd : commit_and_ff(ttok)) committed.push_back(fwd);
        }
        break;
      }
    } else {
      // Sampling - speculative sampling on the masked distributions. Accept with min(1, p_t/p_d), else
      // draw the correction from the leftover max(0, p_t - p_d).
      SampledCategorical tsc;
      ComputeSampledCategorical({masked.data(), static_cast<size_t>(vocab_size)},
                                search.top_k, search.top_p, search.temperature, tsc);
      const std::vector<float> p_t = ScatterToFullVocab(tsc, vocab_size);
      const int32_t dtok = proposal.tokens[i];
      const float accept_p = ComputeAcceptProb(p_t[static_cast<size_t>(dtok)],
                                               proposal.probs[i][static_cast<size_t>(dtok)]);
      if (uni(rng_) < accept_p) {
        committed.push_back(dtok);
        verify_prefix++;
        n_direct++;
        if (penalties_active) dist_prefix.push_back(dtok);
        if (is_eos(dtok)) { eos_hit = true; break; }
        for (int32_t fwd : commit_and_ff(dtok)) pending_forced.push_back(fwd);
      } else {
        rejected = true;
        if (correction_buf.empty()) correction_buf.resize(static_cast<size_t>(vocab_size));
        BuildCorrectionDistribution({p_t.data(), static_cast<size_t>(vocab_size)},
                                    {proposal.probs[i].data(), static_cast<size_t>(vocab_size)},
                                    {correction_buf.data(), static_cast<size_t>(vocab_size)});
        std::discrete_distribution<int> dist(correction_buf.begin(), correction_buf.end());
        const int32_t ctok = static_cast<int32_t>(dist(rng_));
        committed.push_back(ctok);
        corrections_++;
        if (!is_eos(ctok))
          for (int32_t fwd : commit_and_ff(ctok)) committed.push_back(fwd);
        break;
      }
    }
  }

  // Forced tokens that didn't fit this round's K budget carry to the next round - the grammar is
  // already past them. Only when we reached K cleanly - EOS ends generation and a reject leaves
  // nothing pending.
  ff_carry_.clear();
  if (!eos_hit && !rejected)
    ff_carry_.assign(pending_forced.begin(), pending_forced.end());

  for (int32_t t : committed) pending_.push_back(t);
  saved_seed_length_ = seed_length;
  saved_K_ = K;
  saved_n_direct_ = n_direct;
  saved_verify_prefix_ = verify_prefix;
  saved_committed_ = committed;
  saved_last_row_.clear();
  if (verify_prefix >= 1)
    saved_last_row_ = rows[static_cast<size_t>(verify_prefix - 1)];
  reanchor_pending_ = true;
  guidance_round_ = true;
  round_dirty_ = true;

  rounds_++;
  draft_proposed_ += static_cast<size_t>(K);
  draft_accepted_ += static_cast<size_t>(n_direct);
  total_propose_ms_ += propose_ms;
  total_target_ms_ += ms_f(t_target_end - t_target_start).count();
}

// Cleans up after a guidance round and returns the target's logits for the next position. Keeps the
// accepted tokens the verify already cached and feeds the rest back one at a time (re-running the
// last token alone if none are left). Also replays the committed tokens through the draft.
DeviceSpan<float> SpeculativeDecodingStrategy::FinalizeGuidanceRound(Generator& g) {
  auto* spec_state = static_cast<SpeculativeDecodingState*>(g.state_.get());
  const auto& params = *g.search_->params_;
  const int vocab_size = params.config.model.vocab_size;
  const int C = static_cast<int>(saved_committed_.size());
  const int seed = saved_seed_length_;

  auto single = params.p_device->Allocate<int32_t>(1);

  // Draft - replay every committed token from the round's start and save the final logits for the
  // next Propose. Only rewind when the draft cache is past seed (rewinding to the current length is
  // rejected by the cache).
  const int draft_kv_len = seed + saved_K_ - 1;
  if (seed < draft_kv_len)
    spec_state->draft_state().RewindTo(static_cast<size_t>(seed));
  DeviceSpan<float> draft_logits;
  for (int p = 0; p < C; p++) {
    single.CpuSpan()[0] = saved_committed_[static_cast<size_t>(p)];
    single.CopyCpuToDevice();
    draft_logits = spec_state->draft_state().Run(seed + p + 1, single, {});
  }
  auto dcpu = draft_logits.CopyDeviceToCpu();
  spec_state->assign_draft_pending_logits(dcpu.data(), static_cast<size_t>(vocab_size));

  // Target - the verify batch already built KV for the first verify_prefix committed tokens. Trim to
  // there and replay only the tail (a reject correction + its fast-forward tokens), which was never
  // in the batch. When there is no tail, reuse the saved verify row as pos0.
  const int target_kv_len = seed + saved_K_;
  const int vp = saved_verify_prefix_;
  if (C == vp && !saved_last_row_.empty()) {
    if (vp < saved_K_)
      spec_state->target_state().RewindTo(static_cast<size_t>(seed + vp));
    auto out = params.p_device->Allocate<float>(static_cast<size_t>(vocab_size));
    std::copy(saved_last_row_.begin(), saved_last_row_.end(), out.CpuSpan().begin());
    out.CopyCpuToDevice();
    return out;
  }
  if (seed + vp < target_kv_len)
    spec_state->target_state().RewindTo(static_cast<size_t>(seed + vp));
  DeviceSpan<float> target_logits;
  for (int p = vp; p < C; p++) {
    single.CpuSpan()[0] = saved_committed_[static_cast<size_t>(p)];
    single.CopyCpuToDevice();
    target_logits = spec_state->target_state().Run(seed + p + 1, single, {});
    target_runs_++;
  }
  return target_logits;
}

// Commit one already-decided token. The speculative accept logic picked it, so hand it straight
// to the search (device-agnostic): no vocab-sized one-hot + no host->device logits copy per token,
// just the token append with the usual EOS / max-length bookkeeping.
void SpeculativeDecodingStrategy::EmitToken(Generator& g, int32_t tok) {
  g.search_->CommitToken(tok);
}

SpeculativeStats SpeculativeDecodingStrategy::GetStats() const {
  SpeculativeStats s{};
  s.rounds = rounds_;
  s.draft_tokens_proposed = draft_proposed_;
  s.draft_tokens_accepted = draft_accepted_;
  s.correction_tokens = corrections_;
  s.bonus_tokens = bonuses_;
  s.target_forward_passes = target_runs_;
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
