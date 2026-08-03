// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "eagle_speculative_strategy.h"

#include "generators.h"
#include "models/model.h"
#include "search.h"

#include <chrono>

namespace Generators {

namespace {

float ElapsedMilliseconds(std::chrono::steady_clock::time_point start,
                          std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<float, std::milli>(end - start).count();
}

}  // namespace

EagleSpeculativeStrategy::EagleSpeculativeStrategy(
    Generator& generator)
    : SpeculativeDecodingStrategy{
          dynamic_cast<EagleState&>(*generator.state_).target_state(),
          dynamic_cast<EagleState&>(*generator.state_)
              .eagle_model().target_model()},
      state_{dynamic_cast<EagleState&>(*generator.state_)} {}

SpeculativeDecodingStrategy::Proposal EagleSpeculativeStrategy::Propose(
    Generator&, int, int) {
  throw std::runtime_error(
      "EAGLE tree decoding does not use the linear proposal interface.");
}

void EagleSpeculativeStrategy::Advance(
    Generator&, const Proposal&, int, int32_t, int) {
  throw std::runtime_error(
      "EAGLE tree decoding advances its draft state during round finalization.");
}

void EagleSpeculativeStrategy::ResetProposer() {
  selected_tree_indices_.clear();
  tree_ = {};
}

void EagleSpeculativeStrategy::ReconcileProposer(
    Generator&, int, std::span<const int32_t>, int committed_length, bool) {
  auto& draft = state_.draft_state();
  if (draft.cache_length() > static_cast<size_t>(committed_length))
    draft.RewindTo(static_cast<size_t>(committed_length));
}

void EagleSpeculativeStrategy::FinalizeGuidanceProposer(
    Generator&, int, int, std::span<const int32_t>) {
  throw std::runtime_error("EAGLE tree decoding does not support guidance.");
}

int32_t EagleSpeculativeStrategy::ArgMax(
    std::span<const float> logits) const {
  if (logits.empty())
    throw std::runtime_error("Cannot select a token from empty target logits.");
  return static_cast<int32_t>(
      std::max_element(logits.begin(), logits.end()) - logits.begin());
}

int32_t EagleSpeculativeStrategy::PrepareDraft(
    Generator& g, bool& folded_root) {
  auto& target = state_.target_state();
  auto& draft = state_.draft_state();
  const auto sequence = g.GetSequence(0).CopyDeviceToCpu();

  int32_t root_token;
  folded_root = HasPendingAnchorToken();
  if (folded_root) {
    root_token = TakePendingAnchorToken();
  } else {
    const int vocab_size = g.search_->params_->config.model.vocab_size;
    const auto logits = g.GetLogits().CopyDeviceToCpu();
    if (logits.size() < static_cast<size_t>(vocab_size))
      throw std::runtime_error(
          "EAGLE target logits are unavailable for tree construction.");
    root_token = ArgMax(logits.subspan(
        logits.size() - static_cast<size_t>(vocab_size),
        static_cast<size_t>(vocab_size)));
  }

  const size_t stable_length = target.stable_length();
  const size_t draft_length = draft.cache_length();
  if (draft_length > stable_length)
    throw std::runtime_error(
        "EAGLE draft cache is ahead of the target cache.");

  if (draft_length < stable_length) {
    auto features =
        target.CopyFeatures(draft_length, stable_length - draft_length);
    std::vector<int32_t> shifted_ids(stable_length - draft_length);
    for (size_t row = 0; row < shifted_ids.size(); ++row) {
      const size_t next_position = draft_length + row + 1;
      shifted_ids[row] =
          next_position < sequence.size() ? sequence[next_position] : root_token;
    }
    draft.Prepare(std::move(features), shifted_ids);
    ++draft_runs_;
    target.DiscardFeaturesBefore(stable_length - 1);
  } else if (!draft.initialized()) {
    throw std::runtime_error(
        "EAGLE draft state has no target features for the current cache.");
  }

  if (draft.conditioning_token() != root_token) {
    if (stable_length == 0)
      throw std::runtime_error(
          "EAGLE cannot recondition an empty stable draft state.");
    draft.RewindTo(stable_length - 1);
    auto features = target.CopyFeatures(stable_length - 1, 1);
    const std::array<int32_t, 1> shifted_ids{root_token};
    draft.Prepare(std::move(features), shifted_ids);
    ++draft_runs_;
  }
  return root_token;
}

void EagleSpeculativeStrategy::RunRound(Generator& g) {
  const auto& params = *g.search_->params_;
  if (!g.IsGreedySampling())
    throw std::runtime_error(
        "EAGLE tree speculative decoding currently supports greedy search only.");
  if (params.search.batch_size != 1 || params.search.num_beams != 1)
    throw std::runtime_error(
        "EAGLE tree speculative decoding requires batch size 1 and one beam.");

  ClearPendingExternalLogits();
  auto& round = CurrentRound();
  if (round.phase != RoundState::Phase::kIdle &&
      round.phase != RoundState::Phase::kReconcilePending)
    throw std::runtime_error(
        "Cannot start an EAGLE tree round while another round is active.");

  const auto propose_start = std::chrono::steady_clock::now();
  bool folded_root = false;
  const int32_t root_token = PrepareDraft(g, folded_root);
  tree_ = state_.draft_state().BuildTree();
  draft_runs_ +=
      static_cast<size_t>(
          state_.eagle_model().config_->model.eagle->depth);
  const auto propose_end = std::chrono::steady_clock::now();

  const size_t node_count = tree_.tokens.size();
  if (node_count !=
          static_cast<size_t>(
              state_.eagle_model().config_->model.eagle->total_tokens) ||
      tree_.position_ids.size() != node_count ||
      tree_.attention_mask.size() != node_count * node_count)
    throw std::runtime_error(
        "EAGLE draft model returned an invalid fixed tree.");
  if (tree_.tokens.front() != root_token)
    throw std::runtime_error(
        "EAGLE draft tree root does not match its conditioning token.");

  std::vector<int64_t> absolute_positions(node_count);
  const int64_t position_base =
      static_cast<int64_t>(state_.target_state().stable_length());
  std::transform(tree_.position_ids.begin(), tree_.position_ids.end(),
                 absolute_positions.begin(),
                 [position_base](int64_t position) {
                   return position_base + position;
                 });

  auto tree_tokens = params.p_device->Allocate<int32_t>(node_count);
  std::copy(tree_.tokens.begin(), tree_.tokens.end(),
            tree_tokens.CpuSpan().begin());
  tree_tokens.CopyCpuToDevice();
  auto target_start = std::chrono::steady_clock::now();
  auto float_logits = state_.target_state().RunTree(
      tree_tokens, absolute_positions, tree_.attention_mask);
  ++target_runs_;
  ++target_verify_runs_;
  auto target_end = std::chrono::steady_clock::now();

  const size_t vocab_size =
      static_cast<size_t>(params.config.model.vocab_size);
  if (float_logits.size() != node_count * vocab_size)
    throw std::runtime_error(
        "EAGLE target tree logits have an invalid shape.");
  const auto logits = float_logits.CopyDeviceToCpu();

  size_t best_path = 0;
  size_t best_accepted = 0;
  for (size_t path_index = 0;
       path_index < tree_.retrieve_indices.size(); ++path_index) {
    const auto& path = tree_.retrieve_indices[path_index];
    size_t accepted = 0;
    for (size_t depth = 1; depth < path.size(); ++depth) {
      const size_t parent = path[depth - 1];
      const size_t child = path[depth];
      const int32_t target_token = ArgMax(std::span<const float>{
          logits.data() + parent * vocab_size, vocab_size});
      if (tree_.tokens[child] != target_token)
        break;
      ++accepted;
    }
    if (accepted > best_accepted) {
      best_accepted = accepted;
      best_path = path_index;
    }
  }

  const auto& chosen_path = tree_.retrieve_indices[best_path];
  selected_tree_indices_.assign(chosen_path.begin(),
                                chosen_path.begin() + best_accepted + 1);
  const size_t final_logit_row = selected_tree_indices_.back();
  const int32_t final_token = ArgMax(std::span<const float>{
      logits.data() + final_logit_row * vocab_size, vocab_size});

  round.committed.clear();
  round.committed.reserve(selected_tree_indices_.size() + 1);
  for (size_t index : selected_tree_indices_)
    round.committed.push_back(tree_.tokens[index]);
  round.committed.push_back(final_token);
  state_.target_state().CompactTree(selected_tree_indices_);

  round.target_logits =
      params.p_device->Allocate<float>(selected_tree_indices_.size() *
                                       vocab_size);
  auto pending_cpu = round.target_logits.CpuSpan();
  for (size_t row = 0; row < selected_tree_indices_.size(); ++row) {
    const auto source = std::span<const float>{
        logits.data() + selected_tree_indices_[row] * vocab_size, vocab_size};
    std::copy(source.begin(), source.end(),
              pending_cpu.begin() + row * vocab_size);
  }
  round.target_logits.CopyCpuToDevice();

  const size_t first_queued_tree_row = folded_root ? 1 : 0;
  round.target_logits_row0 =
      static_cast<int>(first_queued_tree_row);
  round.current_target_logits_row = -1;
  round.cached_direct_tokens =
      static_cast<int>(selected_tree_indices_.size() -
                       first_queued_tree_row);
  round.emitted_direct_tokens = 0;
  round.pending.clear();
  round.pending.insert(round.pending.end(),
                       round.committed.begin() +
                           static_cast<ptrdiff_t>(first_queued_tree_row),
                       round.committed.end());
  round.final_token = final_token;
  round.n_direct = static_cast<int>(best_accepted);
  round.seed_length = g.search_->GetSequenceLength();
  round.k = static_cast<int>(tree_.tokens.size() - 1);
  round.accepted = static_cast<int>(best_accepted);
  round.evaluated = static_cast<int>(tree_.tokens.size() - 1);
  round.propose_ms = ElapsedMilliseconds(propose_start, propose_end);
  round.target_ms = ElapsedMilliseconds(target_start, target_end);

  total_propose_ms_ += round.propose_ms;
  total_target_ms_ += round.target_ms;
  if (best_accepted + 1 == chosen_path.size())
    ++bonuses_;
  else
    ++corrections_;

  BeginRound(round.k, round.evaluated, round.accepted,
             round.pending.size(), false, true, round.propose_ms,
             round.target_ms, RoundState::Kind::kStandard);
}

SpeculativeDecodingStrategy::RoundState::Phase
EagleSpeculativeStrategy::FinalizeRound(Generator&) {
  auto& round = CurrentRound();
  if (selected_tree_indices_.empty() ||
      round.committed.size() != selected_tree_indices_.size() + 1)
    throw std::runtime_error(
        "EAGLE tree round has no accepted target path to finalize.");

  auto& target = state_.target_state();
  const size_t stable_start = state_.draft_state().cache_length();
  if (target.stable_length() !=
      stable_start + selected_tree_indices_.size())
    throw std::runtime_error(
        "EAGLE compacted target and stable draft lengths are inconsistent.");
  auto features =
      target.CopyFeatures(stable_start, selected_tree_indices_.size());
  std::vector<int32_t> shifted_ids(round.committed.begin() + 1,
                                   round.committed.end());
  state_.draft_state().Prepare(std::move(features), shifted_ids);
  ++draft_runs_;
  target.DiscardFeaturesBefore(target.stable_length() - 1);
  SetPendingAnchorToken(round.committed.back());
  selected_tree_indices_.clear();
  tree_ = {};
  return RoundState::Phase::kReconcilePending;
}

}  // namespace Generators
