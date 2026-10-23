// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(__linux__)
#include "decoding/qwen38_graph_strategy.h"

#include <algorithm>
#include <stdexcept>

#include "generator/generators.h"
#include "models/model.h"
#include "models/qwen38_graph.h"
#include "search.h"

namespace Generators {

namespace {

Qwen38GraphState& GraphState(Generator& generator) {
  auto* state = dynamic_cast<Qwen38GraphState*>(generator.state_.get());
  if (!state) throw std::logic_error("Qwen38GraphStrategy requires a Qwen38GraphState");
  return *state;
}

}  // namespace

Qwen38GraphStrategy::Qwen38GraphStrategy(Generator& generator) : state_{GraphState(generator)} {}

bool Qwen38GraphStrategy::IsEos(const Generator& generator, int64_t token) const {
  const auto& eos = generator.model_->config_->model.eos_token_id;
  return std::find(eos.begin(), eos.end(), static_cast<int>(token)) != eos.end();
}

void Qwen38GraphStrategy::Step(Generator& g) {
  Qwen38GraphEngine& engine = state_.engine();
  Search& search = *g.search_;
  const int max_length = search.params_->search.max_length;
  if (search.GetSequenceLength() >= max_length)
    throw std::runtime_error("qwen38_graph: generation has already reached max_length");

  if (queue_.empty()) {
    const int length = search.GetSequenceLength();
    if (!started_) {
      if (length == 0 || engine.Length() == 0)
        throw std::runtime_error(
            "GenerateNextToken called with no prior state. Please call AppendTokens before calling GenerateNextToken.");
      // The first token is the argmax of the prompt's last row. Head prefill, the first draft and the first verify are
      // enqueued before it is returned so that decode timing starts with the GPU already busy.
      const int32_t first = engine.PendingToken();
      const bool more = !IsEos(g, first) && length + 1 < max_length;
      engine.BeginDecode(more);
      queue_.push_back(first);
      ++stats_.tokens_queued;
      started_ = true;
    } else {
      const Qwen38CycleResult result = engine.WaitCycle();
      const int32_t tokens[4] = {
          static_cast<int32_t>(result.pred0), static_cast<int32_t>(result.pred1),
          static_cast<int32_t>(result.pred2), static_cast<int32_t>(result.pred3)};
      size_t count = std::min(result.TokenCount(), static_cast<size_t>(max_length - length));
      bool hit_eos = false;
      for (size_t i = 0; i < count; ++i) {
        if (IsEos(g, tokens[i])) {
          count = i + 1;
          hit_eos = true;
          break;
        }
      }
      const bool more = !hit_eos && length + static_cast<int>(count) < max_length;
      if (count > 1) engine.SaveBufferedLogits(count - 1);
      engine.Advance(result, more);
      for (size_t i = 0; i < count; ++i) queue_.push_back(tokens[i]);
      stats_.tokens_queued += result.TokenCount();
      ++stats_.rounds;
      ++stats_.completed_rounds;
      ++stats_.target_forward_passes;
      stats_.tokens_discarded += result.TokenCount() - count;
      stats_.ngram_lookup_hits += result.suffix_lookup_hit ? 1 : 0;
      stats_.ngram_lookup_misses += result.suffix_lookup_attempted && !result.suffix_lookup_hit ? 1 : 0;
      stats_.ngram_lookup_tokens_proposed += result.suffix_lookup_tokens_proposed;
      if (engine.mtp_enabled()) {
        const size_t proposed = static_cast<size_t>(result.proposed_count);
        const size_t accepted = std::min(result.AcceptedCount(), count);
        ++stats_.target_verify_forward_passes;
        stats_.draft_forward_passes += proposed;
        stats_.draft_tokens_proposed += proposed;
        stats_.draft_tokens_evaluated += std::min(proposed, accepted + 1);
        stats_.draft_tokens_accepted += accepted;
        ++stats_.acceptance_length_histogram[accepted];
        if (accepted == proposed) {
          ++stats_.full_accept_rounds;
          if (count > proposed) ++stats_.bonus_tokens;
        } else if (accepted == 0) {
          ++stats_.zero_accept_rounds;
          if (count > 0) ++stats_.correction_tokens;
        } else {
          ++stats_.partial_accept_rounds;
          if (count > accepted) ++stats_.correction_tokens;
        }
      }
    }
    g.computed_logits_ = false;
  }

  const int32_t token = queue_.front();
  g.last_action_ = Generator::Action::generated;
  search.CommitToken(token);
  queue_.pop_front();
  ++stats_.tokens_emitted;
}

SpeculativeStats Qwen38GraphStrategy::GetStats() const {
  SpeculativeStats s = stats_;
  s.tokens_buffered = queue_.size();
  if (s.draft_tokens_evaluated > 0)
    s.acceptance_rate = static_cast<float>(s.draft_tokens_accepted) / static_cast<float>(s.draft_tokens_evaluated);
  if (s.completed_rounds > 0) {
    s.mean_emitted_tokens_per_round = static_cast<float>(s.tokens_emitted) / static_cast<float>(s.completed_rounds);
    s.avg_draft_tokens_per_round = static_cast<float>(s.draft_tokens_proposed) / static_cast<float>(s.completed_rounds);
  }
  return s;
}

void Qwen38GraphStrategy::Reset() {
  queue_.clear();
  stats_ = {};
  started_ = false;
  external_logits_ = {};
}

bool Qwen38GraphStrategy::TryGetExternalLogits(Generator& g, DeviceSpan<float>& logits) {
  if (!started_) return false;
  if (g.search_->IsDone())
    throw std::runtime_error("qwen38_flash_graph logits are unavailable after generation is complete.");
  if (external_logits_.empty())
    external_logits_ = g.search_->params_->p_device->Allocate<float>(static_cast<size_t>(g.model_->config_->model.vocab_size));
  state_.engine().DecodeLogits(queue_.size(), external_logits_.CpuSpan());
  external_logits_.CopyCpuToDevice();
  logits = external_logits_;
  return true;
}

void Qwen38GraphStrategy::PrepareForAppend(Generator&) {
  if (started_)
    throw std::runtime_error(
        "qwen38_flash_graph does not support AppendTokens after GenerateNextToken (continuous decoding): the recurrent "
        "state cannot be cropped to an arbitrary token. Create a new Generator or call RewindToLength(0).");
}

void Qwen38GraphStrategy::PrepareForSetLogits(Generator&) {
  throw std::runtime_error("qwen38_flash_graph does not support SetLogits (token selection is done by the engine).");
}

}  // namespace Generators
#endif  // defined(__linux__)
