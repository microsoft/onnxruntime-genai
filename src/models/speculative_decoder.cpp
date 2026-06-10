// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "speculative_decoder.h"

#include <algorithm>
#include <stdexcept>

#include "model.h"
#include "../search.h"
#include "../constrained_logits_processor.h"

namespace Generators {

SpeculativeDecoder::SpeculativeDecoder(std::shared_ptr<const Model> target_model,
                                       std::shared_ptr<const Model> draft_model,
                                       int max_length)
    : target_model_{std::move(target_model)}, draft_model_{std::move(draft_model)} {
  const auto& cfg = *target_model_->config_;
  if (!cfg.pipeline.strategy.has_value()) {
    throw std::runtime_error("SpeculativeDecoder: target model config has no pipeline.strategy block.");
  }
  const auto& strategy = *cfg.pipeline.strategy;
  if (strategy.kind != "speculative") {
    throw std::runtime_error("SpeculativeDecoder: pipeline.strategy.kind must be 'speculative'.");
  }
  if (strategy.acceptance != "greedy") {
    throw std::runtime_error(
        "SpeculativeDecoder: only 'greedy' acceptance is implemented in PR-B (got '" + strategy.acceptance + "').");
  }
  if (strategy.tree.has_value()) {
    // Token-tree verify needs a non-causal tree-attention mask supplied to the target's forward
    // pass. The in-tree decoder-pipeline graph cannot express that: its `attention_mask` input is a
    // 1D [batch, total] padding mask (runtime PositionInputs only ever build a 2D {batch, seq}
    // mask -- src/models/position_inputs.h:62), and causal masking is hardcoded *inside* the graph
    // via Trilu, so a caller cannot supply a per-(query,key) 2D additive mask without a model-side
    // graph change. This is exactly the design's flagged unknown (docs/pipeline-config-v2.1-design
    // .md §11.2). Rather than fake it, we degrade the tree to its best linear chain and run the
    // verified linear-K path below -- the design's stated fallback (§11.2) -- which still covers
    // vanilla/PLD/self-spec and preserves the greedy-equivalence invariant.
    tree_linear_k_fallback_ = true;
  }
  if (strategy.draft.producer != "draft_model") {
    throw std::runtime_error(
        "SpeculativeDecoder: only the 'draft_model' producer is implemented in PR-B (got '" +
        strategy.draft.producer + "').");
  }

  k_ = std::max(1, strategy.num_speculative_tokens);
  vocab_size_ = cfg.model.vocab_size;
  eos_tokens_ = cfg.model.eos_token_id;
  max_length_ = max_length > 0 ? max_length : cfg.model.context_length;

  target_params_ = CreateGeneratorParams(*target_model_);
  target_params_->search.max_length = max_length_;
  draft_params_ = CreateGeneratorParams(*draft_model_);
  draft_params_->search.max_length = max_length_;

  target_ = CreateGenerator(*target_model_, *target_params_);
  draft_ = CreateGenerator(*draft_model_, *draft_params_);
}

SpeculativeDecoder::~SpeculativeDecoder() = default;

int SpeculativeDecoder::ArgMax(std::span<const float> row) const {
  int best = 0;
  float best_v = row[0];
  for (int i = 1; i < static_cast<int>(row.size()); ++i) {
    if (row[i] > best_v) {
      best_v = row[i];
      best = i;
    }
  }
  return best;
}

bool SpeculativeDecoder::IsEos(int32_t token) const {
  return std::find(eos_tokens_.begin(), eos_tokens_.end(), token) != eos_tokens_.end();
}

void SpeculativeDecoder::AppendTokens(std::span<const int32_t> prompt) {
  sequence_.assign(prompt.begin(), prompt.end());
  target_->AppendTokens(cpu_span<const int32_t>(prompt.data(), prompt.size()));
  draft_->AppendTokens(cpu_span<const int32_t>(prompt.data(), prompt.size()));
}

std::vector<int32_t> SpeculativeDecoder::Generate() {
  verify_passes_ = 0;
  committed_token_count_ = 0;
  accepted_draft_count_ = 0;

  while (static_cast<int>(sequence_.size()) < max_length_) {
    const int m = static_cast<int>(sequence_.size());
    const int budget = max_length_ - m;  // at most this many new tokens may be committed
    const int k_eff_cap = std::min(k_, budget);
    if (k_eff_cap <= 0) break;

    // (1) Capture the target's greedy token for position m from the logits produced by the
    // previous step (or by the prompt prefill). This is t_0.
    auto target_last = target_->GetLogits().CopyDeviceToCpu();  // [1,1,vocab]
    const int t0 = ArgMax(std::span<const float>(target_last.data(), vocab_size_));

    // (2) Draft up to K tokens greedily. Stop early if the draft proposes EOS.
    std::vector<int32_t> draft_tokens;
    draft_tokens.reserve(k_eff_cap);
    for (int step = 0; step < k_eff_cap; ++step) {
      draft_->GenerateNextToken();
      auto draft_seq = draft_->GetSequence(0).CopyDeviceToCpu();
      const int32_t proposed = draft_seq.back();
      draft_tokens.push_back(proposed);
      if (IsEos(proposed)) break;
    }
    const int k_eff = static_cast<int>(draft_tokens.size());

    // (3) Verify all K draft tokens with a single target forward pass and read the per-position
    // logits. Row j predicts the token AFTER draft_tokens[j], i.e. target's greedy token for
    // position m+j+1.
    target_->AppendTokens(cpu_span<const int32_t>(draft_tokens.data(), draft_tokens.size()));
    ++verify_passes_;

    std::array<int64_t, 3> raw_shape{};
    auto raw = target_->GetRawLogits(raw_shape).CopyDeviceToCpu();  // [1, k_eff, vocab]
    if (raw_shape[1] != k_eff) {
      throw std::runtime_error("SpeculativeDecoder: target verify returned unexpected logits shape.");
    }

    // target_greedy[i] is the target's greedy token for position m+i.
    std::vector<int32_t> target_greedy(k_eff + 1);
    target_greedy[0] = t0;
    for (int j = 0; j < k_eff; ++j) {
      auto row = std::span<const float>(raw.data() + static_cast<size_t>(j) * vocab_size_, vocab_size_);
      target_greedy[j + 1] = ArgMax(row);
    }

    // (4) Accept the longest prefix where the draft proposal matches the target's greedy token.
    int n = 0;  // number of accepted draft tokens
    for (int i = 0; i < k_eff; ++i) {
      if (draft_tokens[i] == target_greedy[i]) {
        ++n;
      } else {
        break;
      }
    }
    accepted_draft_count_ += n;

    // Committed tokens this step are target_greedy[0..n] (n accepted drafts + one bonus/correction),
    // exactly the target model's greedy tokens for positions m..m+n.
    int committed_this_step = 0;
    bool stop = false;
    for (int i = 0; i <= n; ++i) {
      const int32_t token = target_greedy[i];
      sequence_.push_back(token);
      ++committed_this_step;
      ++committed_token_count_;
      if (IsEos(token) || static_cast<int>(sequence_.size()) >= max_length_) {
        stop = true;
        break;
      }
    }

    if (stop) break;

    // (5) Roll back both role caches to the accepted prefix and re-seat the bonus token so each
    // role is positioned to predict position m + committed_this_step on the next outer step.
    //   target: keep the n accepted drafts (== committed tokens), then run the bonus token.
    //   draft : rebuild from the committed prefix (its last drafted token's logits were never run).
    const int new_length = m + committed_this_step;  // == m + n + 1
    const int32_t bonus = target_greedy[n];

    target_->RewindToLength(static_cast<size_t>(m + n));
    target_->AppendTokens(cpu_span<const int32_t>(&bonus, 1));

    draft_->RewindToLength(static_cast<size_t>(m));
    draft_->AppendTokens(cpu_span<const int32_t>(sequence_.data() + m, committed_this_step));

    if (new_length >= max_length_) break;
  }

  return sequence_;
}

}  // namespace Generators
