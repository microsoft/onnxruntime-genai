// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "n_gram_decoding_strategy.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>

#include "generators.h"
#include "models/model.h"
#include "constrained_logits_processor.h"
#include "models/model_type.h"
#include "openvino/interface.h"
#include "qnn/interface.h"
#include "search.h"
#include "speculative_stats.h"

namespace Generators {

namespace {

int ValidateAndGetNGramSize(Generator& g) {
  ValidateNGramDecoding(*g.model_, *g.search_->params_);
  const auto& params = *g.search_->params_;
  const bool guidance_requested =
      !params.guidance_type.empty() || !params.guidance_data.empty();
  if (guidance_requested && !g.guidance_logits_processor_)
    throw std::runtime_error(
        "N-gram decoding guidance was requested, but guidance is unavailable. "
        "Build ONNX Runtime GenAI with guidance support.");
  return g.search_->params_->speculative.ngram_size;
}

}  // namespace

void ValidateNGramDecoding(const Model& model, const GeneratorParams& params) {
  if (params.guidance_type.empty() != params.guidance_data.empty())
    throw std::runtime_error(
        "N-gram decoding guidance type and data must be provided together.");

  const auto& config = model.config_->model;
  const bool uses_sliding_kv_cache =
      config.decoder.sliding_window &&
      config.decoder.sliding_window->slide_key_value_cache;
  const bool is_plain_decoder_only_text =
      ModelType::IsLLM(config.type) && !ModelType::IsLFM2(config.type) &&
      config.decoder.pipeline.empty() && config.vision.filename.empty() &&
      config.speech.filename.empty();

  NGramDecodingCapabilities capabilities;
  capabilities.uses_guidance =
      !params.guidance_type.empty() || !params.guidance_data.empty();
  capabilities.uses_draft_model = config.draft.has_value();
  capabilities.is_plain_decoder_only_text = is_plain_decoder_only_text;
  capabilities.uses_sliding_kv_cache = uses_sliding_kv_cache;
  capabilities.uses_hybrid_state = !config.decoder.layer_types.empty();
  capabilities.uses_model_managed_state =
      IsOpenVINOStatefulModel(model) || IsQNNStatefulModel(model);
  capabilities.has_pruned_logits = is_plain_decoder_only_text && model.IsPruned();
  ValidateNGramDecodingCapabilities(params.search, capabilities);
}

NGramDecodingStrategy::NGramDecodingStrategy(Generator& g)
    : SpeculativeDecodingStrategy{*g.state_, *g.model_},
      lookup_{ValidateAndGetNGramSize(g)},
      chained_lookup_{g.search_->params_->speculative.ngram_chained_lookup} {}

void NGramDecodingStrategy::Sync(Generator& g) {
  using clock = std::chrono::steady_clock;
  using ms_f = std::chrono::duration<float, std::milli>;

  auto committed = g.search_->GetSequence(0);
  const size_t indexed_length = lookup_.HistorySize();

  if (committed.size() < indexed_length) {
    const auto sync_start = clock::now();
    lookup_.Reset(committed.CopyDeviceToCpu());
    RecordHistorySync(committed.size(), ms_f(clock::now() - sync_start).count());
    return;
  }
  if (committed.size() == indexed_length)
    return;

  auto suffix = committed.subspan(indexed_length, committed.size() - indexed_length);
  const auto sync_start = clock::now();
  lookup_.Append(suffix.CopyDeviceToCpu());
  RecordHistorySync(suffix.size(), ms_f(clock::now() - sync_start).count());
}

void NGramDecodingStrategy::RecordHistorySync(size_t token_count, float elapsed_ms) {
  history_syncs_++;
  history_tokens_synced_ += token_count;
  total_history_sync_ms_ += elapsed_ms;
}

void NGramDecodingStrategy::RecordLookup(std::span<const int32_t> candidates,
                                         size_t lookup_tokens_proposed,
                                         float elapsed_ms) {
  if (candidates.empty())
    lookup_misses_++;
  else
    lookup_hits_++;
  lookup_tokens_proposed_ += lookup_tokens_proposed;
  if (chained_lookup_ && lookup_tokens_proposed > 1)
    chained_tokens_proposed_ += lookup_tokens_proposed - 1;
  total_lookup_ms_ += elapsed_ms;
}

SpeculativeDecodingStrategy::Proposal NGramDecodingStrategy::Propose(
  Generator& g, int K, int seed_length, std::mt19937& round_rng) {
  using clock = std::chrono::steady_clock;
  using ms_f = std::chrono::duration<float, std::milli>;

  (void)seed_length;
  (void)round_rng;
  Sync(g);
  Proposal proposal{ProposalMode::kDeterministic};
  if (!g.guidance_logits_processor_) {
    const auto lookup_start = clock::now();
    proposal.tokens = lookup_.Propose(static_cast<size_t>(K), chained_lookup_);
    RecordLookup(proposal.tokens, proposal.tokens.size(),
                 ms_f(clock::now() - lookup_start).count());
    return proposal;
  }

  const auto& params = *g.search_->params_;
  auto ff_queue = CreateGuidanceFFQueue();
  proposal.tokens.reserve(static_cast<size_t>(K));

  if (!ff_queue.empty()) {
    while (!ff_queue.empty() && static_cast<int>(proposal.tokens.size()) < K) {
      proposal.tokens.push_back(ff_queue.front());
      ff_queue.pop_front();
    }
    return proposal;
  }

  const auto lookup_start = clock::now();
  auto candidates = lookup_.Propose(static_cast<size_t>(K), chained_lookup_);
  const float lookup_ms = ms_f(clock::now() - lookup_start).count();
  size_t lookup_tokens_proposed = 0;
  const int vocab_size = params.config.model.vocab_size;
  auto grammar = g.guidance_logits_processor_->Clone();
  auto mask_buffer = params.p_device->Allocate<float>(static_cast<size_t>(vocab_size));
  for (int32_t candidate : candidates) {
    auto mask_cpu = mask_buffer.CpuSpan();
    std::fill(mask_cpu.begin(), mask_cpu.end(), 0.0f);
    mask_buffer.CopyCpuToDevice();
    grammar->ProcessLogits(mask_buffer);
    const auto masked = mask_buffer.CopyDeviceToCpu();
    if (masked[static_cast<size_t>(candidate)] ==
        std::numeric_limits<float>::lowest()) {
      grammar_candidate_rejections_++;
      break;
    }

    proposal.tokens.push_back(candidate);
    lookup_tokens_proposed++;
    const auto& eos_ids = params.config.model.eos_token_id;
    if (std::find(eos_ids.begin(), eos_ids.end(), candidate) != eos_ids.end())
      break;
    CommitGuidanceProposalToken(*grammar, candidate, ff_queue);
    if (!ff_queue.empty()) {
      while (!ff_queue.empty() && static_cast<int>(proposal.tokens.size()) < K) {
        proposal.tokens.push_back(ff_queue.front());
        ff_queue.pop_front();
      }
      break;
    }
  }
  RecordLookup(candidates, lookup_tokens_proposed, lookup_ms);
  return proposal;
}

void NGramDecodingStrategy::Advance(Generator& g,
                                    const Proposal& proposal,
                                    int n_direct,
                                    int32_t final_token,
                                    int seed_length) {
  (void)proposal;
  (void)n_direct;
  (void)final_token;
  (void)seed_length;
  Sync(g);
}

void NGramDecodingStrategy::ReconcileProposer(Generator& g,
                                              int floor,
                                              std::span<const int32_t> committed,
                                              int committed_length,
                                              bool record_stats) {
  (void)g;
  (void)floor;
  (void)committed_length;
  using clock = std::chrono::steady_clock;
  using ms_f = std::chrono::duration<float, std::milli>;
  const auto sync_start = clock::now();
  lookup_.Reset(committed);
  if (record_stats)
    RecordHistorySync(committed.size(), ms_f(clock::now() - sync_start).count());
}

void NGramDecodingStrategy::FinalizeGuidanceProposer(
    Generator& g,
    int seed_length,
    int proposal_length,
    std::span<const int32_t> committed) {
  (void)seed_length;
  (void)proposal_length;
  (void)committed;
  Sync(g);
}

void NGramDecodingStrategy::ResetProposer() {
  lookup_.Reset();
}

void NGramDecodingStrategy::PopulateProposerStats(SpeculativeStats& stats) const {
  stats.ngram_lookup_hits = lookup_hits_;
  stats.ngram_lookup_misses = lookup_misses_;
  stats.ngram_lookup_tokens_proposed = lookup_tokens_proposed_;
  stats.ngram_chained_tokens_proposed = chained_tokens_proposed_;
  stats.ngram_grammar_candidate_rejections = grammar_candidate_rejections_;
  stats.ngram_history_syncs = history_syncs_;
  stats.ngram_history_tokens_synced = history_tokens_synced_;
  stats.total_ngram_history_sync_ms = total_history_sync_ms_;
  stats.total_ngram_lookup_ms = total_lookup_ms_;
}

}  // namespace Generators
