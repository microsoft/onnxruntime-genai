// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "n_gram_decoding_strategy.h"

#include <stdexcept>

#include "generators.h"
#include "models/model.h"
#include "models/model_type.h"
#include "search.h"

namespace Generators {

namespace {

int ValidateAndGetNGramSize(Generator& g) {
  const auto& params = *g.search_->params_;
  const auto& model = g.model_->config_->model;
  const int ngram_size = params.speculative.ngram_size;

  if (params.search.batch_size != 1)
    throw std::runtime_error("N-gram decoding requires batch_size=1 in this release.");
  if (params.search.num_beams != 1)
    throw std::runtime_error("N-gram decoding does not support beam search in this release.");
  if (!g.IsGreedySampling())
    throw std::runtime_error("N-gram decoding supports greedy decoding only in this release.");
  if (g.guidance_logits_processor_)
    throw std::runtime_error("N-gram decoding does not support guidance in this release.");
  if (!ModelType::IsLLM(model.type) || ModelType::IsLFM2(model.type) ||
      !model.decoder.pipeline.empty() || !model.vision.filename.empty() ||
      !model.speech.filename.empty())
    throw std::runtime_error(
        "N-gram decoding requires a plain decoder-only text model in this release.");
  if ((model.decoder.sliding_window && model.decoder.sliding_window->slide_key_value_cache) ||
      !model.decoder.layer_types.empty())
    throw std::runtime_error(
        "N-gram decoding requires a rewindable KV cache; sliding-window and hybrid state models "
        "are not supported in this release.");
  if (g.model_->IsPruned())
    throw std::runtime_error(
        "N-gram decoding requires target logits for every verified token; pruned last-token-only "
        "logits are not supported in this release.");

  return ngram_size;
}

}  // namespace

NGramDecodingStrategy::NGramDecodingStrategy(Generator& g)
    : SpeculativeDecodingStrategy{*g.state_, *g.model_},
      lookup_{ValidateAndGetNGramSize(g)} {}

void NGramDecodingStrategy::Sync(Generator& g) {
  auto committed = g.search_->GetSequence(0).CopyDeviceToCpu();
  lookup_.Sync(committed);
}

SpeculativeDecodingStrategy::Proposal NGramDecodingStrategy::Propose(
    Generator& g, int K, int seed_length) {
  (void)seed_length;
  Sync(g);
  Proposal proposal;
  proposal.tokens = lookup_.Propose(static_cast<size_t>(K));
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
  (void)record_stats;
  lookup_.Sync(committed);
}

void NGramDecodingStrategy::FinalizeGuidanceProposer(
    Generator& g,
    int seed_length,
    int proposal_length,
    std::span<const int32_t> committed) {
  (void)g;
  (void)seed_length;
  (void)proposal_length;
  (void)committed;
  throw std::runtime_error("N-gram decoding does not support guidance in this release.");
}

void NGramDecodingStrategy::ResetProposer() {
  lookup_.Clear();
}

}  // namespace Generators
