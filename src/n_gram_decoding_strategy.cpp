// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "n_gram_decoding_strategy.h"

#include <stdexcept>

#include "generators.h"
#include "models/model.h"
#include "models/model_type.h"
#include "openvino/interface.h"
#include "qnn/interface.h"
#include "search.h"

namespace Generators {

namespace {

int ValidateAndGetNGramSize(Generator& g) {
  ValidateNGramDecoding(*g.model_, *g.search_->params_);
  return g.search_->params_->speculative.ngram_size;
}

}  // namespace

void ValidateNGramDecoding(const Model& model, const GeneratorParams& params) {
  const auto& config = model.config_->model;
  const bool uses_sliding_kv_cache =
      config.decoder.sliding_window &&
      config.decoder.sliding_window->slide_key_value_cache;
  const bool is_plain_decoder_only_text =
      ModelType::IsLLM(config.type) && !ModelType::IsLFM2(config.type) &&
      config.decoder.pipeline.empty() && config.vision.filename.empty() &&
      config.speech.filename.empty();

  NGramDecodingCapabilities capabilities;
  capabilities.batch_size = params.search.batch_size;
  capabilities.num_beams = params.search.num_beams;
  capabilities.num_return_sequences = params.search.num_return_sequences;
  capabilities.uses_guidance =
      !params.guidance_type.empty() || !params.guidance_data.empty();
  capabilities.uses_draft_model =
      ModelType::UsesDraftModelSpeculation(config.type, config.draft.filename);
  capabilities.is_plain_decoder_only_text = is_plain_decoder_only_text;
  capabilities.uses_sliding_kv_cache = uses_sliding_kv_cache;
  capabilities.uses_hybrid_state = !config.decoder.layer_types.empty();
  capabilities.uses_model_managed_state =
      IsOpenVINOStatefulModel(model) || IsQNNStatefulModel(model);
  capabilities.has_pruned_logits = is_plain_decoder_only_text && model.IsPruned();
  ValidateNGramDecodingCapabilities(capabilities);
}

NGramDecodingStrategy::NGramDecodingStrategy(Generator& g)
    : SpeculativeDecodingStrategy{*g.state_, *g.model_},
      lookup_{ValidateAndGetNGramSize(g)} {}

void NGramDecodingStrategy::Sync(Generator& g) {
  auto committed = g.search_->GetSequence(0);
  const size_t indexed_length = lookup_.HistorySize();

  if (committed.size() < indexed_length) {
    lookup_.Reset(committed.CopyDeviceToCpu());
    return;
  }
  if (committed.size() == indexed_length)
    return;

  auto suffix = committed.subspan(indexed_length, committed.size() - indexed_length);
  lookup_.Append(suffix.CopyDeviceToCpu());
}

SpeculativeDecodingStrategy::Proposal NGramDecodingStrategy::Propose(
    Generator& g, int K, int seed_length) {
  (void)seed_length;
  Sync(g);
  Proposal proposal{ProposalMode::kDeterministic};
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
  lookup_.Reset(committed);
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
  lookup_.Reset();
}

}  // namespace Generators
