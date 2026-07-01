// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "standard_decoding_strategy.h"

#include "generators.h"
#include "logging.h"
#include "search.h"
#include "constrained_logits_processor.h"

namespace Generators {

void StandardDecodingStrategy::Step(Generator& g) {
  if (g.search_->GetSequenceLength() == 0 && !g.computed_logits_)
    throw std::runtime_error(
        "GenerateNextToken called with no prior state. Please call AppendTokens, SetLogits, or "
        "SetInputs before calling GenerateNextToken.");

  // Phi3 models switch rope factors at a threshold token; recompute position
  // IDs and KV cache by rewinding and re-appending the sequence.
  if (g.phi3_rope_threshold_ != 0 && g.search_->GetSequenceLength() == g.phi3_rope_threshold_) {
    auto current_seq = cpu_span<int32_t>(g.GetSequence(0).CopyDeviceToCpu());
    g.RewindToLength(0);
    g.AppendTokens(current_seq);
  }

  if (!g.computed_logits_) {
    auto next_tokens = g.search_->GetNextTokens();
    if (g.last_action_ == Generator::Action::rewound)
      g.search_->AppendTokens(next_tokens);
    g.ComputeLogits(next_tokens);
  }
  if (g.guidance_logits_processor_) {
    auto logits = g.GetLogits();
    g.guidance_logits_processor_->ProcessLogits(logits);
  }
  g.computed_logits_ = false;
  auto& search = g.search_->params_->search;
  g.search_->ApplyMinLength(search.min_length);
  g.search_->ApplyRepetitionPenalty(search.repetition_penalty);

  if (g_log.enabled && g_log.generate_next_token) {
    auto& stream = Log("generate_next_token");
    stream << SGR::Fg_Green << "do_sample: " << SGR::Reset << search.do_sample << ' '
           << SGR::Fg_Green << "top_k: " << SGR::Reset << search.top_k << ' '
           << SGR::Fg_Green << "top_p: " << SGR::Reset << search.top_p << ' '
           << SGR::Fg_Green << "temperature: " << SGR::Reset << search.temperature << ' '
           << SGR::Fg_Cyan << "sequence length: " << SGR::Reset << g.search_->GetSequenceLength()
           << std::endl;
  }

  g.last_action_ = Generator::Action::generated;
  switch (g.sampling_method_) {
    case Generator::SamplingMethod::kGreedy:
      g.search_->SelectTop();
      return;
    case Generator::SamplingMethod::kTopKTopP:
      g.search_->SampleTopKTopP(search.top_k, search.top_p, search.temperature);
      return;
    case Generator::SamplingMethod::kTopK:
      g.search_->SampleTopK(search.top_k, search.temperature);
      return;
    case Generator::SamplingMethod::kTopP:
      g.search_->SampleTopP(search.top_p, search.temperature);
      return;
    default:
      throw std::runtime_error("Unknown sampling method");
  }
}

}  // namespace Generators
