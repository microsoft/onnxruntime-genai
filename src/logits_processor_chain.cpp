// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "generators.h"
#include "logits_processor_chain.h"
#include "search.h"
#include "constrained_logits_processor.h"

#include <limits>
#include <stdexcept>

namespace Generators {

namespace {

// repetition_penalty / min_length delegate to the existing search scoring kernels so behavior matches
// the legacy path exactly (these need the sequence-token context the search engine owns).
struct RepetitionPenaltyOp : LogitsProcessorOp {
  RepetitionPenaltyOp(Search& search, float penalty) : search_{search}, penalty_{penalty} {}
  void Process(DeviceSpan<float> /*logits*/) override { search_.ApplyRepetitionPenalty(penalty_); }
  Search& search_;
  float penalty_;
};

struct MinLengthOp : LogitsProcessorOp {
  MinLengthOp(Search& search, int min_length) : search_{search}, min_length_{min_length} {}
  void Process(DeviceSpan<float> /*logits*/) override { search_.ApplyMinLength(min_length_); }
  Search& search_;
  int min_length_;
};

// logit_bias adds a fixed per-token delta to every batch row. A delta of -inf (or a large negative
// value) effectively masks a token. Applied in declared order, so a later op sees the biased logits.
struct LogitBiasOp : LogitsProcessorOp {
  LogitBiasOp(std::vector<std::pair<int32_t, float>> bias, int vocab_size, int batch_size)
      : bias_{std::move(bias)}, vocab_size_{vocab_size}, batch_size_{batch_size} {}

  void Process(DeviceSpan<float> logits) override {
    auto cpu = logits.CopyDeviceToCpu();
    for (int b = 0; b < batch_size_; ++b) {
      auto row = cpu.subspan(static_cast<size_t>(b) * vocab_size_, vocab_size_);
      for (const auto& [token, delta] : bias_) {
        if (token >= 0 && token < vocab_size_)
          row[token] += delta;
      }
    }
    logits.CopyCpuToDevice();
  }

  std::vector<std::pair<int32_t, float>> bias_;
  int vocab_size_;
  int batch_size_;
};

// grammar reuses the existing ConstrainedLogitsProcessor (llguidance) verbatim, including its
// stateful CommitTokens/Reset lifecycle owned by the Generator.
struct GrammarOp : LogitsProcessorOp {
  explicit GrammarOp(ConstrainedLogitsProcessor& guidance) : guidance_{guidance} {}
  void Process(DeviceSpan<float> logits) override { guidance_.ProcessLogits(logits); }
  void Reset() override { guidance_.Reset(); }
  ConstrainedLogitsProcessor& guidance_;
};

}  // namespace

LogitsProcessorChain::LogitsProcessorChain(const Config::Search& search,
                                           Search& search_engine,
                                           ConstrainedLogitsProcessor* guidance,
                                           int vocab_size,
                                           int batch_size)
    : search_engine_{search_engine},
      do_sample_{search.do_sample},
      top_k_{search.top_k},
      top_p_{search.top_p},
      temperature_{search.temperature} {
  for (const auto& spec : search.logits_processors) {
    if (spec.op == "repetition_penalty") {
      ops_.push_back(std::make_unique<RepetitionPenaltyOp>(search_engine_, spec.value.value_or(1.0f)));
    } else if (spec.op == "min_length") {
      ops_.push_back(std::make_unique<MinLengthOp>(search_engine_, spec.int_value.value_or(0)));
    } else if (spec.op == "logit_bias") {
      ops_.push_back(std::make_unique<LogitBiasOp>(spec.bias, vocab_size, batch_size));
    } else if (spec.op == "grammar") {
      if (!guidance)
        throw std::runtime_error(
            "logits chain: 'grammar' op requires constrained decoding (build with USE_GUIDANCE and "
            "enable guidance in the config).");
      ops_.push_back(std::make_unique<GrammarOp>(*guidance));
    } else if (spec.op == "temperature") {
      // Scalar sampler ops are realized by the terminal sampler (existing fused kernels), so they are
      // collected rather than applied as standalone transforms.
      temperature_ = spec.value.value_or(temperature_);
    } else if (spec.op == "top_k") {
      top_k_ = spec.int_value.value_or(top_k_);
      do_sample_ = true;
    } else if (spec.op == "top_p") {
      top_p_ = spec.value.value_or(top_p_);
      do_sample_ = true;
    } else if (spec.op == "sample") {
      // Terminal marker; the chain always runs the sampler after the ordered ops.
    } else if (spec.op == "combine") {
      // Contrastive / CFG (§5/§6): needs multi-session role logits not wired into the non-speculative
      // GenerateNextToken path. Parsed-but-deferred to keep the declarative core verifiable.
      throw std::runtime_error(
          "logits chain: 'combine' (contrastive/CFG) op is not yet supported on the non-speculative "
          "generation path (deferred, see PR-D decision note).");
    } else {
      throw std::runtime_error("logits chain: unknown op '" + spec.op + "'.");
    }
  }
}

void LogitsProcessorChain::Apply(DeviceSpan<float> logits) {
  for (auto& op : ops_)
    op->Process(logits);
  RunSampler();
}

void LogitsProcessorChain::RunSampler() {
  // Mirror Generator::InitializeSamplingMethod selection, but from the chain-collected scalar params.
  if (!do_sample_ || top_k_ == 1 || temperature_ == 0) {
    search_engine_.SelectTop();
    return;
  }
  if (top_p_ > 0.0f && top_p_ < 1.0f && top_k_ > 1) {
    search_engine_.SampleTopKTopP(top_k_, top_p_, temperature_);
  } else if (top_k_ > 1) {
    search_engine_.SampleTopK(top_k_, temperature_);
  } else {
    search_engine_.SampleTopP(top_p_, temperature_);
  }
}

void LogitsProcessorChain::Reset() {
  for (auto& op : ops_)
    op->Reset();
}

}  // namespace Generators
