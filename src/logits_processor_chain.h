// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <vector>

#include "config.h"

namespace Generators {

struct Search;
struct ConstrainedLogitsProcessor;
template <typename T>
struct DeviceSpan;

// v2.1 (issue #2114 §6): ordered logit-processor interface. Each op transforms the per-token logits
// in place. Ops are applied in declared order before the terminal sampling step. This is the
// composable, data-driven generalization of the single fixed guidance->penalty->sampling sequence
// previously hard-coded in Generator::GenerateNextToken.
struct LogitsProcessorOp {
  virtual ~LogitsProcessorOp() = default;

  // Apply this op to the logits in place. `logits` is the live search score buffer (shape
  // [batch_size, vocab_size]); mutating it is what the terminal sampler subsequently reads.
  virtual void Process(DeviceSpan<float> logits) = 0;

  // Reset stateful ops (e.g. grammar) after a rewind. Stateless ops keep the default no-op.
  virtual void Reset() {}
};

// Ordered chain of logit-processor ops plus the terminal sampler settings. Built from
// Config::Search::logits_processors. When the chain is non-empty, Generator::GenerateNextToken
// delegates the logits-processing + sampling step to LogitsProcessorChain::Apply instead of the
// legacy fixed order. When the config declares no chain, the chain is never constructed and the
// legacy path runs byte-for-byte unchanged.
struct LogitsProcessorChain {
  // `search_engine` and `guidance` are owned by the Generator and must outlive the chain.
  // `guidance` may be null; a `grammar` op then throws (constrained decoding unavailable).
  LogitsProcessorChain(const Config::Search& search,
                       Search& search_engine,
                       ConstrainedLogitsProcessor* guidance,
                       int vocab_size,
                       int batch_size);

  // Apply all ordered ops to `logits` in declared order, then run the terminal sampler, which
  // selects the next token through the owning search engine.
  void Apply(DeviceSpan<float> logits);

  // Reset stateful ops (called from Generator::RewindToLength).
  void Reset();

 private:
  void RunSampler();

  Search& search_engine_;
  std::vector<std::unique_ptr<LogitsProcessorOp>> ops_;

  // Terminal sampler settings, seeded from the scalar search params and overridden by any
  // temperature/top_k/top_p ops encountered in the chain. The terminal sampler reuses the existing
  // search kernels (SelectTop / SampleTopK / SampleTopP / SampleTopKTopP) so numerics never diverge.
  bool do_sample_{};
  int top_k_{};
  float top_p_{};
  float temperature_{};
};

}  // namespace Generators
