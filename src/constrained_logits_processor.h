// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "smartptrs.h"

#if USE_GUIDANCE
#include <llguidance.h>
#endif

namespace Generators {

struct Model;
struct Tokenizer;

namespace test {
struct GuidanceProcessorTestAccess;
}

struct ConstrainedLogitsProcessor {
  ConstrainedLogitsProcessor() = default;
  virtual ~ConstrainedLogitsProcessor() = default;

  // Commits the selected tokens to the constrained system and also trigger mask recomputation
  // The input is the current token in the batch and internally verifies that it is valid in the current
  // context and also updates the internal state of the constraint system
  virtual void CommitTokens(std::span<int32_t> tokens) = 0;

  // ProcessLogits applies token-level masking to the logits
  // Based on the masks which are derived from constraints, it sets the logits to -inf for invalid tokens
  virtual void ProcessLogits(DeviceSpan<float> logits) = 0;

  // Returns a host-resident mask row when the processor supports scheduler-owned batched
  // application. An empty span asks the scheduler to use ProcessLogits() instead. The returned
  // span is invalidated by any subsequent call on the same processor.
  virtual std::span<const uint32_t> GetReadyMask() { return {}; }

  // Reset is used to reset the constraints of the logits processor and then recompute the mask, used after rewinding
  virtual void Reset() = 0;

  // Return a clone of the ff_tokens for the given index
  virtual std::vector<int32_t> GetFFTokens(size_t index) = 0;

  // Clone as an independent grammar cursor at the current state. Speculative decoding uses this to
  // mask a draft's proposals without disturbing the verify cursor.
  virtual std::unique_ptr<ConstrainedLogitsProcessor> Clone() const = 0;
};

#if USE_GUIDANCE
struct GuidanceLogitsProcessor : public ConstrainedLogitsProcessor {
  // llguidance need to use tokenizer.json to add special tokens
  static constexpr const char* kDefaultVocabFile = "tokenizer.json";

  // tokenizer need to tokenize token with special prefix
  static constexpr const char* kTokenizePrefixStr = "\x02";

  GuidanceLogitsProcessor(const State& state);
  GuidanceLogitsProcessor(const Model& model, const GeneratorParams& params);

  void ProcessLogits(DeviceSpan<float> logits) override;
  std::span<const uint32_t> GetReadyMask() override;
  void CommitTokens(std::span<int32_t> tokens) override;
  void Reset() override;
  std::vector<int32_t> GetFFTokens(size_t index) override;
  std::unique_ptr<ConstrainedLogitsProcessor> Clone() const override;

  // tokenize_partial is used to tokenize the input tokens with special prefix, this will get stable
  // token ids.
  static std::vector<int32_t> tokenize_partial(const Tokenizer* tokenizer, const size_t prefix_len,
                                               const uint8_t* bytes, size_t bytes_len);

 private:
  friend void ScheduleGuidanceMaskComputation(
      std::span<ConstrainedLogitsProcessor* const> processors);
  friend struct test::GuidanceProcessorTestAccess;

  // Empty processor for Clone() to populate; skips the heavy tokenizer/constraint construction.
  GuidanceLogitsProcessor() = default;

  // Acquire immutable tokenizer and compiled initial-grammar assets from the model-local cache.
  void InitializeGuidanceAssets(const Model& model);

  // Clone request-local mutable cursors from the cached initial grammar.
  void InitializeLlgConstraints();

  struct LlgConstraintDeleter {
    void operator()(LlgConstraint* lc) const {
      llg_free_constraint(lc);
    }
  };

  // Compute the mask synchronously and store in masks_
  void ComputeMask();
  void EnsureMaskScheduled();
  void EnsureMaskReady();
  static std::vector<uint32_t> ComputeMasks(
      const GeneratorParams& params, uint32_t eos_token,
      const std::vector<std::shared_ptr<LlgConstraint>>& constraints);

  std::shared_ptr<const GeneratorParams> params_;
  uint32_t eos_token_;
  size_t mask_words_per_row_{};
  std::vector<uint32_t> masks_;
  std::shared_future<std::vector<uint32_t>> pending_masks_;
  bool mask_dirty_{};
  DeviceSpan<uint32_t> device_masks_;
  // Keep constraints after the grammar asset: reverse member destruction releases every cursor
  // before the asset releases its llguidance tokenizer.
  std::shared_ptr<const struct GuidanceGrammarAsset> grammar_asset_;
  std::vector<std::shared_ptr<LlgConstraint>> llg_constraints_;

  std::vector<std::vector<int32_t>> ff_tokens_batch_;
};
#endif

// Schedules one llguidance parallel mask job for every dirty processor in the span. Processors
// that do not use llguidance are ignored.
void ScheduleGuidanceMaskComputation(
    std::span<ConstrainedLogitsProcessor* const> processors);

struct GuidanceCacheStats {
  uint64_t tokenizer_initializations{};
  uint64_t grammar_hits{};
  uint64_t grammar_misses{};
  uint64_t grammar_waits{};
  uint64_t grammar_compile_microseconds{};
  uint64_t grammar_evictions{};
  uint64_t cached_grammars{};
  uint64_t cached_key_bytes{};
};

GuidanceCacheStats GetGuidanceCacheStats(const Model& model);

// Validates a requested guidance grammar before any processor is constructed for it. Shared by
// every CreateGuidanceLogitsProcessor overload below so the generic Generator and the Engine's
// Request reject a malformed or unsupported guidance request identically, regardless of whether
// this build was compiled with use_guidance=true.
//   - neither `guidance_type` nor `guidance_data` set: returns false (no guidance requested).
//   - exactly one set: throws (the pair is malformed).
//   - both set but `guidance_type` is not a supported grammar kind: throws.
//   - both set and supported: returns true. The caller still needs USE_GUIDANCE to build a
//     processor; CreateGuidanceLogitsProcessor throws its own error when that support is missing.
bool ValidateGuidanceRequest(std::string_view guidance_type, std::string_view guidance_data);

std::unique_ptr<ConstrainedLogitsProcessor> CreateGuidanceLogitsProcessor(const State& state);
std::unique_ptr<ConstrainedLogitsProcessor> CreateGuidanceLogitsProcessor(
    const Model& model, std::shared_ptr<const GeneratorParams> params);

// Engine-facing overload. A turn's guidance request is validated for shape and build support
// first -- neither of which needs a model -- and only then requires the model the params carry.
std::unique_ptr<ConstrainedLogitsProcessor> CreateGuidanceLogitsProcessor(
    std::shared_ptr<const GeneratorParams> params);

}  // namespace Generators
