// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <future>

#if USE_GUIDANCE
#include <llguidance.h>
#endif

namespace Generators {

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
  // Reset is used to reset the masks and the constrains of the logits processor and then recompute the mask, used after rewinding
  virtual void Reset() = 0;
  // ResetWithoutCompute is used to reset the masks and constraints for logits processor without computing the mask for chat
  virtual void ResetWithoutCompute() = 0;
};

#if USE_GUIDANCE
struct GuidanceLogitsProcessor : public ConstrainedLogitsProcessor {
  // llguidance need to use tokenizer.json to add special tokens
  static constexpr const char* kDefaultVocabFile = "tokenizer.json";
  // tokenizer need to tokenize token with special prefix
  static constexpr const char* kTokenizePrefixStr = "\x02";

  GuidanceLogitsProcessor(const State& state);
  void ProcessLogits(DeviceSpan<float> logits) override;
  void CommitTokens(std::span<int32_t> tokens) override;
  void Reset() override;
  void ResetWithoutCompute() override;
  // GetMask is used to get the logits mask
  std::vector<std::vector<uint32_t>> GetMask();
  // tokenize_partial is used to tokenize the input tokens with special prefix, this will get stable
  // token ids.
  static std::vector<int32_t> tokenize_partial(const Tokenizer* tokenizer, const size_t prefix_len,
                                               const uint8_t* bytes, size_t bytes_len);

 private:
  std::vector<std::vector<uint32_t>> ComputeMask();
  struct LlgConstraintDeleter {
    void operator()(LlgConstraint* lc) const {
      llg_free_constraint(lc);
    }
  };

  struct LlgTokenizerDeleter {
    void operator()(LlgTokenizer* lt) const {
      llg_free_tokenizer(lt);
    }
  };

  std::shared_ptr<const GeneratorParams> params_;
  uint32_t eos_token_;
  std::vector<std::vector<uint32_t>> masks_;
  std::vector<std::unique_ptr<LlgConstraint, LlgConstraintDeleter>> llg_constraints_;
  std::unique_ptr<LlgTokenizer, LlgTokenizerDeleter> llg_tokenizer_;
  std::shared_ptr<Tokenizer> tokenizer_;

  std::future<std::vector<std::vector<uint32_t>>> mask_future_;
  std::vector<std::vector<uint32_t>> logits_masks_;

  struct TokenizeData {
    Tokenizer* tokenizer;
    size_t prefix_len;
  };
  TokenizeData tokenize_data_;
};
#endif

std::unique_ptr<ConstrainedLogitsProcessor> CreateGuidanceLogitsProcessor(const State& state);

}  // namespace Generators
