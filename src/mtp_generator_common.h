// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <deque>
#include <memory>
#include <vector>

#include "models/onnxruntime_api.h"
#include "smartptrs.h"
#include "speculative_stats.h"

namespace Generators {

struct DeviceInterface;
struct GeneratorParams;
struct Model;
struct Tensor;

struct MtpGeneratorInterface {
  virtual ~MtpGeneratorInterface() = default;
  virtual void AppendTokens(cpu_span<const int32_t> input_ids) = 0;
  virtual void GenerateNextToken() = 0;
  virtual bool IsDone() const = 0;
  virtual const std::vector<int32_t>& GetSequence() const = 0;
  virtual SpeculativeStats GetSpeculativeStats() const = 0;
  virtual size_t Forwards() const = 0;
  virtual size_t Accepts() const = 0;
  virtual size_t Trials() const = 0;
};

void ValidateMtpPair(const Model& target_model, const Model& draft_model,
                     const GeneratorParams& params);
int32_t ArgmaxRow(const float* values, int vocab_size);
int32_t ArgmaxRow(const uint8_t* values, ONNXTensorElementDataType type, int vocab_size);
void CopyTensorRow(OrtValue& source, int row, Tensor& destination, DeviceInterface& device);
SpeculativeStats FinalizeSpeculativeStats(SpeculativeStats stats, size_t buffered_tokens);

// Length of the longest prefix of `drafts` the target agrees with: draft 0 must match the target's
// standing prediction, and each later draft must match the target's argmax at the preceding verify
// row. `verify_argmax` is only read when at least one draft is accepted.
int CountAcceptedDrafts(const int32_t* drafts, const int32_t* verify_argmax, int count,
                        int32_t target_prediction);

// Delivery plumbing shared by the MTP generators: a round commits several tokens at once, so they
// are buffered here and handed out one per GenerateNextToken call. Subclasses only implement the
// prompt and the round itself.
struct MtpGeneratorBase : MtpGeneratorInterface {
  void GenerateNextToken() override;
  bool IsDone() const override { return done_ && pending_tokens_.empty(); }
  const std::vector<int32_t>& GetSequence() const override { return emitted_sequence_; }
  SpeculativeStats GetSpeculativeStats() const override { return FinalizeSpeculativeStats(stats_, pending_tokens_.size()); }
  size_t Forwards() const override { return stats_.target_forward_passes; }
  size_t Accepts() const override { return stats_.draft_tokens_accepted; }
  size_t Trials() const override { return stats_.draft_tokens_evaluated; }

 protected:
  // Buffers sequence_[first_index..] for delivery, finishing at an eos token or max_length_.
  void QueueCommittedTokens(size_t first_index);
  bool IsEos(int32_t token) const;

  std::vector<int32_t> eos_token_ids_;
  int max_length_{};

  std::vector<int32_t> sequence_;          // internally committed tokens (may include lookahead)
  std::vector<int32_t> emitted_sequence_;  // prompt + tokens exposed through GenerateNextToken
  std::deque<int32_t> pending_tokens_;     // committed tokens waiting to be exposed
  SpeculativeStats stats_{};
  bool primed_{false};  // whether AppendTokens has run the prompt
  bool done_{false};

 private:
  // Only GenerateNextToken drives a round, so subclasses override this without calling it.
  virtual void RunRound() = 0;
};

std::unique_ptr<MtpGeneratorInterface> CreateMtpGenerator(
    const Model& target_model, const Model& draft_model, const GeneratorParams& params);

}  // namespace Generators
