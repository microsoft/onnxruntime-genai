// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// The implementation is based on huggingface transformers generation_beam_search.py

#pragma once
#if 0
#include <queue>
#include <math.h>
#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cpu/containers.h"
#include "contrib_ops/cpu/transformers/sequences.h"
#include "contrib_ops/cpu/transformers/generation_shared.h"
#endif

namespace Generators {

struct HypothesisScore {
  std::span<const int32_t> hypothesis;
  float score;
};

struct BeamHypotheses {
  // As these are constructed as an uninitialized array of memory, we need an Init method
  void Init(float length_penalty, std::span<HypothesisScore> beams);

  // Add a new hypothesis
  void Add(std::span<const int32_t> hypothesis, float sum_logprobs);

  // Return true if this beats the worst score in the hypothesis
  bool CanImprove(float best_sum_logprobs, int current_length) const;

  // Output results
  void Output(size_t top_k,                            // number of sequences to return
              size_t max_length,                     // max sequence length
              std::span<int32_t> sequences,        // buffer with pad token, shape (num_return_sequences, max_length)
              std::span<float> sequences_scores);  // buffer for sequence scores, with shape (num_return_sequences)

  std::span<HypothesisScore> beams_;  // Beam width sized array of hypotheses, sorted by highest scoring
  int beams_used_;                    // Number of elements used in beams_
  float length_penalty_;
  bool done_;
};

struct BeamSearchScorer {
  BeamSearchScorer(const SearchParams& parameters,
                   OrtAllocator& allocator);

  void Process(Sequences& sequences,
               std::span<const float> next_scores,
               std::span<const int32_t> next_tokens,
               std::span<const int32_t> next_indices);

  void Finalize(Sequences& sequences,
                size_t num_return_sequences,
                std::span<int32_t> output_sequences,
                std::span<float> output_sequence_scores);

  bool IsDone() const { return not_done_count_ == 0; }

  std::span<float> GetNextScores() { return next_beam_scores_; }
  std::span<int32_t> GetNextTokens() { return next_beam_tokens_; }
  std::span<int32_t> GetNextIndicesCPU() { return next_beam_indices_; }

 private:
  size_t batch_size_;
  size_t num_beams_;
  size_t max_length_;
  int pad_token_id_;
  int eos_token_id_;
  bool early_stopping_;
  int not_done_count_;  // When zero, every batch entry is done (starts at batch_size_)

  IAllocatorUniquePtr<float> next_beam_scores_ptr_;
  std::span<float> next_beam_scores_;

  IAllocatorUniquePtr<int32_t> next_beam_tokens_ptr_;
  std::span<int32_t> next_beam_tokens_;

  IAllocatorUniquePtr<int32_t> next_beam_indices_ptr_;
  std::span<int32_t> next_beam_indices_;

  IAllocatorUniquePtr<int32_t> hypothesis_buffer_ptr_;  // Allocated buffer to hold all hypotheses
  std::span<int32_t> hypothesis_buffer_;                // Span of the allocated buffer
  int hypothesis_buffer_used_{};                        // Offset of available buffer, or length of used buffer.

  IAllocatorUniquePtr<HypothesisScore> hypothesis_scores_ptr_;  // num_beams_ * batch_size_, divided into num_beams_ chunks per BeamHypothesis in beam_hyps_
  IAllocatorUniquePtr<BeamHypotheses> beam_hyps_ptr_;
  std::span<BeamHypotheses> beam_hyps_;  // Shape is batch_size_
};

}