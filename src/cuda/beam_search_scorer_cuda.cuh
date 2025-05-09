// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "models/onnxruntime_api.h"
#include "smartptrs.h"

namespace Generators {
namespace cuda {

struct HypothesisScore {
  const int32_t* hypothesis;
  int hypothesis_length;
  float score;
};

struct BeamHypotheses {
  HypothesisScore* beams_;  // Beam width sized array of hypotheses, sorted by highest scoring
  int beams_count_;
  int beams_used_;  // Number of elements used in beams_
  float length_penalty_;
  bool done_;

  // Add a new hypothesis
  __device__ void Add(const int32_t* hypothesis, int hypothesis_length, float sum_logprobs);

  // Return true if this beats the worst score in the hypothesis
  __device__ bool CanImprove(float best_sum_logprobs, int current_length) const;
};

struct BeamScorerState {
  int batch_size_;
  int num_beams_;
  int max_length_;
  int pad_token_id_;
  bool early_stopping_;
  int not_done_count_;  // When zero, every batch entry is done (starts at batch_size_)

  int hypothesis_buffer_used_;  // Offset of available buffer, or length of used buffer.
};

void LaunchInitializeBeamHypotheses(std::span<BeamHypotheses> beam_hyps, float length_penalty, std::span<HypothesisScore> beams, int num_beams, cudaStream_t stream);

void LaunchBeamSearchScorer_Process(BeamScorerState& state_cpu,
                                    BeamScorerState& state,
                                    std::span<const int32_t> eos_token_ids,
                                    std::span<const int32_t> sequences,
                                    int sequence_length,
                                    std::span<BeamHypotheses> beam_hyps_,
                                    std::span<float> next_beam_scores_,
                                    std::span<int32_t> next_beam_tokens_,
                                    std::span<int32_t> next_beam_indices_,
                                    std::span<int32_t> hypothesis_buffer_,
                                    std::span<const float> next_scores,
                                    std::span<const int32_t> next_tokens,
                                    std::span<const int32_t> next_indices,
                                    cudaStream_t stream);

void LaunchBeamSearchScorer_AppendNextTokenToSequences(BeamScorerState& state_cpu,
                                                       BeamScorerState& state,
                                                       std::span<const int32_t> sequences,
                                                       std::span<int32_t> next_sequences,
                                                       int sequence_length,
                                                       std::span<int32_t> next_beam_tokens,
                                                       std::span<int32_t> next_beam_indices,
                                                       cudaStream_t stream);

void LaunchBeamSearchScorer_Finalize(int batch_size,
                                     BeamScorerState& state,
                                     std::span<const int32_t> sequences,
                                     int sequence_length,
                                     std::span<BeamHypotheses> beam_hyps_,
                                     std::span<int32_t> hypothesis_buffer,
                                     std::span<const float> final_beam_scores,
                                     cudaStream_t stream);

// Since we need to index through a couple layers of GPU memory, we need to provide a way to get the pointers
void LaunchBeamSearchScorer_GetHypothesisPtr(size_t batch_id,
                                             size_t beam_id,
                                             gpu_span<BeamHypotheses> beam_hyps,
                                             int32_t** hypothesis_ptr,
                                             int* hypothesis_length,
                                             float* hypothesis_score,
                                             cudaStream_t stream);

void LaunchInitScoresKernel(float* beam_scores,
                            int batch_size,
                            int num_beams,
                            cudaStream_t stream);

void LaunchNextTokenKernel(const int64_t* next_token_indices,
                           int32_t* next_indices,
                           int32_t* next_tokens,
                           int batch_size,
                           int top_k,
                           int vocab_size,
                           cudaStream_t stream);

void LaunchUpdateGptKernel(const int32_t* old_mask_data,
                           int32_t* mask_data,
                           int32_t* next_positions,
                           int batch_beam_size,
                           int current_length,
                           cudaStream_t stream);

}  // namespace cuda
}  // namespace Generators