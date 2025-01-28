// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include <assert.h>
#include <algorithm>
#include "span.h"
#include "beam_search_scorer_cuda.cuh"

namespace Generators {
namespace cuda {

__global__ void InitializeBeamHypotheses(BeamHypotheses* beam_hyps, int beam_hyps_count, float length_penalty, HypothesisScore* beams, int num_beams) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= beam_hyps_count)
    return;

  BeamHypotheses& beam_hyp = beam_hyps[index];
  beam_hyp.beams_ = beams + index * num_beams;
  beam_hyp.beams_count_ = num_beams;
  beam_hyp.beams_used_ = 0;
  beam_hyp.length_penalty_ = length_penalty;
  beam_hyp.done_ = false;
}

// For counts that are typically far less than 256, this will round up the count to the next multiple of 32
// If this winds up being >256 then it uses a block size of 256 and calculates the appropriate grid_size
struct GridBlock32 {
  GridBlock32(int count) {
    block_size_ = (count + 31) & ~31;  // Round up to nearest multiple of 32
    if (block_size_ > 256) {
      grid_size_ = (block_size_ + 255) / 256;
      block_size_ = 256;
    }
  }

  int grid_size_{1};
  int block_size_;
};

void LaunchInitializeBeamHypotheses(std::span<BeamHypotheses> beam_hyps,
                                    float length_penalty,
                                    std::span<HypothesisScore> beams,
                                    int num_beams,
                                    cudaStream_t stream) {
  GridBlock32 gb32{static_cast<int>(beam_hyps.size())};
  InitializeBeamHypotheses<<<gb32.grid_size_, gb32.block_size_, 0, stream>>>(beam_hyps.data(),
                                                                             static_cast<int>(beam_hyps.size()),
                                                                             length_penalty,
                                                                             beams.data(),
                                                                             num_beams);
}

__device__ void BeamHypotheses::Add(const int32_t* hypothesis, int hypothesis_length, float sum_logprobs) {
  float score = sum_logprobs / pow(static_cast<float>(hypothesis_length), length_penalty_);

  size_t index = beams_used_;
  // If the array is full, don't add unless it's better than the worst element
  if (index == beams_count_) {
    if (score <= beams_[--index].score)
      return;
  } else
    beams_used_++;

  // Rotate existing elements over while the new element scores higher
  for (; index > 0 && score > beams_[index - 1].score; index--)
    beams_[index] = beams_[index - 1];

  beams_[index] = HypothesisScore{hypothesis, hypothesis_length, score};
}

__device__ bool BeamHypotheses::CanImprove(float best_sum_logprobs, int current_length) const {
  float current_score = best_sum_logprobs / pow(static_cast<float>(current_length), length_penalty_);
  return beams_[beams_count_ - 1].score < current_score;
}

__global__ void BeamSearchScorer_Process(BeamScorerState& state_cpu,
                                         BeamScorerState& state,
                                         const int32_t* sequences_buffer,
                                         int sequence_length,
                                         BeamHypotheses* beam_hyps_,
                                         float* next_beam_scores_,
                                         int32_t* next_beam_tokens_,
                                         int32_t* next_beam_indices_,
                                         int32_t* hypothesis_buffer_,
                                         const float* next_scores,
                                         const int32_t* next_tokens,
                                         const int32_t* next_indices) {
  // Sequences shape is (batch_size * num_beams, total_sequence_length)
  // It contains word ID of whole sequence generated so far.
  // It is different from subgraph input_ids, which only need one word when past state is not empty.

  int batch = threadIdx.x;
  int batch_start = batch * state.num_beams_;

  cuda::BeamHypotheses& beam_hyp = beam_hyps_[batch];
  if (!beam_hyp.done_) {
    // Next tokens for this sentence.
    size_t beam_idx = 0;
    size_t top_k = 2 * state.num_beams_;
    for (size_t j = 0; j < top_k; j++) {
      int32_t next_token = next_tokens[batch * top_k + j];
      float next_score = next_scores[batch * top_k + j];
      int32_t next_index = next_indices[batch * top_k + j];

      int batch_beam_idx = batch_start + next_index;
      // Add to generated hypotheses if end of sentence.
      if ((state.eos_token_id_ >= 0) && (next_token == state.eos_token_id_)) {
        bool is_beam_token_worse_than_top_num_beams = (j >= state.num_beams_);
        if (is_beam_token_worse_than_top_num_beams) {
          continue;
        }

        // Clone the sequence and append to buffer.
        // TODO(aciddelgado): why do we need to clone the sequence here? A: Because we overwrite the sequences with other beams
        const int32_t* src = sequences_buffer + batch_beam_idx * state.max_length_;
        auto clone = hypothesis_buffer_ + atomicAdd(&state.hypothesis_buffer_used_, sequence_length);

        for (unsigned i = 0; i < sequence_length; i++)
          clone[i] = src[i];
        beam_hyp.Add(clone, sequence_length, next_score);
      } else {
        // Add next predicted token since it is not eos_token.
        next_beam_scores_[batch_start + beam_idx] = next_score;
        next_beam_tokens_[batch_start + beam_idx] = next_token;
        next_beam_indices_[batch_start + beam_idx] = batch_beam_idx;
        ++beam_idx;
      }

      // Once the beam for next step is full, don't add more tokens to it.
      if (beam_idx == state.num_beams_)
        break;
    }

    //  Check if we are done so that we can save a pad step if all(done)
    if (beam_hyp.beams_used_ == state.num_beams_) {
      if (state.early_stopping_ || !beam_hyp.CanImprove(*std::max_element(next_scores + batch_start, next_scores + batch_start + top_k), sequence_length)) {
        beam_hyp.done_ = true;
        if (atomicAdd(&state.not_done_count_, -1) == 1)
          state_cpu.not_done_count_ = 0;  // Update the CPU side
      }
    }
  } else {
    // Pad the batch.
    for (size_t beam_idx = 0; beam_idx < state.num_beams_; beam_idx++) {
      next_beam_scores_[batch_start + beam_idx] = 0.0f;
      next_beam_tokens_[batch_start + beam_idx] = state.pad_token_id_;
      next_beam_indices_[batch_start + beam_idx] = 0;
    }
  }
}

void LaunchBeamSearchScorer_Process(BeamScorerState& state_cpu,
                                    BeamScorerState& state,
                                    std::span<const int32_t> sequences,
                                    int sequence_length,
                                    std::span<BeamHypotheses> beam_hyps,
                                    std::span<float> next_beam_scores,
                                    std::span<int32_t> next_beam_tokens,
                                    std::span<int32_t> next_beam_indices,
                                    std::span<int32_t> hypothesis_buffer,
                                    std::span<const float> next_scores,
                                    std::span<const int32_t> next_tokens,
                                    std::span<const int32_t> next_indices,
                                    cudaStream_t stream) {
  BeamSearchScorer_Process<<<1, state_cpu.batch_size_, 0, stream>>>(state_cpu,
                                                                    state,
                                                                    sequences.data(),
                                                                    sequence_length,
                                                                    beam_hyps.data(),
                                                                    next_beam_scores.data(),
                                                                    next_beam_tokens.data(),
                                                                    next_beam_indices.data(),
                                                                    hypothesis_buffer.data(),
                                                                    next_scores.data(),
                                                                    next_tokens.data(),
                                                                    next_indices.data());
}

__global__ void BeamSearchScorer_AppendNextTokenToSequences1(BeamScorerState& state,
                                                             int batch_beam_size,
                                                             const int32_t* sequences_buffer,
                                                             int32_t* next_sequences,
                                                             int sequence_length,
                                                             int32_t* next_beam_indices_) {
  int beam_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (beam_idx >= batch_beam_size)
    return;
  int sequence_index = threadIdx.y + blockIdx.y * blockDim.y;
  if (sequence_index >= sequence_length)
    return;

  int beam_index = next_beam_indices_[beam_idx];
  next_sequences[beam_idx * state.max_length_ + sequence_index] = sequences_buffer[beam_index * state.max_length_ + sequence_index];
}

__global__ void BeamSearchScorer_AppendNextTokenToSequences2(BeamScorerState& state,
                                                             int32_t* next_sequences,
                                                             int sequence_length,
                                                             const int32_t* next_beam_tokens_) {
  int beam_idx = threadIdx.x;
  next_sequences[beam_idx * state.max_length_ + sequence_length] = next_beam_tokens_[beam_idx];
}

void LaunchBeamSearchScorer_AppendNextTokenToSequences(BeamScorerState& state_cpu,
                                                       BeamScorerState& state,
                                                       std::span<const int32_t> sequences,
                                                       std::span<int32_t> next_sequences,
                                                       int sequence_length,
                                                       std::span<int32_t> next_beam_tokens,
                                                       std::span<int32_t> next_beam_indices,
                                                       cudaStream_t stream) {
  const int max_threads = 512;
  int batch_beam_size = state_cpu.batch_size_ * state_cpu.num_beams_;
  dim3 block_size;
  dim3 grid_size;
  if (batch_beam_size * sequence_length <= max_threads) {  // Can fit into a single thread block
    block_size.x = batch_beam_size;
    block_size.y = sequence_length;
  } else {
    if (sequence_length <= max_threads) {  // Sequence length fits into thread block, but batch_beam_size does not, so chunk it
      block_size.x = max_threads / sequence_length;
      block_size.y = sequence_length;

      grid_size.x = (batch_beam_size + block_size.x - 1) / block_size.x;
    } else {  // Exceed max_threads in every dimension, so divide into max_thread chunks
      block_size.x = 1;
      block_size.y = max_threads;

      grid_size.x = batch_beam_size;
      grid_size.y = (sequence_length + block_size.y - 1) / block_size.y;
    }
  }
  BeamSearchScorer_AppendNextTokenToSequences1<<<grid_size, block_size, 0, stream>>>(state,
                                                                                     batch_beam_size,
                                                                                     sequences.data(),
                                                                                     next_sequences.data(),
                                                                                     sequence_length,
                                                                                     next_beam_indices.data());

  BeamSearchScorer_AppendNextTokenToSequences2<<<1, batch_beam_size, 0, stream>>>(state,
                                                                                  next_sequences.data(),
                                                                                  sequence_length,
                                                                                  next_beam_tokens.data());
}

__global__ void BeamSearchScorer_Finalize(BeamScorerState& state,
                                          const int32_t* sequences_buffer,
                                          int sequence_length,
                                          BeamHypotheses* beam_hyps_,
                                          int32_t* hypothesis_buffer_,
                                          const float* final_beam_scores) {
  int batch_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_index >= state.batch_size_)
    return;

  // Finalize all open beam hypotheses and add to generated hypotheses.
  cuda::BeamHypotheses& beam_hyp = beam_hyps_[batch_index];
  if (!beam_hyp.done_) {
    for (size_t beam_index = 0; beam_index < state.num_beams_; beam_index++) {
      size_t batch_beam_index = batch_index * state.num_beams_ + beam_index;
      float final_score = final_beam_scores[batch_beam_index];

      // Clone the sequence and append to buffer.
      const int32_t* src = sequences_buffer + batch_beam_index * state.max_length_;
      auto clone = hypothesis_buffer_ + atomicAdd(&state.hypothesis_buffer_used_, sequence_length);

      for (unsigned i = 0; i < sequence_length; i++)
        clone[i] = src[i];
      beam_hyp.Add(clone, sequence_length, final_score);
    }
  }
}

void LaunchBeamSearchScorer_Finalize(int batch_size,
                                     BeamScorerState& state,
                                     std::span<const int32_t> sequences,
                                     int sequence_length,
                                     std::span<BeamHypotheses> beam_hyps,
                                     std::span<int32_t> hypothesis_buffer,
                                     std::span<const float> final_beam_scores,
                                     cudaStream_t stream) {
  BeamSearchScorer_Finalize<<<1, batch_size, 0, stream>>>(state,
                                                          sequences.data(),
                                                          sequence_length,
                                                          beam_hyps.data(),
                                                          hypothesis_buffer.data(),
                                                          final_beam_scores.data());
}

__global__ void BeamSearchScorer_GetHypothesisPtr(size_t batch_id,
                                                  size_t beam_id,
                                                  BeamHypotheses* beam_hyps_data,
                                                  int32_t** hypothesis_ptr,
                                                  int* hypothesis_length,
                                                  float* hypothesis_score) {
  auto& beam_hyp = beam_hyps_data[batch_id];
  auto& item = beam_hyp.beams_[beam_id];
  hypothesis_ptr[0] = const_cast<int32_t*>(item.hypothesis);
  hypothesis_length[0] = item.hypothesis_length;
  hypothesis_score[0] = item.score;
}

void LaunchBeamSearchScorer_GetHypothesisPtr(size_t batch_id,
                                             size_t beam_id,
                                             gpu_span<BeamHypotheses> beam_hyps,
                                             int32_t** hypothesis_ptr,
                                             int* hypothesis_length,
                                             float* hypothesis_score,
                                             cudaStream_t stream) {
  BeamSearchScorer_GetHypothesisPtr<<<1, 1, 0, stream>>>(batch_id,
                                                         beam_id,
                                                         beam_hyps.data(),
                                                         hypothesis_ptr,
                                                         hypothesis_length,
                                                         hypothesis_score);
}

__global__ void InitScoresKernel(float* beam_scores,
                                 int num_beams,
                                 int total_elements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total_elements) {
    int beam_index = index % num_beams;
    beam_scores[index] = beam_index > 0 ? static_cast<float>(-1e9) : 0.0f;
  }
}

void LaunchInitScoresKernel(
    float* beam_scores,
    int batch_size,
    int num_beams,
    cudaStream_t stream) {
  int total_elements = batch_size * num_beams;
  constexpr int blockSize = 256;
  const int gridSize = (total_elements + blockSize - 1) / blockSize;
  InitScoresKernel<<<gridSize, blockSize, 0, stream>>>(beam_scores, num_beams, total_elements);
}

}  // namespace cuda
}  // namespace Generators
