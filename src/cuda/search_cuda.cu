// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include "../generators.h"
#include "cuda_common.h"
#include "interface.h"
#include "search_cuda.cuh"

namespace Generators {
namespace cuda {

__global__ void ExpandInputSequences(int32_t* input_sequences, int32_t* sequences, int batch_size, int beam_size, int sequence_length, int max_length) {
  // The original inputs are not expanded, this expands them in place into the sequences
  for (size_t batch = 0; batch < batch_size; batch++) {
    for (size_t beam = 0; beam < beam_size; beam++) {
      for (int j = 0; j < sequence_length; j++) {
        sequences[(batch * beam_size + beam) * max_length + j] = input_sequences[batch * sequence_length + j];
      }
    }
  }
}

void Launch_ExpandInputSequences(const std::span<int32_t> input_sequences, std::span<int32_t> sequences, int batch_size, int beam_size, int max_length, cudaStream_t stream) {
  const int total_elements = static_cast<int>(input_sequences.size());
  const int new_length = total_elements / batch_size;
  ExpandInputSequences<<<1, 1, 0, stream>>>(input_sequences.data(), sequences.data(), batch_size, beam_size, new_length, max_length);
}

__global__ void AppendNextTokensToSequences(const int32_t* next_tokens, int32_t* sequences, int batch_beam_size, int past_length, int new_length, int max_length) {
  // Append next tokens to each sequence.
  int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_index = global_index / new_length;
  int token_index = global_index % new_length;
  if (global_index < batch_beam_size * new_length) {
    sequences[batch_index * max_length + past_length + token_index] = next_tokens[global_index];
  }
}

void Launch_AppendNextTokensToSequences(std::span<const int32_t> next_tokens, std::span<int32_t> sequences, int batch_beam_size, int past_length, int max_length, cudaStream_t stream) {
  const int total_elements = static_cast<int>(next_tokens.size());
  const int blockSize = std::min(total_elements, 256);
  const int gridSize = (total_elements + blockSize - 1) / blockSize;
  const int new_length = total_elements / batch_beam_size;
  AppendNextTokensToSequences<<<gridSize, blockSize, 0, stream>>>(next_tokens.data(), sequences.data(), batch_beam_size, past_length, new_length, max_length);
}

__global__ void GetLastTokens(int32_t* next_tokens, const int32_t* sequences, int batch_beam_size, int sequence_length, int max_length) {
  // Get the last token of each sequence.
  int batch_beam_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_beam_index < batch_beam_size) {
    next_tokens[batch_beam_index] = sequences[batch_beam_index * max_length + sequence_length - 1];
  }
}

void Launch_GetLastTokens(int32_t* next_tokens, const int32_t* sequences, int batch_beam_size, int sequence_length, int max_length, cudaStream_t stream) {
  const int blockSize = std::min(batch_beam_size, 256);
  const int gridSize = (batch_beam_size + blockSize - 1) / blockSize;
  GetLastTokens<<<gridSize, blockSize, 0, stream>>>(next_tokens, sequences, batch_beam_size, sequence_length, max_length);
}

__global__ void ArgMax(cub::KeyValuePair<int, float>* argmaxen, int32_t* next_tokens, int batch_size) {
  int batch_index = threadIdx.x;
  next_tokens[batch_index] = argmaxen[batch_index].key;
}

struct ArgMaxDataImpl : ArgMaxData {
  cuda_unique_ptr<uint8_t> temp_storage_;
  size_t temp_storage_element_size_{};  // Size per batch, temp_storage_ is this size * batch_size

  gpu_span<cub::KeyValuePair<int, float>> argmaxen_;
  cuda_unique_ptr<cub::KeyValuePair<int, float>> argmaxen_owner_;
};

__global__ void CheckForEOSAndPad(int32_t* next_tokens, int next_tokens_count, bool* eos_meet, int eos_token_id, int pad_token_id, bool* done_cpu) {
  // Look for EOS tokens, if seen set EOS flag and replace with pad token
  for (size_t batch_id = 0; batch_id < next_tokens_count; ++batch_id) {
    if (next_tokens[batch_id] == eos_token_id || eos_meet[batch_id] == true) {
      eos_meet[batch_id] = true;
      next_tokens[batch_id] = pad_token_id;
    }
  }

  // When all batches are finished, stop earlier to avoid wasting computation.
  // TODO: Merge this with the above so we don't have to double scan. Just keep track of 'batches left'
  {
    size_t batch_id = 0;
    while (batch_id < next_tokens_count) {
      if (eos_meet[batch_id] == false) {
        break;
      }
      ++batch_id;
    }
    if (batch_id == next_tokens_count) {
      *done_cpu = true;
      return;
    }
  }
}

void Launch_CheckForEOSAndPad(int32_t* next_tokens, int next_tokens_count, bool* eos_meet, int eos_token_id, int pad_token_id, bool* done_cpu, cudaStream_t stream) {
  CheckForEOSAndPad<<<1, 1, 0, stream>>>(next_tokens, next_tokens_count, eos_meet, eos_token_id, pad_token_id, done_cpu);
}

__global__ void AddProbsKernel(float* log_probs,
                               float* cum_log_probs,
                               const int vocab_size,
                               const int total_elements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_beam_index = index / vocab_size;

  if (index < total_elements)
    log_probs[index] += cum_log_probs[batch_beam_index];
}

void LaunchAddProbsKernel(float* log_probs,
                          float* cum_log_probs,
                          const int batch_size,
                          const int num_beams,
                          const int vocab_size,
                          cudaStream_t stream) {
  int total_elements = batch_size * num_beams * vocab_size;
  constexpr int blockSize = 256;
  const int gridSize = (total_elements + blockSize - 1) / blockSize;
  AddProbsKernel<<<gridSize, blockSize, 0, stream>>>(log_probs, cum_log_probs, vocab_size, total_elements);
}

__global__ void SetScoreProcessor(float* next_token_scores, int batch_beam_size, int vocab_size, int token, float score) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= batch_beam_size)
    return;

  next_token_scores[index * vocab_size + token] = score;
}

void LaunchSetScoreProcessor(float* next_token_scores, int batch_beam_size, int vocab_size, int token, float score, cudaStream_t stream) {
  int total_elements = batch_beam_size;
  constexpr int blockSize = 256;
  const int gridSize = (total_elements + blockSize - 1) / blockSize;

  SetScoreProcessor<<<gridSize, blockSize, 0, stream>>>(next_token_scores, batch_beam_size, vocab_size, token, score);
}

__global__ void RepetitionPenaltyProcessor(const int32_t* sequences, float* next_token_scores, int max_sequence_length, int vocab_size, int total_elements, int current_sequence_length, float repetition_penalty) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total_elements)
    return;

  int batch_beam_index = index / vocab_size;
  int word_id = index % vocab_size;

  const int32_t* current_sequence = sequences + batch_beam_index * max_sequence_length;
  bool found = false;
  for (int i = 0; i < current_sequence_length; i++) {
    if (current_sequence[i] == word_id) {
      found = true;
      break;
    }
  }
  if (found) {
    float score = next_token_scores[index];
    next_token_scores[index] = score < 0 ? score * repetition_penalty : score / repetition_penalty;
  }
}

void LaunchRepetitionPenaltyProcessor(const int32_t* sequences, float* next_token_scores, int batch_size, int num_beams, int vocab_size, int max_sequence_length, int current_sequence_length, float repetition_penalty, cudaStream_t stream) {
  int total_elements = batch_size * num_beams * vocab_size;
  constexpr int blockSize = 256;
  const int gridSize = (total_elements + blockSize - 1) / blockSize;

  RepetitionPenaltyProcessor<<<gridSize, blockSize, 0, stream>>>(sequences, next_token_scores, max_sequence_length, vocab_size, total_elements, current_sequence_length, repetition_penalty);
}

}  // namespace cuda
}  // namespace Generators
