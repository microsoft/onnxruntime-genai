#include <cuda_runtime.h>
#include <algorithm>

using ScoreType = float; // TODO: Move to header includable by cuda

namespace Generators {

__global__ void SoftMax(int32_t* next_tokens, const ScoreType* next_token_scores, int batch_size, int vocab_size) {
  // next_tokens = torch.argmax(scores, dim=-1)
  for (size_t i = 0; i < batch_size; i++) {
    int32_t best_token = 0;
    ScoreType best_score = next_token_scores[0];
    for (int32_t token = 1; token < vocab_size; token++) {
      if (next_token_scores[token] > best_score) {
        best_score = next_token_scores[token];
        best_token = token;
      }
    }
    next_tokens[i] = best_token;
    next_token_scores += vocab_size;
  }
}

void Launch_SoftMax(int32_t* next_tokens, const ScoreType* next_token_scores, int batch_size, int vocab_size, cudaStream_t stream) {
  SoftMax<<<1, 1, 0, stream>>>(next_tokens, next_token_scores, batch_size, vocab_size);
}

__global__ void log_softmax(ScoreType* values, unsigned count) {
  float max = *std::max_element(values, values+count);
//  std::vector<float> scaled(values.begin(), values.end());
  float sum=0.0f;
  for (unsigned i=0;i<count;i++)
    sum += std::exp(values[i]-max);

  float log_max = std::log(sum);
  std::transform(values, values+count, values, [max, log_max](float v) { return v - max - log_max; });
}

void Launch_log_softmax(ScoreType* values, unsigned count, cudaStream_t stream) {
  log_softmax<<<1, 1, 0, stream>>>(values, count);
}

__global__ void CheckForEOS(int32_t* next_tokens, int next_tokens_count, bool* eos_meet, int eos_token_id, int pad_token_id, bool *done_cpu) {
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

void Launch_CheckForEOS(int32_t* next_tokens, int next_tokens_count, bool* eos_meet, int eos_token_id, int pad_token_id, bool *done_cpu, cudaStream_t stream) {
  CheckForEOS<<<1, 1, 0, stream>>>(next_tokens, next_tokens_count, eos_meet, eos_token_id, pad_token_id, done_cpu);
}

__global__ void AddProbsKernel(ScoreType* log_probs,
                               ScoreType* cum_log_probs,
                               const int vocab_size,
                               const int total_elements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_beam_index = index / vocab_size;

  if (index < total_elements)
    log_probs[index] += cum_log_probs[batch_beam_index];
}

void LaunchAddProbsKernel(ScoreType* log_probs,
                          ScoreType* cum_log_probs,
                          const int batch_size,
                          const int num_beams,
                          const int vocab_size,
                          cudaStream_t stream) {
  int total_elements = batch_size * num_beams * vocab_size;
  constexpr int blockSize = 256;
  const int gridSize = (total_elements + blockSize - 1) / blockSize;
  AddProbsKernel<<<gridSize, blockSize, 0, stream>>>(log_probs, cum_log_probs, vocab_size, total_elements);
}

__global__ void RepetitionPenaltyProcessor(const int32_t* sequences, ScoreType* next_token_scores, int max_sequence_length, int vocab_size, int total_elements, int current_sequence_length, ScoreType repetition_penalty) {
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
    ScoreType score = next_token_scores[index];
    next_token_scores[index] = score < 0 ? score * repetition_penalty : score / repetition_penalty;
  }
}

void LaunchRepetitionPenaltyProcessor(const int32_t* sequences, ScoreType* next_token_scores, int batch_size, int num_beams, int vocab_size, int max_sequence_length, int current_sequence_length, ScoreType repetition_penalty, cudaStream_t stream) {

  int total_elements = batch_size * num_beams * vocab_size;
  constexpr int blockSize = 256;
  const int gridSize = (total_elements + blockSize - 1) / blockSize;

  RepetitionPenaltyProcessor<<<gridSize, blockSize, 0, stream>>>(sequences, next_token_scores, max_sequence_length, vocab_size, total_elements, current_sequence_length, repetition_penalty);
}


}
