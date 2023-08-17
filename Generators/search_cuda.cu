#include <cuda_runtime.h>

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

}
