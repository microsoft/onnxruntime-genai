#include <cuda_runtime.h>
#include <assert.h>
#include "span.h"

namespace Generators {
namespace cuda {

__global__ void ExpandInputSequences(const int32_t* input_sequences, int32_t* sequences, int batch_size, int beam_size, int current_length, int max_length) {
  // The original inputs are not expanded, this expands them in place into the sequences
  for (size_t batch = 0; batch < batch_size; batch++) {
    for (size_t beam = 0; beam < beam_size; beam++) {
      for (int j = 0; j < current_length; j++) {
        sequences[(batch * beam_size + beam) * max_length + j] =
            static_cast<int32_t>(input_sequences[batch * current_length + j]);
      }
    }
  }
}

void Launch_ExpandInputSequences(std::span<const int32_t> input_sequences, std::span<int32_t> sequences, int batch_size, int beam_size, int current_length, int max_length, cudaStream_t stream) {
  ExpandInputSequences<<<1, 1, 0, stream>>>(input_sequences.data(), sequences.data(), batch_size, beam_size, current_length, max_length);
}

__global__ void AppendNextTokenToSequences(const int32_t* next_tokens, int32_t* sequences, int batch_beam_size, int current_length, int max_length) {
  // Append next token to each sequence.
  for (int i = 0; i < batch_beam_size; i++) {
    sequences[i * max_length + current_length] = next_tokens[i];
  }
}

void Launch_AppendNextTokenToSequences(std::span<const int32_t> next_tokens, std::span<int32_t> sequences, int batch_beam_size, int current_length, int max_length, cudaStream_t stream) {
  AppendNextTokenToSequences<<<1, 1, 0, stream>>>(next_tokens.data(), sequences.data(), batch_beam_size, current_length, max_length);
}

// TODO(aciddelgado): parallelize this kernel.
__global__ void AppendUserTokensToSequences(const int32_t* user_tokens, int32_t* sequences, int batch_beam_size, int num_beams, int past_length, int new_length, int max_length) {
  // Append user tokens to each sequence.
  for (int i = 0; i < batch_beam_size; i++) {
    for (int j = 0; j < new_length; j++) {
      sequences[i * max_length + past_length + j] = user_tokens[(i / num_beams) * new_length + j];
    }
  }
}

void Launch_AppendUserTokensToSequences(std::span<const int32_t> user_tokens, std::span<int32_t> sequences, int batch_beam_size, int num_beams, int past_length, int new_length, int max_length, cudaStream_t stream) {
  AppendUserTokensToSequences<<<1, 1, 0, stream>>>(user_tokens.data(), sequences.data(), batch_beam_size, num_beams, past_length, new_length, max_length);
}

// TODO(aciddelgado): parallelize this kernel.
__global__ void GetLastTokens(const int32_t* sequences, int32_t* last_tokens, int batch_beam_size, int current_length, int max_length) {
  // Get the last token of each sequence.
  for (int i = 0; i < batch_beam_size; i++) {
    last_tokens[i] = sequences[i * max_length + current_length - 1];
  }
}

void Launch_GetLastTokens(std::span<const int32_t> sequences, std::span<int32_t> last_tokens, int batch_beam_size, int current_length, int max_length, cudaStream_t stream) {
  // Get the last token of each sequence.
  GetLastTokens<<<1, 1, 0, stream>>>(sequences.data(), last_tokens.data(), batch_beam_size, current_length, max_length);
}

}  // namespace cuda
}  // namespace Generators
