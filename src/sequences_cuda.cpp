// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "sequences_cuda.h"

namespace Generators {
namespace cuda {
void Launch_ExpandInputSequences(std::span<const int32_t> input_sequences, std::span<int32_t> sequences, int batch_size, int beam_size, int current_length, int max_length, cudaStream_t stream);
void Launch_AppendNextTokenToSequences(std::span<const int32_t> next_tokens, std::span<int32_t> sequences, int batch_beam_size, int current_length, int max_length, cudaStream_t stream);
void Launch_AppendUserTokensToSequences(std::span<const int32_t> next_tokens, std::span<int32_t> sequences, int batch_beam_size, int num_beams, int past_length, int new_length, int max_length, cudaStream_t stream);
void Launch_GetLastTokens(std::span<const int32_t> sequences, std::span<int32_t> last_tokens, int batch_beam_size, int current_length, int max_length, cudaStream_t stream);
}  // namespace cuda

Sequences_Cuda::Sequences_Cuda(int batch_size, int beam_size, int max_length, cudaStream_t stream)
    : stream_{stream},
      batch_beam_size_{batch_size * beam_size},
      max_length_{max_length},
      current_length_{0} {
  size_t sequences_size = batch_beam_size_ * max_length;

  if (beam_size == 1) {
    sequences_buffer_ = CudaMallocArray<int32_t>(sequences_size);
    sequences_ = gpu_span<int32_t>(sequences_buffer_.get(), sequences_size);
  } else {
    sequences_buffer_ = CudaMallocArray<int32_t>(2 * sequences_size);
    sequences_ = gpu_span<int32_t>(sequences_buffer_.get(), sequences_size);
    sequences_next_ = gpu_span<int32_t>(sequences_buffer_.get() + sequences_size, sequences_size);
  }

  // TODO: input_sequences will be in cuda memory in the future, for now make a temp copy

  gpu_span<int32_t> input_sequences_gpu;

  cudaStreamSynchronize(stream);  // Until we remove the todo above, wait for this to complete as input_sequences_gpu is on the stack
}

RoamingArray<int32_t> Sequences_Cuda::GetSequence(size_t batch_beam_index) {
  auto span = sequences_.subspan(batch_beam_index * max_length_, current_length_);
  return gpu_span<int32_t>{span.data(), span.size()};
}

int Sequences_Cuda::GetSequenceLength() const {
  return current_length_;
}

void Sequences_Cuda::AppendNextTokenToSequences(std::span<const int32_t> next_tokens) {
  if (g_log.enabled && g_log.append_next_tokens) {
    auto& stream = Log("append_next_tokens");
    DumpCudaSpan(stream, next_tokens);
    stream << std::endl;
  }

  cuda::Launch_AppendNextTokenToSequences(next_tokens, sequences_, batch_beam_size_, current_length_, max_length_, stream_);
  ++current_length_;
}

void Sequences_Cuda::AppendUserTokensToSequences(gpu_span<int32_t> user_tokens, int num_beams) {
  int new_length = static_cast<int>(user_tokens.size()) * num_beams / batch_beam_size_;
  int past_length = current_length_;
  cuda::Launch_AppendUserTokensToSequences(user_tokens, sequences_, batch_beam_size_, num_beams, past_length, new_length, max_length_, stream_);
  current_length_ += new_length;
}

void Sequences_Cuda::RewindTo(size_t index) {
  current_length_ = static_cast<int>(index);
  assert(current_length_ >= 0);
}

void Sequences_Cuda::GetLastTokens(gpu_span<int32_t>& last_tokens) {
  // TODO(aciddelgado): throw error when no last tokens
  cuda::Launch_GetLastTokens(sequences_, last_tokens, batch_beam_size_, current_length_, max_length_, stream_);
}

void Sequences_Cuda::AfterDeviceAppendedNextToken() {
  ++current_length_;

  // Rotate buffer for next round.
  std::swap(sequences_, sequences_next_);
}

}  // namespace Generators