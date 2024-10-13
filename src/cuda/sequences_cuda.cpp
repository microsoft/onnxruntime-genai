// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "interface.h"
#include "sequences_cuda.h"

namespace Generators {
namespace cuda {
void Launch_ExpandInputSequences(std::span<const int32_t> input_sequences, std::span<int32_t> sequences, int batch_size, int beam_size, int current_length, int max_length, cudaStream_t stream);
void Launch_AppendNextTokenToSequences(std::span<const int32_t> next_tokens, std::span<int32_t> sequences, int batch_beam_size, int current_length, int max_length, cudaStream_t stream);
}  // namespace cuda

Sequences_Cuda::Sequences_Cuda(std::span<const int32_t> input_sequences, int batch_size, int beam_size, int max_length, cudaStream_t stream)
    : stream_{stream},
      batch_beam_size_{batch_size * beam_size},
      max_length_{max_length},
      current_length_{static_cast<int>(input_sequences.size()) / batch_size} {
  assert(current_length_ * batch_size == input_sequences.size());  // Ensure size divided perfectly
  size_t sequences_size = batch_beam_size_ * max_length;

  auto& device = GetCudaDeviceInterface();

  sequences_ = device.Allocate<int32_t>(sequences_size, false /*cpu_accessible*/);
  if (beam_size > 1)
    sequences_next_ = device.Allocate<int32_t>(sequences_size, false /*cpu_accessible*/);

  // TODO: input_sequences will be in cuda memory in the future, for now make a temp copy

  gpu_span<int32_t> input_sequences_gpu;
  auto input_sequences_temp = CudaMallocArray<int32_t>(input_sequences.size(), &input_sequences_gpu);
  cudaMemcpyAsync(input_sequences_gpu.data(), input_sequences.data(), input_sequences.size_bytes(), cudaMemcpyHostToDevice, stream);

  cuda::Launch_ExpandInputSequences(input_sequences_gpu, sequences_->DeviceSpan(), batch_size, beam_size, current_length_, max_length, stream_);
  cudaStreamSynchronize(stream);  // Until we remove the todo above, wait for this to complete as input_sequences_gpu is on the stack
}

DeviceMemorySpan<int32_t> Sequences_Cuda::GetSequence(size_t batch_beam_index) {
  return sequences_->subspan(batch_beam_index * max_length_, current_length_);
}

int Sequences_Cuda::GetSequenceLength() const {
  return current_length_;
}

void Sequences_Cuda::AppendNextTokenToSequences(std::span<const int32_t> next_tokens) {
  if (GetLogItems().enabled && GetLogItems().append_next_tokens) {
    auto& stream = Log("append_next_tokens");
    DumpCudaSpan(stream, next_tokens);
    stream << std::endl;
  }

  cuda::Launch_AppendNextTokenToSequences(next_tokens, sequences_->DeviceSpan(), batch_beam_size_, current_length_, max_length_, stream_);
  ++current_length_;
}

void Sequences_Cuda::AfterDeviceAppendedNextToken() {
  ++current_length_;

  // Rotate buffer for next round.
  std::swap(sequences_, sequences_next_);
}

}  // namespace Generators