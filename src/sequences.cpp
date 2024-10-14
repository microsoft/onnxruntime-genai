// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "sequences.h"

namespace Generators {

Sequences::Sequences(std::span<const int32_t> input_sequences, int batch_size, int beam_size, int max_length)
    : batch_beam_size_{batch_size * beam_size},
      max_length_{max_length},
      current_length_{static_cast<int>(input_sequences.size()) / batch_size} {
  assert(current_length_ * batch_size == input_sequences.size());  // Ensure size divided perfectly
  const size_t sequences_size = static_cast<size_t>(batch_beam_size_) * max_length;

  auto& device = GetCpuDeviceInterface();

  sequences_ = device.Allocate<int32_t>(sequences_size, true);
  if (beam_size > 1)
    sequences_next_ = device.Allocate<int32_t>(sequences_size, true);

  // The original inputs are not expanded, this expands them in place into the sequences
  auto span = sequences_->CpuSpan();
  for (size_t batch = 0; batch < batch_size; batch++) {
    for (size_t beam = 0; beam < beam_size; beam++) {
      for (int j = 0; j < current_length_; j++) {
        span[(batch * beam_size + beam) * max_length + j] =
            static_cast<int32_t>(input_sequences[batch * current_length_ + j]);
      }
    }
  }
}

DeviceMemorySpan<int32_t> Sequences::GetSequence(size_t batch_beam_index) {
  return sequences_->subspan(batch_beam_index * max_length_, current_length_);
}

int Sequences::GetSequenceLength() const {
  return current_length_;
}

void Sequences::AppendNextTokenToSequences(std::span<const int32_t> batch_beam_indices, std::span<const int32_t> batch_beam_next_tokens) {
  auto sequences_span = sequences_->CpuSpan();
  auto sequences_next_span = sequences_next_->CpuSpan();

  for (ptrdiff_t i = 0; i < batch_beam_size_; i++) {
    int batch_beam_index = batch_beam_indices[i];
    std::span<const int32_t> source = sequences_span.subspan(static_cast<size_t>(batch_beam_index) * max_length_, current_length_);
    std::span<int32_t> target = sequences_next_span.subspan(i * max_length_, current_length_);
    copy(source, target);

    // Append next token to each beam.
    sequences_next_span[i * max_length_ + current_length_] = batch_beam_next_tokens[i];
  }

  ++current_length_;

  // Rotate buffer for next round.
  std::swap(sequences_, sequences_next_);
}

void Sequences::AppendNextTokenToSequences(std::span<const int32_t> next_tokens) {
  auto sequences_span = sequences_->CpuSpan();

  if (g_log.enabled && g_log.append_next_tokens) {
    auto& stream = Log("append_next_tokens");
    DumpSpan(stream, next_tokens);
    stream << std::endl;
  }
  // Append next token to each sequence.
  for (int i = 0; i < batch_beam_size_; i++) {
    sequences_span[i * max_length_ + current_length_] = next_tokens[i];
  }

  ++current_length_;
}

}  // namespace Generators
