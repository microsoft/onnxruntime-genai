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

  if (beam_size == 1) {
    sequences_buffer_ = std::make_unique<int32_t[]>(sequences_size);
    sequences_ = cpu_span<int32_t>(sequences_buffer_.get(), sequences_size);
  } else {
    sequences_buffer_ = std::make_unique<int32_t[]>(2 * sequences_size);
    sequences_ = cpu_span<int32_t>(sequences_buffer_.get(), sequences_size);
    sequences_next_ = cpu_span<int32_t>(sequences_buffer_.get() + sequences_size, sequences_size);
  }

  // The original inputs are not expanded, this expands them in place into the sequences
  for (size_t batch = 0; batch < batch_size; batch++) {
    for (size_t beam = 0; beam < beam_size; beam++) {
      for (int j = 0; j < current_length_; j++) {
        sequences_[(batch * beam_size + beam) * max_length + j] =
            static_cast<int32_t>(input_sequences[batch * current_length_ + j]);
      }
    }
  }
}

cpu_span<int32_t> Sequences::GetSequence(size_t batch_beam_index) {
  auto span = sequences_.subspan(batch_beam_index * max_length_, current_length_);
  return cpu_span<int32_t>{span.data(), span.size()};
}

int Sequences::GetSequenceLength() const {
  return current_length_;
}

void Sequences::AppendNextTokenToSequences(std::span<const int32_t> batch_beam_indices, std::span<const int32_t> batch_beam_next_tokens) {
  for (ptrdiff_t i = 0; i < batch_beam_size_; i++) {
    int batch_beam_index = batch_beam_indices[i];
    std::span<const int32_t> source = sequences_.subspan(batch_beam_index * max_length_, current_length_);
    std::span<int32_t> target = sequences_next_.subspan(i * max_length_, current_length_);
    copy(source, target);

    // Append next token to each beam.
    sequences_next_[i * max_length_ + current_length_] = batch_beam_next_tokens[i];
  }

  ++current_length_;

  // Rotate buffer for next round.
  std::swap(sequences_, sequences_next_);
}

void Sequences::AppendNextTokenToSequences(std::span<const int32_t> next_tokens) {
  if (g_log.enabled && g_log.append_next_tokens) {
    auto& stream = Log("append_next_tokens");
    DumpSpan(stream, next_tokens);
    stream << std::endl;
  }
  // Append next token to each sequence.
  for (int i = 0; i < batch_beam_size_; i++) {
    sequences_[i * max_length_ + current_length_] = next_tokens[i];
  }

  ++current_length_;
}

}  // namespace Generators
