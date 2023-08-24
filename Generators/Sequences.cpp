// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "Generators.h"
#include "sequences.h"

void Sequences::Init(std::span<int32_t> buffer, int batch_beam_size, int sequence_length, int max_length) {
  size_t sequences_size = SafeInt<size_t>(batch_beam_size) * max_length;
  assert(buffer.size() == sequences_size + sequences_size);

  sequences[0] = buffer.subspan(0, sequences_size);
  sequences[1] = buffer.subspan(sequences_size);

  current_sequences_buffer = 0;

  batch_beam_size_ = batch_beam_size;
  max_length_ = max_length;
  current_length_ = sequence_length;
}

void Sequences::InitDevice(std::span<int32_t> buffer) {
  device_sequences[0] = buffer.subspan(0, buffer.size() / 2);
  device_sequences[1] = buffer.subspan(buffer.size() / 2);
}

std::span<const int32_t> Sequences::GetSequence(int beam_index) const {
  std::span<const int32_t> buffer = sequences[current_sequences_buffer];\
  return buffer.subspan(beam_index * max_length_, current_length_);
}

int Sequences::GetSequenceLength() const {
  return current_length_;
}

#ifdef DEBUG_GENERATION
void Sequences::PrintSequences(const IConsoleDumper* dumper) const {
  for (int i = 0; i < batch_beam_size_; i++) {
    std::span<const int32_t> sequence = GetSequence(i);
    dumper->Print("sequences", i, false);
    dumper->Print(nullptr, sequence.data(), 1, current_length_);
  }
}
#endif

void Sequences::AppendNextTokenToSequences(
    std::span<int32_t> beam_indices,
    std::span<int32_t> beam_next_tokens) {
  std::span<const int32_t> input = sequences[current_sequences_buffer];
  std::span<int32_t> output = sequences[current_sequences_buffer ^ 1];

  for (int i = 0; i < batch_beam_size_; i++) {
    int beam_index = beam_indices[i];
    std::span<const int32_t> source = input.subspan(SafeInt<size_t>(beam_index) * max_length_, current_length_);
    std::span<int32_t> target = output.subspan(SafeInt<size_t>(i) * max_length_, current_length_);
    copy(source, target);

    // Append next token to each beam.
    output[SafeInt<size_t>(i) * max_length_ + current_length_] = beam_next_tokens[i];
  }

  ++current_length_;

  // Rotate buffer for next round.
  current_sequences_buffer ^= 1;
}

void Sequences::AppendNextTokenToSequences(std::span<const int32_t> next_tokens) {
  auto output = sequences[0];

  // Append next token to each sequence.
  for (int i = 0; i < batch_beam_size_; i++) {
    output[SafeInt<size_t>(i) * max_length_ + current_length_] = next_tokens[i];
  }

  ++current_length_;
}

void Sequences::AfterDeviceAppendedNextToken() {
  ++current_length_;
  current_sequences_buffer ^= 1;
}
