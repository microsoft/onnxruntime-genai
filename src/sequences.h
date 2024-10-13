#pragma once
namespace Generators {

// This class keeps track of sequences generated.
struct Sequences {
  Sequences(std::span<const int32_t> input_sequence, int batch_size, int beam_size, int max_length);

  // Returns a sequence of word IDs for a given beam index ( beam_index < batch_beam_size).
  DeviceMemorySpan<int32_t> GetSequence(size_t batch_beam_index);
  DeviceMemory<int32_t>& GetSequences() { return *sequences_; }

  // Returns current sequence length.
  int GetSequenceLength() const;

  // Used by Beam search:
  // Shuffles sequences around based on batch_beam_indices, then append next token to selected sequences.
  void AppendNextTokenToSequences(std::span<const int32_t> batch_beam_indices, std::span<const int32_t> batch_beam_next_tokens);

  // Used by Greedy search:
  void AppendNextTokenToSequences(std::span<const int32_t> next_tokens);

 private:
  // Two buffers of shape (batch_size, num_beams, max_seq_length) to store sequences.
  // At each time, there is only one buffer is active. The other one will be active in next token.
  // Each AppendNextTokenToSequences call will trigger a rotation of active buffer.
  std::shared_ptr<DeviceMemory<int32_t>> sequences_;
  std::shared_ptr<DeviceMemory<int32_t>> sequences_next_;  // This only exists for beam search, to allow for the easy reordering of sequences

  int batch_beam_size_;
  int max_length_;
  int current_length_;
};

}  // namespace Generators