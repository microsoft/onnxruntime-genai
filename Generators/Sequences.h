#pragma once
namespace Generators {

// This class keeps track of sequences generated.
class Sequences {
 public:
  // Initialize the sequence.
  void Init(std::span<int32_t> buffer, int batch_beam_size, int sequence_length, int max_length);
  void InitDevice(std::span<int32_t> buffer);

  // Returns a sequence of word IDs for a given beam index ( beam_index < batch_beam_size).
  std::span<const int32_t> GetSequence(int beam_index) const;
  std::span<const int32_t> GetCurrentDeviceSequences() const { return device_sequences[current_sequences_buffer]; }
  std::span<int32_t> GetNextDeviceSequences() { return device_sequences[current_sequences_buffer ^ 1]; }

  // Returns current sequence length.
  int GetSequenceLength() const;

#ifdef DEBUG_GENERATION
  // Print the sequences to StdOut in debug mode
  void PrintSequences(const IConsoleDumper* dumper) const;
#endif

  // Select sequences based on beam indices, then append next token to selected sequences.
  void AppendNextTokenToSequences(
      std::span<int32_t> beam_indices,
      std::span<int32_t> beam_next_tokens);

  void AppendNextTokenToSequences(
      std::span<const int32_t> next_tokens);

  void AfterDeviceAppendedNextToken();

 private:
  // Two buffers of shape (batch_size, num_beams, max_seq_length) to store sequences.
  // At each time, there is only one buffer is active. The other one will be active in next token.
  // Each AppendNextTokenToSequences call will trigger a rotation of active buffer.
  std::span<int32_t> sequences[2];
  std::span<int32_t> device_sequences[2];

  // Index (either 0 or 1) of two buffers that is currently is active.
  int current_sequences_buffer;

  int batch_beam_size_;
  int max_length_;
  int current_length_;
};

}