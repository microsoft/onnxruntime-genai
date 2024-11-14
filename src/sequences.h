#pragma once
namespace Generators {

// This class keeps track of sequences generated.
struct Sequences {
  Sequences(const GeneratorParams& params)
      : max_length_{params.search.max_length},
        current_length_{static_cast<int>(params.input_ids.size()) / params.batch_size} {
    assert(current_length_ * params.batch_size == params.input_ids.size());  // Ensure size divided perfectly

    const size_t sequences_size = static_cast<size_t>(params.BatchBeamSize()) * max_length_;
    sequences_ = params.p_device->Allocate<int32_t>(sequences_size);
    if (params.search.num_beams > 1)
      sequences_next_ = params.p_device->Allocate<int32_t>(sequences_size);
  }

  // Returns a sequence of word IDs for a given beam index ( beam_index < batch_beam_size).
  DeviceSpan<int32_t> GetSequence(size_t batch_beam_index) {
    return sequences_.subspan(batch_beam_index * max_length_, current_length_);
  }

  DeviceSpan<int32_t> GetSequences() { return sequences_; }
  DeviceSpan<int32_t> GetNextSequences() { return sequences_next_; }

  // Returns current sequence length.
  int GetSequenceLength() const { return current_length_; }

  // After tokens are appended, this function must be called to update the state & log the tokens
  void AfterAppendNextTokens(DeviceSpan<int32_t> next_tokens);

  const int max_length_;

 private:
  // Two buffers of shape (batch_size, num_beams, max_seq_length) to store sequences.
  // At each time, there is only one buffer is active. The other one will be active in next token.
  // Each AppendNextTokenToSequences call will trigger a rotation of active buffer.
  DeviceSpan<int32_t> sequences_;
  DeviceSpan<int32_t> sequences_next_;  // This only exists for beam search, to allow for the easy reordering of sequences

  int current_length_;
};

}  // namespace Generators