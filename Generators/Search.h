namespace Generators {

struct BeamSearchScorer;

struct SearchParams {
  int num_beams{1};
  int batch_size {};
  int sequence_length {};
  int max_length {10};
  int pad_token_id{98};
  int eos_token_id{98};
  int vocab_size {};

  float length_penalty{1.0f};
  bool early_stopping{false};

  int BatchBeamSize() const { return num_beams*batch_size; }

  const int32_t *input_ids{}; // Array of [sequence_length][batchsize]
};

struct GreedySearchParams : SearchParams {
};

struct BeamSearchParams : SearchParams {
};

struct Search {

  Search(SearchParams params);

  std::span<int32_t> GetNextTokens();
  std::span<int32_t> GetNextIndices();

  int GetSequenceLength();

  bool IsDone() const { return done_; }
  void SetLogits(OrtValue& logits);
  // Extra scoring steps go here
  
  //
  void CheckForEOS();
  std::span<ScoreType> GetScores(int batch_beam_index);
  Sequences& GetSequences() { return sequences_; }

  void SetInputSequence();

  SearchParams params_;

  std::span<int32_t> sequences_space_;  // shape (2, beam_size*batch_size, max_length)
  BufferUniquePtr sequences_space_buffer_;

  std::span<int32_t> sequence_lengths_;  // shape (beam_size*batch_size)
  BufferUniquePtr sequence_lengths_buffer_;

  std::span<bool> eos_meet_;  // shape (beam_size*batch_size)
  BufferUniquePtr eos_meet_buffer_;

  std::span<int32_t> next_tokens_;  // shape (beam_size*batch_size)

  std::span<ScoreType> next_token_scores_;  // shape (beam_size*batch_size, vocab_size)
  BufferUniquePtr next_token_scores_buffer_;

  Sequences sequences_;
  bool done_{};
};

struct GreedySearch : Search {
  GreedySearch(SearchParams params);

  std::span<int32_t> GetNextTokens();
  void NextTokensFromLogits();
  void AppendNextTokensToSequences();

private:
  BufferUniquePtr next_tokens_buffer_;
  BufferUniquePtr temp_topk_buffer_;
  BufferUniquePtr staging_for_past_state_reorder_buffer_;
};

struct BeamSearch : Search {
  BeamSearch(SearchParams params);

  std::span<int32_t> GetNextTokens();
  std::span<int32_t> GetNextIndices();
  void NextTokensFromLogits();
  void AppendNextTokensToSequences();
  void Finalize(size_t num_return_sequences, std::span<int32_t> output, std::span<float> sequence_scores);

private:
  std::unique_ptr<BeamSearchScorer> beam_scorer_;
};

namespace Processors {
  void MinLength(Search& search, int min_length);
  void RepetitionPenalty(Search& search, ScoreType penalty);
}

}