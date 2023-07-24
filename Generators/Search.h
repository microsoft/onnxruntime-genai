struct SearchParams {
  int num_heads{1};
  int head_size {1};
  int num_beams{1};
  int batch_size {};
  int vocab_size {};
  int sequence_length {};
  int max_length {};
};

struct Search {

  Search(SearchParams params);

  explicit operator bool() { return true; }

  SearchParams params_;
  Sequences sequences_;

  IGreedySearchState<float> search_state_;

  BufferUniquePtr sequences_space_buffer_;
  BufferUniquePtr sequence_lengths_buffer_;
  BufferUniquePtr next_token_scores_buffer_;
  BufferUniquePtr next_tokens_buffer_;
  BufferUniquePtr next_positions_buffer_;
  BufferUniquePtr eos_meet_buffer_;
  BufferUniquePtr temp_topk_buffer_;
  BufferUniquePtr staging_for_past_state_reorder_buffer_;

  std::unique_ptr<OrtValue> output_sequences;
};
