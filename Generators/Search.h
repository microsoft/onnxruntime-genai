struct Gpt;

struct SearchParams {
  int num_heads{1};
  int head_size {1};
  int num_beams{1};
  int batch_size {};
  int vocab_size {};
  int sequence_length {};
  int max_length {10};
  int pad_token_id{98};
  int eos_token_id{98};

  int BatchBeamSize() const { return num_beams*batch_size; }
};

struct Search {

  Search(Gpt &model, SearchParams params);

  void SetSequence(gsl::span<const int32_t> input_ids_in_cpu);
  void Run();
  void ProcessLogits();
  void Finalize();

  explicit operator bool() { return !done_; }

  Gpt& model_;
  SearchParams params_;
  Sequences sequences_;
  bool done_{};

  IGreedySearchState<float> search_state_;

  BufferUniquePtr sequences_space_buffer_;
  BufferUniquePtr sequence_lengths_buffer_;
  BufferUniquePtr next_token_scores_buffer_;
  BufferUniquePtr next_tokens_buffer_;
  BufferUniquePtr next_positions_buffer_;
  BufferUniquePtr eos_meet_buffer_;
  BufferUniquePtr temp_topk_buffer_;
  BufferUniquePtr staging_for_past_state_reorder_buffer_;

  std::unique_ptr<OrtValue> output_sequences_;
  std::unique_ptr<OrtValue> position_ids_;
};
