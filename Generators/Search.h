struct Model
{
  virtual void CreateInputs(gsl::span<int32_t> sequence_lengths)=0;
  virtual void UpdateInputs(gsl::span<const int32_t> next_tokens, OrtValue& position_ids, gsl::span<const int32_t> beam_indices, int current_length)=0;
  virtual OrtValue& GetInputIds() = 0;
  virtual OrtValue& GetLogits() = 0;
  virtual void Run() = 0;
  virtual int GetVocabSize() = 0;
};

struct BeamSearchScorer;

struct SearchParams {
  int num_beams{1};
  int batch_size {};
  int sequence_length {};
  int max_length {10};
  int pad_token_id{98};
  int eos_token_id{98};

  float length_penalty{1.0f};
  bool early_stopping{false};

  int BatchBeamSize() const { return num_beams*batch_size; }
};

struct Search {

  Search(Model &model, SearchParams params);

  void SetSequence(gsl::span<const int32_t> input_ids_in_cpu);

  bool IsDone() const { return done_; }
  void RunModel();
  // Extra scoring steps go here
  
  //
  void NextTokensFromLogits();
  void CheckForEOS();
  void AppendNextTokensToSequences();

  void Finalize(size_t num_return_sequences, gsl::span<int32_t> output, gsl::span<float> sequence_scores);

  gsl::span<ScoreType> GetScores(int batch_beam_index);

  Model& model_;
  SearchParams params_;
  Sequences sequences_;
  bool done_{};
  bool first_run_{true};
  int vocab_size_{model_.GetVocabSize()};

  IGreedySearchState search_state_;

  std::unique_ptr<BeamSearchScorer> beam_scorer_;
  
  BufferUniquePtr sequences_space_buffer_;
  BufferUniquePtr sequence_lengths_buffer_;
  BufferUniquePtr next_token_scores_buffer_;
  BufferUniquePtr next_tokens_buffer_;
  BufferUniquePtr next_positions_buffer_;
  BufferUniquePtr eos_meet_buffer_;
  BufferUniquePtr temp_topk_buffer_;
  BufferUniquePtr staging_for_past_state_reorder_buffer_;

  std::unique_ptr<OrtValue> position_ids_;
};

namespace Processors {
  void MinLength(Search& search, int min_length);
  void RepetitionPenalty(Search& search, ScoreType penalty);
}
