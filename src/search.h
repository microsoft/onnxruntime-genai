#include "sequences.h"

namespace Generators {

struct BeamSearchScorer;

struct Search {
  Search(SearchParams params);

  std::span<int32_t> GetNextTokens();
  std::span<int32_t> GetNextIndices();

  int GetSequenceLength();

  bool IsDone() const { return done_; }
  void SetLogits(std::span<const ScoreType> logits);
  // Extra scoring steps go here

  //
  std::span<ScoreType> GetScores(int batch_beam_index);
  Sequences& GetSequences() { return sequences_; }

  SearchParams params_;

  std::span<int32_t> sequence_lengths_;  // shape (beam_size*batch_size)
  std::unique_ptr<int32_t[]> sequence_lengths_buffer_;

  std::span<int32_t> next_tokens_;  // shape (beam_size*batch_size)

  std::span<ScoreType> next_token_scores_;  // shape (beam_size*batch_size, vocab_size)
  std::unique_ptr<ScoreType[]> next_token_scores_buffer_;

  Sequences sequences_;
  bool done_{};
};

struct GreedySearch : Search {
  GreedySearch(SearchParams params);

  std::span<int32_t> GetNextTokens();

  void SelectTop1();

 private:
  void AppendNextTokensToSequences();

  std::unique_ptr<int32_t[]> next_tokens_buffer_;
  std::unique_ptr<int32_t[]> temp_topk_buffer_;

  std::span<bool> eos_seen_;  // shape (batch_size)
  std::unique_ptr<bool[]> eos_seen_buffer_;
  int not_done_count_{params_.batch_size};  // When zero, every batch entry is done (starts at batch_size_)
};

struct BeamSearch : Search {
  BeamSearch(SearchParams params);
  ~BeamSearch();

  std::span<int32_t> GetNextTokens();
  std::span<int32_t> GetNextIndices();

  void SelectTopK();

  void Finalize(size_t num_return_sequences, std::span<int32_t> output, std::span<float> sequence_scores);

 private:
  void AppendNextTokensToSequences();

  std::unique_ptr<BeamSearchScorer> beam_scorer_;
};

namespace Processors {
void MinLength(Search& search, int min_length);
void RepetitionPenalty(Search& search, ScoreType penalty);
}  // namespace Processors

}  // namespace Generators