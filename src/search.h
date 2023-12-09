#include "sequences.h"

namespace Generators {

struct BeamSearchScorer;

struct Search_Cpu : Search {
  Search_Cpu(SearchParams params);

  int GetSequenceLength() const override;
  RoamingArray<int32_t> GetSequenceLengths() override { return sequence_lengths_; }
  RoamingArray<int32_t> GetSequence(int index) override { return sequences_.GetSequence(index); }

  bool IsDone() const override { return done_; }
  void SetLogits(RoamingArray<float> logits) override;
  // Extra scoring steps go here

  //
  std::span<ScoreType> GetScores(int batch_beam_index);
  Sequences& GetSequences() { return sequences_; }

  SearchParams params_;

  cpu_span<int32_t> sequence_lengths_;  // shape (beam_size*batch_size)
  std::unique_ptr<int32_t[]> sequence_lengths_buffer_;

  cpu_span<int32_t> next_tokens_;  // shape (beam_size*batch_size)

  std::span<ScoreType> next_token_scores_;  // shape (beam_size*batch_size, vocab_size)
  std::unique_ptr<ScoreType[]> next_token_scores_buffer_;

  Sequences sequences_;
  bool done_{};
};

struct GreedySearch_Cpu : Search_Cpu {
  GreedySearch_Cpu(SearchParams params);

  RoamingArray<int32_t> GetNextTokens() override;

  void SelectTop() override;
  void SampleTopK(int k, float temperature) override;
  void SampleTopP(float p, float temperature) override;

 private:
  bool PadIfAlreadyEOS(size_t batch_id);
  void SetNextToken(size_t batch_id, int32_t token);
  void AppendNextTokensToSequences();

  std::unique_ptr<int32_t[]> next_tokens_buffer_;
  std::unique_ptr<int32_t[]> temp_topk_buffer_;

  std::span<bool> eos_seen_;  // shape (batch_size)
  std::unique_ptr<bool[]> eos_seen_buffer_;
  int not_done_count_{params_.batch_size};  // When zero, every batch entry is done (starts at batch_size_)
};

struct BeamSearch_Cpu : Search_Cpu {
  BeamSearch_Cpu(SearchParams params);
  ~BeamSearch_Cpu();

  RoamingArray<int32_t> GetNextTokens() override;
  RoamingArray<int32_t> GetNextIndices() override;

  void SelectTop() override;

  void Finalize(size_t num_return_sequences, RoamingArray<int32_t> output, RoamingArray<float> sequence_scores) override;

 private:
  void AppendNextTokensToSequences();

  std::unique_ptr<BeamSearchScorer> beam_scorer_;
};

namespace Processors {
void MinLength(Search_Cpu& search, int min_length);
void RepetitionPenalty(Search_Cpu& search, ScoreType penalty);
}  // namespace Processors

}  // namespace Generators