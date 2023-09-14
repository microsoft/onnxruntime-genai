#pragma once
#include "sequences_cuda.h"

namespace Generators {

struct BeamSearchScorer_Cuda;

struct SearchParams_Cuda : SearchParams {
  Ort::Allocator* p_allocator_cuda;
  cudaStream_t cuda_stream;
};

struct Search_Cuda {
  Search_Cuda(SearchParams_Cuda &params);

  std::span<int32_t> GetNextTokens();
  std::span<int32_t> GetNextIndices();

  int GetSequenceLength();

  bool IsDone() const { cudaStreamSynchronize(params_.cuda_stream); return *done_cpu_; } // TODO: Use an event
  void SetLogits(std::span<const ScoreType> logits);
  // Extra scoring steps go here

  //
  void CheckForEOS();
  std::span<ScoreType> GetScores(int batch_beam_index);
  std::span<ScoreType> GetScores();
  Sequences_Cuda& GetSequences() { return sequences_; }

  SearchParams_Cuda params_;

  std::span<int32_t> sequence_lengths_;  // shape (beam_size*batch_size)
  std::unique_ptr<int32_t[]> sequence_lengths_buffer_;

  std::span<bool> eos_meet_;  // shape (beam_size*batch_size)
  cuda_unique_ptr<bool> eos_meet_buffer_;

  std::span<int32_t> next_tokens_;  // shape (beam_size*batch_size)

  std::span<ScoreType> next_token_scores_;  // shape (beam_size*batch_size, vocab_size)
  cuda_unique_ptr<ScoreType> next_token_scores_buffer_;

  cuda_host_unique_ptr<bool> done_cpu_;

  Sequences_Cuda sequences_;
};

struct GreedySearch_Cuda : Search_Cuda {
  GreedySearch_Cuda(SearchParams_Cuda &params);

  std::span<int32_t> GetNextTokens();
  void NextTokensFromLogits();
  void AppendNextTokensToSequences();

 private:

  cuda_unique_ptr<int32_t> next_tokens_buffer_;
};

struct BeamSearch_Cuda : Search_Cuda {
  BeamSearch_Cuda(SearchParams_Cuda &params);
  ~BeamSearch_Cuda();

  std::span<int32_t> GetNextTokens();
  std::span<int32_t> GetNextIndices();

  void NextTokensFromLogits();
  void AppendNextTokensToSequences();
  void Finalize(size_t num_return_sequences, std::span<int32_t> output, std::span<float> sequence_scores);

  bool IsDone() const;

 private:
  std::unique_ptr<BeamSearchScorer_Cuda> beam_scorer_;

  cuda_unique_ptr<int32_t> topk_next_tokens_;
  cuda_unique_ptr<int32_t> topk_next_indices_;
  cuda_unique_ptr<ScoreType> topk_next_scores_;

  // temp buffer for topk computation, including:
  // 1st stage needs:
  //   temp score: (batch_size * num_beams * parts_vocab, 2 * num_beams)
  //   temp token: (batch_size * num_beams * parts_vocab, 2 * num_beams)
  // 2nd stage needs:
  //   temp score: (batch_size * num_beams, 2 * num_beams)
  //   temp token: (batch_size * num_beams, 2 * num_beams)
  // in total, it will be:
  // 2 * (batch_size * num_beams * (parts_vocab + 1), 2 * num_beams)
  cuda_unique_ptr<ScoreType> topk_buffer_;
};

namespace Processors_Cuda {
void MinLength(Search_Cuda& search, int min_length);
void RepetitionPenalty(Search_Cuda& search, ScoreType penalty);
}  // namespace Processors_Cuda

}  // namespace Generators