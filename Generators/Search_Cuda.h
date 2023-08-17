#pragma once
namespace Generators {

struct SearchParams_Cuda : SearchParams {
  Ort::Allocator *p_allocator_cuda;
  cudaStream_t cuda_stream;
};

struct Search_Cuda {
  Search_Cuda(SearchParams_Cuda &params);

  gsl::span<int32_t> GetNextTokens();
  gsl::span<int32_t> GetNextIndices();

  int GetSequenceLength();

  bool IsDone() const { cudaStreamSynchronize(params_.cuda_stream); return *done_cpu_; } // TODO: Use an event
  void SetLogits(OrtValue& logits);
  // Extra scoring steps go here

  //
  void CheckForEOS();
  gsl::span<ScoreType> GetScores(int batch_beam_index);
  Sequences& GetSequences() { return sequences_; }

  void SetInputSequence();

  SearchParams_Cuda params_;

  Ort::Allocator& allocator_cpu_;
  Ort::Allocator& allocator_cuda_;

  gsl::span<int32_t> sequences_space_;  // shape (2, beam_size*batch_size, max_length)
  BufferUniquePtr sequences_space_buffer_;

  gsl::span<int32_t> sequence_lengths_;  // shape (beam_size*batch_size)
  BufferUniquePtr sequence_lengths_buffer_;

  gsl::span<bool> eos_meet_;  // shape (beam_size*batch_size)
  BufferUniquePtr eos_meet_buffer_;

  gsl::span<int32_t> next_tokens_;  // shape (beam_size*batch_size)

  gsl::span<ScoreType> next_token_scores_;  // shape (beam_size*batch_size, vocab_size)
  BufferUniquePtr next_token_scores_buffer_;

  cuda_host_unique_ptr<bool> done_cpu_;

  Sequences sequences_;
};

struct GreedySearch_Cuda : Search_Cuda {
  GreedySearch_Cuda(SearchParams_Cuda &params);

  gsl::span<int32_t> GetNextTokens();
  void NextTokensFromLogits();
  void AppendNextTokensToSequences();

 private:

  cuda_host_unique_ptr<int32_t> next_tokens_cpu_;  // shape (beam_size*batch_size)

  BufferUniquePtr next_tokens_buffer_;
  BufferUniquePtr temp_topk_buffer_;
  BufferUniquePtr staging_for_past_state_reorder_buffer_;
};

struct BeamSearch_Cuda : Search_Cuda {
  BeamSearch_Cuda(SearchParams_Cuda &params);

  gsl::span<int32_t> GetNextTokens();
  gsl::span<int32_t> GetNextIndices();
  void NextTokensFromLogits();
  void AppendNextTokensToSequences();
  void Finalize(size_t num_return_sequences, gsl::span<int32_t> output, gsl::span<float> sequence_scores);

 private:
  std::unique_ptr<BeamSearchScorer> beam_scorer_;
};

namespace Processors_Cuda {
void MinLength(Search_Cuda& search, int min_length);
void RepetitionPenalty(Search_Cuda& search, ScoreType penalty);
}  // namespace Processors_Cuda

}  // namespace Generators