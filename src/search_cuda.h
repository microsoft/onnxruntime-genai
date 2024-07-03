#pragma once
#include "sequences_cuda.h"
#include "search_cuda.cuh"
#include "cuda_sampling.cuh"

namespace Generators {

struct BeamSearchScorer_Cuda;

struct Search_Cuda : Search {
  Search_Cuda(const GeneratorParams& params);

  int GetSequenceLength() const override;
  RoamingArray<int32_t> GetSequenceLengths() override { return sequence_lengths_; }
  RoamingArray<int32_t> GetSequence(size_t index) override { return sequences_.GetSequence(index); }

  bool IsDone() const {
    cudaStreamSynchronize(params_->cuda_stream);
    return *done_cpu_;
  }  // TODO: Use an event
  void SetLogits(RoamingArray<float> logits);

  void ApplyMinLength(int min_length) override;
  void ApplyRepetitionPenalty(float penalty) override;

  std::span<float> GetScores(int batch_beam_index);
  std::span<float> GetScores();
  Sequences_Cuda& GetSequences() { return sequences_; }

  cpu_span<int32_t> sequence_lengths_;  // shape (beam_size*batch_size)
  std::unique_ptr<int32_t[]> sequence_lengths_buffer_;

  gpu_span<bool> eos_meet_;  // shape (beam_size*batch_size)
  cuda_unique_ptr<bool> eos_meet_buffer_;

  gpu_span<int32_t> next_tokens_;  // shape (beam_size*batch_size)

  gpu_span<float> next_token_scores_;  // shape (beam_size*batch_size, vocab_size)

  cuda_host_unique_ptr<bool> done_cpu_;

  Sequences_Cuda sequences_;
};

struct GreedySearch_Cuda : Search_Cuda {
  GreedySearch_Cuda(const GeneratorParams& params);

  RoamingArray<int32_t> GetNextTokens() override;
  RoamingArray<int32_t> GetNextIndices() override { return gpu_span<int32_t>{}; }

  void SelectTop() override;
  void SampleTopK(int k, float t) override;
  void SampleTopP(float p, float t) override;
  void SampleTopKTopP(int k, float p, float t) override;

 private:
  void CheckForEOS();
  void AppendNextTokensToSequences();

  cuda_unique_ptr<int32_t> next_tokens_buffer_;
  std::unique_ptr<cuda::ArgMaxData> argmaxdata_;
  std::unique_ptr<cuda::SamplingData> samplingdata_;
};

struct BeamSearch_Cuda : Search_Cuda {
  BeamSearch_Cuda(const GeneratorParams& params);
  ~BeamSearch_Cuda();

  RoamingArray<int32_t> GetNextTokens() override;
  RoamingArray<int32_t> GetNextIndices() override;

  void SelectTop() override;
  void Finalize(size_t num_return_sequences, RoamingArray<int32_t> output, RoamingArray<float> sequence_scores) override;

  bool IsDone() const;

 private:
  void AppendNextTokensToSequences();

  std::unique_ptr<BeamSearchScorer_Cuda> beam_scorer_;

  cuda_unique_ptr<int32_t> topk_next_tokens_;
  cuda_unique_ptr<int32_t> topk_next_indices_;
  cuda_unique_ptr<float> topk_next_scores_;

  // temp buffer for topk computation, including:
  // 1st stage needs:
  //   temp score: (batch_size * num_beams * parts_vocab, 2 * num_beams)
  //   temp token: (batch_size * num_beams * parts_vocab, 2 * num_beams)
  // 2nd stage needs:
  //   temp score: (batch_size * num_beams, 2 * num_beams)
  //   temp token: (batch_size * num_beams, 2 * num_beams)
  // in total, it will be:
  // 2 * (batch_size * num_beams * (parts_vocab + 1), 2 * num_beams)
  cuda_unique_ptr<float> topk_buffer_;
};

namespace Processors_Cuda {
void MinLength(Search_Cuda& search, int min_length);
void RepetitionPenalty(Search_Cuda& search, float penalty);
}  // namespace Processors_Cuda

}  // namespace Generators