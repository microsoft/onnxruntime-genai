// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda_runtime.h>
#include "search_cuda.cuh"
#include "cuda_sampling.h"

namespace Generators {

struct BeamSearchScorer_Cuda;

struct Search_Cuda : Search {
  Search_Cuda(const GeneratorParams& params);

  DeviceSpan<int32_t> GetSequenceLengths() override { return sequence_lengths_; }

  bool IsDone() const {
    cudaStreamSynchronize(GetStream());
    return *done_cpu_;
  }  // TODO: Use an event

  DeviceSpan<float> GetLogits() const override;
  void SetLogits(DeviceSpan<float> logits) override;

  void ApplyMinLength(int min_length) override;
  void ApplyRepetitionPenalty(float penalty) override;

  std::span<float> GetScores(int batch_beam_index);
  std::span<float> GetScores();

  DeviceSpan<int32_t> sequence_lengths_;  // shape (beam_size*batch_size)

  gpu_span<bool> eos_seen_;  // shape (beam_size*batch_size)
  cuda_unique_ptr<bool> eos_seen_buffer_;
  DeviceSpan<int32_t> eos_token_ids_;

  gpu_span<int32_t> next_tokens_;        // shape (beam_size*batch_size)
  DeviceSpan<float> next_token_scores_;  // shape (beam_size*batch_size, vocab_size)

  cuda_host_unique_ptr<bool> done_cpu_;
};

struct GreedySearch_Cuda : Search_Cuda {
  GreedySearch_Cuda(const GeneratorParams& params);

  DeviceSpan<int32_t> GetNextTokens() override;
  DeviceSpan<int32_t> GetNextIndices() override { return {}; }

  void SelectTop() override { SampleTopKTopP(1, 0.0, 1.0); }
  void SampleTopK(int k, float t) override { SampleTopKTopP(k, 1.0, t); }
  void SampleTopP(float p, float t) override { SampleTopKTopP(-1, p, t); }
  void SampleTopKTopP(int k, float p, float t) override;
  void AppendTokens(DeviceSpan<int32_t>& next_tokens) override;  // shape (batch_size, sequence_length)
  void RewindTo(size_t index) override;

 private:
  DeviceSpan<int32_t> next_tokens_buffer_;
  std::unique_ptr<cuda::ArgMaxData> argmaxdata_;
  std::unique_ptr<cuda::SamplingData> samplingdata_;
};

struct BeamSearch_Cuda : Search_Cuda {
  BeamSearch_Cuda(const GeneratorParams& params);
  ~BeamSearch_Cuda();

  DeviceSpan<int32_t> GetNextTokens() override;
  DeviceSpan<int32_t> GetNextIndices() override;
  // In Beam Search there are batch_size * num_beams sequences. Index is batch_id * num_beams + beam_id... Easier to use the other version.
  DeviceSpan<int32_t> GetSequence(size_t index) override;
  DeviceSpan<int32_t> GetSequence(size_t batch_id, size_t beam_id);

  void AppendTokens(DeviceSpan<int32_t>& next_tokens) override;

  void SelectTop() override;

  bool IsDone() const;

 private:
  void Finalize(size_t num_return_sequences);

  bool finalized_{};  // To avoid calling Finalize multiple times

  std::unique_ptr<BeamSearchScorer_Cuda> beam_scorer_;

  cuda_unique_ptr<int32_t> topk_next_tokens_;
  cuda_unique_ptr<int32_t> topk_next_indices_;
  cuda_unique_ptr<float> topk_next_scores_;
  cuda_unique_ptr<float> softmax_buffer_;

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