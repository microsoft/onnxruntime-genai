// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Generators {

struct BeamSearchScorer_Cuda {
  BeamSearchScorer_Cuda(const GeneratorParams& parameters);

  void Process(Sequences& sequences,
               std::span<const float> next_scores,
               std::span<const int32_t> next_tokens,
               std::span<const int32_t> next_indices);

  void Finalize(Sequences& sequences,
                size_t num_return_sequences);

  bool IsDone() const { return false; }  // For CUDA we speculatively run the next step while we wait for the GPU to report status. We use 'IsDoneLater()' for this
  bool IsDoneLater() const;

  DeviceSpan<float> GetNextScores() { return next_beam_scores_; }
  DeviceSpan<int32_t> GetNextTokens() { return next_beam_tokens_; }
  DeviceSpan<int32_t> GetNextIndices() { return next_beam_indices_; }
  DeviceSpan<int32_t> GetBeamHypothesis(size_t batch_id, size_t beam_id);

 private:
  mutable cuda_event_holder event_process_complete_;
  cuda_host_unique_ptr<cuda::BeamScorerState> state_cpu_;
  cuda_unique_ptr<cuda::BeamScorerState> state_gpu_;
  cudaStream_t stream_;

  DeviceSpan<float> next_beam_scores_;
  DeviceSpan<int32_t> next_beam_tokens_;
  DeviceSpan<int32_t> next_beam_indices_;

  DeviceSpan<int32_t> hypothesis_buffer_;  // Allocated buffer to hold all hypotheses
  size_t hypothesis_buffer_used_{};        // Offset of available buffer, or length of used buffer.

  cuda_unique_ptr<cuda::HypothesisScore> hypothesis_scores_ptr_;  // num_beams_ * batch_size_, divided into num_beams_ chunks per BeamHypothesis in beam_hyps_
  cuda_unique_ptr<cuda::BeamHypotheses> beam_hyps_ptr_;
  gpu_span<cuda::BeamHypotheses> beam_hyps_;  // Shape is batch_size_
};

}  // namespace Generators
