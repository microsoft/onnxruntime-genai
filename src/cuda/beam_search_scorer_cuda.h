namespace Generators {

struct BeamSearchScorer_Cuda {
  BeamSearchScorer_Cuda(const GeneratorParams& parameters);

  void Process(Sequences_Cuda& sequences,
               std::span<const float> next_scores,
               std::span<const int32_t> next_tokens,
               std::span<const int32_t> next_indices);

  void Finalize(Sequences_Cuda& sequences,
                size_t num_return_sequences);

  bool IsDone() const { return false; }  // For CUDA we speculatively run the next step while we wait for the GPU to report status. We use 'IsDoneLater()' for this
  bool IsDoneLater() const;

  gpu_span<float> GetNextScores() { return next_beam_scores_; }
  gpu_span<int32_t> GetNextTokens() { return next_beam_tokens_; }
  cpu_span<int32_t> GetNextIndicesCPU() {
    cudaMemcpyAsync(next_beam_indices_cpu_.data(), next_beam_indices_.data(), next_beam_indices_.size_bytes(), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    return next_beam_indices_cpu_;
  }
  gpu_span<int32_t> GetNextIndicesGPU() { return next_beam_indices_; }
  DeviceMemorySpan<int32_t> GetBeamHypothesis(size_t batch_id, size_t beam_id) const;

 private:
  mutable cuda_event_holder event_process_complete_;
  cuda_host_unique_ptr<cuda::BeamScorerState> state_cpu_;
  cuda_unique_ptr<cuda::BeamScorerState> state_gpu_;
  cudaStream_t stream_;

  cuda_unique_ptr<float> next_beam_scores_ptr_;
  gpu_span<float> next_beam_scores_;

  cuda_unique_ptr<int32_t> next_beam_tokens_ptr_;
  gpu_span<int32_t> next_beam_tokens_;

  cuda_unique_ptr<int32_t> next_beam_indices_ptr_;
  gpu_span<int32_t> next_beam_indices_;

  std::unique_ptr<int32_t[]> next_beam_indices_cpu_ptr_;
  cpu_span<int32_t> next_beam_indices_cpu_;

  std::shared_ptr<DeviceMemory<int32_t>> hypothesis_buffer_ptr_;  // Allocated buffer to hold all hypotheses
  std::span<int32_t> hypothesis_buffer_;                          // Span of the allocated buffer
  size_t hypothesis_buffer_used_{};                               // Offset of available buffer, or length of used buffer.

  cuda_unique_ptr<cuda::HypothesisScore> hypothesis_scores_ptr_;  // num_beams_ * batch_size_, divided into num_beams_ chunks per BeamHypothesis in beam_hyps_
  cuda_unique_ptr<cuda::BeamHypotheses> beam_hyps_ptr_;
  gpu_span<cuda::BeamHypotheses> beam_hyps_;  // Shape is batch_size_
};

}  // namespace Generators
