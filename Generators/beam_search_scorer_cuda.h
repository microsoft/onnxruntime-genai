namespace Generators {

struct BeamSearchScorer_Cuda {
  BeamSearchScorer_Cuda(const SearchParams_Cuda& parameters,
                       OrtAllocator& allocator_cpu, OrtAllocator& allocator_cuda);

  void Process(ISequences& sequences,
               gsl::span<const float> next_scores,
               gsl::span<const int32_t> next_tokens,
               gsl::span<const int32_t> next_indices);

  void Finalize(ISequences& sequences,
                size_t num_return_sequences,
                gsl::span<int32_t> output_sequences,
                gsl::span<float> output_sequence_scores);

  bool IsDone() const { return false; }  // For CUDA we speculatively run the next step while we wait for the GPU to report status. We use 'IsDoneLater()' for this
  bool IsDoneLater() const;

  gsl::span<float> GetNextScores() { return next_beam_scores_; }
  gsl::span<int32_t> GetNextTokens() { return next_beam_tokens_; }
  gsl::span<int32_t> GetNextIndicesCPU() {
    cudaMemcpyAsync(next_beam_indices_cpu_.data(), next_beam_indices_.data(), next_beam_indices_.size_bytes(), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    return next_beam_indices_cpu_;
  }
  gsl::span<int32_t> GetNextIndicesGPU() { return next_beam_indices_; }

 private:
  mutable cuda_event_holder event_process_complete_;
  cuda_host_unique_ptr<cuda::BeamScorerState> state_cpu_;
  cuda_unique_ptr<cuda::BeamScorerState> state_gpu_;
  cudaStream_t stream_;

  IAllocatorUniquePtr<float> next_beam_scores_ptr_;
  gsl::span<float> next_beam_scores_;

  IAllocatorUniquePtr<int32_t> next_beam_tokens_ptr_;
  gsl::span<int32_t> next_beam_tokens_;

  IAllocatorUniquePtr<int32_t> next_beam_indices_ptr_;
  gsl::span<int32_t> next_beam_indices_;

  IAllocatorUniquePtr<int32_t> next_beam_indices_cpu_ptr_;
  gsl::span<int32_t> next_beam_indices_cpu_;

  IAllocatorUniquePtr<int32_t> hypothesis_buffer_ptr_;  // Allocated buffer to hold all hypotheses
  gsl::span<int32_t> hypothesis_buffer_;                // Span of the allocated buffer
  size_t hypothesis_buffer_used_{};                     // Offset of available buffer, or length of used buffer.

  IAllocatorUniquePtr<cuda::HypothesisScore> hypothesis_scores_ptr_;  // num_beams_ * batch_size_, divided into num_beams_ chunks per BeamHypothesis in beam_hyps_
  IAllocatorUniquePtr<cuda::BeamHypotheses> beam_hyps_ptr_;
  gsl::span<cuda::BeamHypotheses> beam_hyps_;  // Shape is batch_size_
};

}
