#include "Generators.h"
#include "Search_Cuda.h"
#include "beam_search_scorer.h"

namespace Generators {

struct CudaBeamSearchScorer : transformers::IBeamScorer {
  CudaBeamSearchScorer(const transformers::IGenerationParameters& parameters,
                       AllocatorPtr& allocator, AllocatorPtr& allocator_cpu,
                       Stream* stream);

  void Process(transformers::ISequences& sequences,
               gsl::span<const float>& next_scores,
               gsl::span<const int32_t>& next_tokens,
               gsl::span<const int32_t>& next_indices) override;

  void Finalize(transformers::ISequences& sequences,
                gsl::span<const float>& final_beam_scores,
                Tensor* output_sequences,
                Tensor* output_sequence_scores) override;

  bool IsDone() const override { return false; }  // For CUDA we speculatively run the next step while we wait for the GPU to report status. We use 'IsDoneLater()' for this
  bool IsDoneLater() const override;

  gsl::span<float> GetNextScores() override { return next_beam_scores_; }
  gsl::span<int32_t> GetNextTokens() override { return next_beam_tokens_; }
  gsl::span<int32_t> GetNextIndicesCPU() override {
    CUDA_CALL_THROW(cudaMemcpyAsync(next_beam_indices_cpu_.data(), next_beam_indices_.data(), next_beam_indices_.size_bytes(), cudaMemcpyDeviceToHost, stream_));
    CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
    return next_beam_indices_cpu_;
  }
  gsl::span<int32_t> GetNextIndicesGPU() override { return next_beam_indices_; }

 private:
  mutable cuda::AutoDestoryCudaEvent event_process_complete_;
  IAllocatorUniquePtr<cuda::BeamScorerState> state_cpu_;
  IAllocatorUniquePtr<cuda::BeamScorerState> state_gpu_;
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