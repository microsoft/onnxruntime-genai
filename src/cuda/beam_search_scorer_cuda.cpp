#include "generators.h"
#include "search.h"
#include "search_cuda.h"
#include "beam_search_scorer_cuda.cuh"
#include "beam_search_scorer_cuda.h"
#include "interface.h"

namespace Generators {

BeamSearchScorer_Cuda::BeamSearchScorer_Cuda(const GeneratorParams& parameters)
    : stream_{parameters.cuda_stream} {
  state_cpu_ = CudaMallocHostArray<cuda::BeamScorerState>(1);
  state_cpu_->batch_size_ = static_cast<size_t>(parameters.batch_size);
  state_cpu_->num_beams_ = static_cast<size_t>(parameters.search.num_beams);
  state_cpu_->max_length_ = static_cast<size_t>(parameters.search.max_length);
  state_cpu_->pad_token_id_ = parameters.config.model.pad_token_id;
  state_cpu_->eos_token_id_ = parameters.config.model.eos_token_id;
  state_cpu_->early_stopping_ = parameters.search.early_stopping;
  state_cpu_->not_done_count_ = parameters.batch_size;
  state_cpu_->hypothesis_buffer_used_ = 0;
  state_gpu_ = CudaMallocArray<cuda::BeamScorerState>(1);
  cudaMemcpyAsync(state_gpu_.get(), state_cpu_.get(), sizeof(cuda::BeamScorerState), ::cudaMemcpyHostToDevice, stream_);

  size_t batch_beam_size = state_cpu_->batch_size_ * state_cpu_->num_beams_;

  auto& device = GetCudaDeviceInterface();

  gpu_span<cuda::HypothesisScore> beams;
  hypothesis_scores_ptr_ = CudaMallocArray<cuda::HypothesisScore>(batch_beam_size, &beams);
  beam_hyps_ptr_ = CudaMallocArray<cuda::BeamHypotheses>(state_cpu_->batch_size_, &beam_hyps_);

  cuda::LaunchInitializeBeamHypotheses(beam_hyps_, parameters.search.length_penalty, beams, parameters.search.num_beams, stream_);

  next_beam_scores_ = parameters.p_device->Allocate<float>(batch_beam_size);
  next_beam_tokens_ = parameters.p_device->Allocate<int32_t>(batch_beam_size);
  next_beam_indices_ = parameters.p_device->Allocate<int32_t>(batch_beam_size);

  cuda::LaunchInitScoresKernel(next_beam_scores_.Span().data(), parameters.batch_size, parameters.search.num_beams, stream_);

  // Space to store intermediate sequence with length sequence_length, sequence_length + 1, ..., max_sequence_length.
  size_t per_beam = (state_cpu_->max_length_ * (state_cpu_->max_length_ + 1) - (parameters.sequence_length - 1) * parameters.sequence_length) / 2;
  hypothesis_buffer_ = device.Allocate<int32_t>(batch_beam_size * per_beam);
}

void BeamSearchScorer_Cuda::Process(Sequences& sequences,
                                    std::span<const float> next_scores,
                                    std::span<const int32_t> next_tokens,
                                    std::span<const int32_t> next_indices) {
  cuda::LaunchBeamSearchScorer_Process(*state_cpu_,
                                       *state_gpu_,
                                       sequences.GetSequences().Span(),
                                       sequences.GetSequenceLength(),
                                       beam_hyps_,
                                       next_beam_scores_.Span(),
                                       next_beam_tokens_.Span(),
                                       next_beam_indices_.Span(),
                                       hypothesis_buffer_.Span(),
                                       next_scores,
                                       next_tokens,
                                       next_indices,
                                       stream_);
  cudaEventRecord(event_process_complete_, stream_);

  cuda::LaunchBeamSearchScorer_AppendNextTokenToSequences(*state_cpu_,
                                                          *state_gpu_,
                                                          sequences.GetSequences().Span(),
                                                          sequences.GetNextSequences().Span(),
                                                          sequences.GetSequenceLength(),
                                                          next_beam_tokens_.Span(),
                                                          next_beam_indices_.Span(),
                                                          stream_);
}

bool BeamSearchScorer_Cuda::IsDoneLater() const {
  cudaEventSynchronize(event_process_complete_);
  return state_cpu_->not_done_count_ == 0;
}

void BeamSearchScorer_Cuda::Finalize(Sequences& sequences,
                                     size_t num_return_sequences) {
  cuda::LaunchBeamSearchScorer_Finalize(state_cpu_->batch_size_, *state_gpu_, sequences.GetSequences().Span(), sequences.GetSequenceLength(), beam_hyps_, hypothesis_buffer_.Span(), next_beam_scores_.Span(), stream_);
}

DeviceSpan<int32_t> BeamSearchScorer_Cuda::GetBeamHypothesis(size_t batch_id, size_t beam_id) {
  cuda_host_unique_ptr<int32_t*> hypothesis_ptr = CudaMallocHostArray<int32_t*>(1);
  cuda_host_unique_ptr<int> hypothesis_length = CudaMallocHostArray<int>(1);
  cuda_host_unique_ptr<float> hypothesis_score = CudaMallocHostArray<float>(1);
  cuda::LaunchBeamSearchScorer_GetHypothesisPtr(batch_id, beam_id, beam_hyps_, hypothesis_ptr.get(), hypothesis_length.get(), hypothesis_score.get(), stream_);
  CudaCheck() == cudaStreamSynchronize(stream_);
  std::span<int32_t> hypothesis(*hypothesis_ptr.get(), *hypothesis_length.get());
  // Translate the hypothesis span back to the original device buffer span
  return hypothesis_buffer_.subspan(hypothesis.data() - hypothesis_buffer_.Span().data(), hypothesis.size());
}

}  // namespace Generators
