#include "Generators.h"
#include "Search_Cuda.h"
#include "beam_search_scorer_cuda.cuh"
#include "beam_search_scorer_cuda.h"

namespace Generators {

BeamSearchScorer_Cuda::BeamSearchScorer_Cuda(const SearchParams_Cuda& parameters,
                                           OrtAllocator& allocator_cpu, OrtAllocator& allocator_cuda)
    : stream_{parameters.cuda_stream} {

  state_cpu_ = CudaMallocHostArray<cuda::BeamScorerState>(1);
  state_cpu_->batch_size_ = static_cast<size_t>(parameters.batch_size);
  state_cpu_->num_beams_ = static_cast<size_t>(parameters.num_beams);
  state_cpu_->max_length_ = static_cast<size_t>(parameters.max_length);
  state_cpu_->pad_token_id_ = parameters.pad_token_id;
  state_cpu_->eos_token_id_ = parameters.eos_token_id;
  state_cpu_->early_stopping_ = parameters.early_stopping;
  state_cpu_->not_done_count_ = parameters.batch_size;
  state_cpu_->hypothesis_buffer_used_ = 0;
  state_gpu_ = CudaMallocArray<cuda::BeamScorerState>(1);
  cudaMemcpyAsync(state_gpu_.get(), state_cpu_.get(), sizeof(cuda::BeamScorerState), ::cudaMemcpyHostToDevice, stream_);

  size_t batch_beam_size = state_cpu_->batch_size_ * state_cpu_->num_beams_;

  auto beams = Allocate<cuda::HypothesisScore>(allocator_cuda, batch_beam_size, hypothesis_scores_ptr_);
  beam_hyps_ = Allocate<cuda::BeamHypotheses>(allocator_cuda, state_cpu_->batch_size_, beam_hyps_ptr_);

  cuda::LaunchInitializeBeamHypotheses(beam_hyps_, parameters.length_penalty, beams, parameters.num_beams, stream_);

  next_beam_scores_ = Allocate<float>(allocator_cuda, batch_beam_size, next_beam_scores_ptr_);
  next_beam_tokens_ = Allocate<int32_t>(allocator_cuda, batch_beam_size, next_beam_tokens_ptr_);
  next_beam_indices_ = Allocate<int32_t>(allocator_cuda, batch_beam_size, next_beam_indices_ptr_);
  next_beam_indices_cpu_ = Allocate<int32_t>(allocator_cpu, batch_beam_size, next_beam_indices_cpu_ptr_);

  cuda::LaunchInitScoresKernel(next_beam_scores_.data(), parameters.batch_size, parameters.num_beams, stream_);

  // Space to store intermediate sequence with length sequence_length, sequence_length + 1, ..., max_sequence_length.
  size_t per_beam = (SafeInt<size_t>(state_cpu_->max_length_) * (state_cpu_->max_length_ + 1) - (parameters.sequence_length - 1) * parameters.sequence_length) / 2;
  hypothesis_buffer_ = Allocate<int32_t>(allocator_cuda, batch_beam_size * per_beam, hypothesis_buffer_ptr_);
}

void BeamSearchScorer_Cuda::Process(Sequences& sequences,
                                    std::span<const float> next_scores,
                                    std::span<const int32_t> next_tokens,
                                    std::span<const int32_t> next_indices) {
  cuda::LaunchBeamSearchScorer_Process(*state_cpu_,
                                       *state_gpu_,
                                       sequences.GetCurrentDeviceSequences(),
                                       sequences.GetSequenceLength(),
                                       beam_hyps_,
                                       next_beam_scores_,
                                       next_beam_tokens_,
                                       next_beam_indices_,
                                       hypothesis_buffer_,
                                       next_scores,
                                       next_tokens,
                                       next_indices,
                                       stream_);
  cudaEventRecord(event_process_complete_, stream_);

  cuda::LaunchBeamSearchScorer_AppendNextTokenToSequences(*state_cpu_,
                                                          *state_gpu_,
                                                          sequences.GetCurrentDeviceSequences(),
                                                          sequences.GetNextDeviceSequences(),
                                                          sequences.GetSequenceLength(),
                                                          next_beam_tokens_,
                                                          next_beam_indices_,
                                                          stream_);
}

bool BeamSearchScorer_Cuda::IsDoneLater() const {
  cudaEventSynchronize(event_process_complete_);
  return state_cpu_->not_done_count_ == 0;
}

void BeamSearchScorer_Cuda::Finalize(Sequences& sequences,
                                    size_t num_return_sequences,
                                     std::span<int32_t> output,           // Word IDs of each sequence, with shape (batch_size * num_return_sequences, max_sequence_length)
                                     std::span<float> sequence_scores) {  // Score of each sequence, with shape (batch_size * num_return_sequences).
  assert(!output.empty());
  cuda::LaunchBeamSearchScorer_Finalize(state_cpu_->batch_size_, *state_gpu_, sequences.GetCurrentDeviceSequences(), sequences.GetSequenceLength(), beam_hyps_, next_beam_scores_, output, sequence_scores, stream_);
}

}
