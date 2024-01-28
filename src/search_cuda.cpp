#include "generators.h"
#include "search_cuda.h"
#include "beam_search_scorer_cuda.cuh"
#include "beam_search_scorer_cuda.h"
#include "beam_search_topk.h"
#include <iostream>
#include <queue>
#include <random>

#include "top_p.cuh"

namespace Generators {

void OnCudaError(cudaError_t error) {
  printf("Cuda Error: %s\n", cudaGetErrorString(error));
  assert(false);
  throw std::exception();
}

Search_Cuda::Search_Cuda(const SearchParams& params)
    : params_{params},
      sequences_{params.input_ids, params.batch_size, params.num_beams, params_.max_length, params_.cuda_stream} {
  auto batch_beam_size = params.BatchBeamSize();
  sequence_lengths_buffer_ = std::make_unique<int32_t[]>(batch_beam_size);
  sequence_lengths_ = cpu_span<int32_t>(sequence_lengths_buffer_.get(), batch_beam_size);

  eos_meet_buffer_ = CudaMallocArray<bool>(batch_beam_size, &eos_meet_);
  cudaMemsetAsync(eos_meet_.data(), 0, eos_meet_.size_bytes(), params_.cuda_stream);

  // below buffers are on cpu or cuda
  size_t next_token_size = batch_beam_size * params_.vocab_size;
  next_token_scores_buffer_ = CudaMallocArray<float>(next_token_size, &next_token_scores_);
  cudaMemsetAsync(next_token_scores_.data(), 0, next_token_scores_.size_bytes(), params_.cuda_stream);

  done_cpu_ = CudaMallocHostArray<bool>(1);
  *done_cpu_ = false;
}

GreedySearch_Cuda::GreedySearch_Cuda(const SearchParams& params)
    : Search_Cuda{params} {
  next_tokens_buffer_ = CudaMallocArray<int32_t>(params.batch_size, &next_tokens_);
  cudaMemsetAsync(next_tokens_.data(), 0, next_tokens_.size_bytes(), params_.cuda_stream);
}

BeamSearch_Cuda::BeamSearch_Cuda(const SearchParams& params)
    : Search_Cuda{params} {
  assert(params_.num_beams > 1);  // If 1, use GreedySearch
  auto batch_beam_size = params_.BatchBeamSize();
  beam_scorer_ = std::make_unique<BeamSearchScorer_Cuda>(params_);

  topk_next_tokens_ = CudaMallocArray<int32_t>(2 * batch_beam_size);
  topk_next_indices_ = CudaMallocArray<int32_t>(2 * batch_beam_size);
  topk_next_scores_ = CudaMallocArray<float>(2 * batch_beam_size);

  constexpr size_t max_parts_of_vocab = 128;
  size_t topk_buffer_size = batch_beam_size * (max_parts_of_vocab + 1) * params_.num_beams * 2 * 2;
  topk_buffer_ = CudaMallocArray<float>(topk_buffer_size);
  static_assert(sizeof(float) == sizeof(int32_t));  // The topk_buffer assumes these match, fix for float16

  cudaMemsetAsync(topk_buffer_.get(), 0, topk_buffer_size * sizeof(float), params_.cuda_stream);
}

BeamSearch_Cuda::~BeamSearch_Cuda() = default;

void Search_Cuda::SetLogits(RoamingArray<float> logits_unk) {
  gpu_span<float> logits = logits_unk;
  // Logits has shape (batch_size, input_length, vocab_size),
  // where input_length equals to parameters_->sequence_length for first subgraph call, and 1 for the remaining calls.
  // RyanHill: Does it really? The output of gpt2 is always a input_length of 1, regardless of input sequence length

  auto batch_beam_size = params_.BatchBeamSize();
  auto input_length = logits.size() / (batch_beam_size * params_.vocab_size);
  assert(logits.size() % (batch_beam_size * params_.vocab_size) == 0);  // Should divide evenly

  // TODO: if input_length==1, use token scores directly

  // Get logits for the last token:
  //    next_token_logits = logits[:, -1, :], and the result shape is (batch_size, vocab_size)
  // When input_length == 1, use logits directly in SoftmaxCPU below so it only need for input_length > 1.
  const float* current_logits = logits.data() + (input_length - 1) * params_.vocab_size;
  for (int i = 0; i < batch_beam_size; i++) {
    std::span<const float> source(current_logits, params_.vocab_size);
    std::span<float> target = next_token_scores_.subspan(i * params_.vocab_size, params_.vocab_size);
    CudaCheck() == cudaMemcpyAsync(target.data(), source.data(), source.size_bytes(), cudaMemcpyDeviceToDevice, params_.cuda_stream);
    current_logits += input_length * params_.vocab_size;

    cuda::Launch_log_softmax(target.data(), static_cast<int>(target.size()), params_.cuda_stream);
  }

  // float* cpu_logits = new float[params_.batch_size * params_.vocab_size];
  // cudaStreamSynchronize(params_.cuda_stream);
  // cudaMemcpy(cpu_logits, next_token_scores_.data(), params_.batch_size * params_.vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
  // for (int i = 0; i < params_.batch_size; i++) {
  //   std::cout << "Batch " << i << "\r\n";
  //   for (int j = 0; j < params_.vocab_size; j++) {
  //     std::cout << cpu_logits[i * params_.vocab_size + j] << " ";
  //   }
  //   std::cout << "\r\n";
  // }
}

RoamingArray<int32_t> GreedySearch_Cuda::GetNextTokens() {
  return next_tokens_;
}

RoamingArray<int32_t> BeamSearch_Cuda::GetNextTokens() {
  return beam_scorer_->GetNextTokens();
}

RoamingArray<int32_t> BeamSearch_Cuda::GetNextIndices() {
  return beam_scorer_->GetNextIndicesCPU();
}

int Search_Cuda::GetSequenceLength() const {
  return sequences_.GetSequenceLength();
}

void BeamSearch_Cuda::SelectTop() {
  auto beam_scores = beam_scorer_->GetNextScores();

  // Add beam score to next token scores. Corresponding python code is like:
  //    next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
  cuda::LaunchAddProbsKernel(next_token_scores_.data(), beam_scores.data(),
                             params_.batch_size, params_.num_beams, params_.vocab_size, params_.cuda_stream);

  // TODO: Write output scores?

  if (params_.num_beams <= 32) {
    constexpr size_t max_parts_of_vocab = 128;
    size_t candidate_count = params_.BatchBeamSize() * 2 * params_.num_beams;
    float* topk_tmp_buffer = topk_buffer_.get();
    float* topk_scores_1st_stage = topk_tmp_buffer;
    int32_t* topk_tokens_1st_stage = reinterpret_cast<int32_t*>(topk_scores_1st_stage + candidate_count * max_parts_of_vocab);
    float* topk_scores_2nd_stage = reinterpret_cast<float*>(topk_tokens_1st_stage + candidate_count * max_parts_of_vocab);
    int32_t* topk_tokens_2nd_stage = reinterpret_cast<int32_t*>(topk_scores_2nd_stage + candidate_count);

    cuda::BeamSearchTopK(next_token_scores_.data(),
                         params_.batch_size,
                         params_.num_beams,
                         params_.vocab_size,
                         2 * params_.num_beams,
                         topk_scores_1st_stage,
                         topk_tokens_1st_stage,
                         topk_scores_2nd_stage,
                         topk_tokens_2nd_stage,
                         topk_next_scores_.get(),
                         topk_next_tokens_.get(),
                         topk_next_indices_.get(),
                         params_.cuda_stream);
  } else
    assert(false);

  CudaCheck() == cudaStreamSynchronize(params_.cuda_stream);

  size_t size = params_.BatchBeamSize() * 2;
  std::span<float> next_scores{topk_next_scores_.get(), size};
  std::span<int32_t> next_tokens{topk_next_tokens_.get(), size};
  std::span<int32_t> next_indices{topk_next_indices_.get(), size};

#if 0
  DumpCudaMemory("Next Scores", next_scores);
  DumpCudaMemory("Next Tokens", next_tokens);
  DumpCudaMemory("Next Indices", next_indices);
#endif

  beam_scorer_->Process(sequences_, next_scores, next_tokens, next_indices);
  next_tokens_ = beam_scorer_->GetNextTokens();

  AppendNextTokensToSequences();
}

void GreedySearch_Cuda::SelectTop() {
  auto next_token_scores = next_token_scores_.data();
  cuda::Launch_ArgMax(argmaxdata_, next_tokens_.data(), next_token_scores, params_.batch_size, params_.vocab_size, params_.cuda_stream);

  CheckForEOS();
  AppendNextTokensToSequences();
}

// TODO: commented out in case of benchmarking
// void SoftMax(std::span<float> scores, float temperature);
// void TopPSampling(int32_t* d_next_token, float* d_scores, int size, float threshold, float temperature) {
//   auto scores_buffer = CudaMallocHostArray<float>(size);
//   std::span<float> scores{scores_buffer.get(), static_cast<size_t>(size)};
//   cudaMemcpy(scores.data(), d_scores, size * sizeof(float), cudaMemcpyDeviceToHost);

//   SoftMax(scores, temperature);

//   // Sort an array of indices by scores
//   std::vector<int32_t> indices(scores.size());
//   std::iota(indices.begin(), indices.end(), 0);
//   std::sort(indices.begin(), indices.end(), [scores = scores.data()](int32_t i, int32_t j) { return scores[i] > scores[j]; });

//   int32_t token = 0;
//   // Find the first token where the cumulative probability exceeds the threshold
//   for (int i = 0; i < scores.size(); i++) {
//     threshold -= scores[indices[i]];
//     if (threshold > 0)
//       continue;

//     token = indices[i];
//     break;
//   }

//   cudaMemcpy(d_next_token, &token, sizeof(token), cudaMemcpyHostToDevice);
// }

void GreedySearch_Cuda::SampleTopP(float p, float temperature) {
  // TODO: commented out in case of benchmarking
  // std::random_device rd;
  // std::mt19937 gen(rd());
  // std::uniform_real_distribution<float> dis(0, p);
  // for (int i = 0; i < params_.batch_size; i++) {
  //   std::span<float> scores = next_token_scores_.subspan(i * params_.vocab_size, params_.vocab_size);
  //   TopPSampling(next_tokens_.data() + i, scores.data(), static_cast<int>(scores.size()), dis(gen), temperature);
  // }

  std::span<float> scores = next_token_scores_.subspan(0, params_.batch_size * params_.vocab_size);
  cuda::SampleTopPKernel(next_tokens_.data(), scores.data(), int(scores.size() / params_.batch_size), params_.batch_size, p, temperature, params_.cuda_stream);

  CheckForEOS();
  AppendNextTokensToSequences();
}

void GreedySearch_Cuda::CheckForEOS() {
  assert(next_tokens_.size() == eos_meet_.size());
  cuda::Launch_CheckForEOS(next_tokens_.data(), static_cast<int>(next_tokens_.size()), eos_meet_.data(), params_.eos_token_id, params_.pad_token_id, done_cpu_.get(), params_.cuda_stream);
}

void GreedySearch_Cuda::AppendNextTokensToSequences() {
  sequences_.AppendNextTokenToSequences(next_tokens_);

  if (sequences_.GetSequenceLength() == params_.max_length)
    *done_cpu_ = true;
}

bool BeamSearch_Cuda::IsDone() const {
  beam_scorer_->IsDone();
  return beam_scorer_->IsDoneLater() || sequences_.GetSequenceLength() == params_.max_length;
}

void BeamSearch_Cuda::AppendNextTokensToSequences() {
  sequences_.AfterDeviceAppendedNextToken();
}

void BeamSearch_Cuda::Finalize(size_t num_return_sequences, RoamingArray<int32_t> output, RoamingArray<float> sequence_scores) {
  beam_scorer_->Finalize(sequences_, num_return_sequences, output.GetGPU(), sequence_scores.GetGPU());
}

#if 0
// Not needed, for greedy can just grab the output sequence directly?
void GreedySearch::Finalize(size_t num_return_sequences, std::span<int32_t> output, std::span<float> sequence_scores) {
  auto shape=output_sequences_->GetTensorTypeAndShapeInfo()->GetShape();
  size_t shape_count = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());

  // Copy the sequences to output
  std::span<int32_t> output{ output_sequences_->GetTensorMutableData<int32_t>(), shape_count};
  for (int batch_id = 0; batch_id < params_.batch_size; ++batch_id) {
    auto batch_output = output.subspan(
        static_cast<size_t>(batch_id) * params_.max_length,
        params_.max_length);
    std::span<const int32_t> sequence_source = sequences_.GetSequence(batch_id);
    std::copy(sequence_source, batch_output);
  }
}
#endif

std::span<float> Search_Cuda::GetScores(int batch_beam_index) {
  assert(batch_beam_index >= 0 && batch_beam_index < params_.BatchBeamSize());
  return next_token_scores_.subspan(batch_beam_index * params_.vocab_size, params_.vocab_size);
}

std::span<float> Search_Cuda::GetScores() {
  return next_token_scores_;
}

namespace Processors_Cuda {

void MinLength(Search_Cuda& search, int min_length) {
  if (search.sequences_.GetSequenceLength() >= min_length)
    return;

  const int batch_beam_size = search.params_.BatchBeamSize();
  for (int i = 0; i < batch_beam_size; i++) {
    std::span<float> beam_token_scores = search.GetScores(i);
    beam_token_scores[search.params_.eos_token_id] = std::numeric_limits<float>::lowest();
  }
}

void RepetitionPenalty(Search_Cuda& search, float penalty) {
  cuda::LaunchRepetitionPenaltyProcessor(search.sequences_.GetSequences().data(),
                                         search.GetScores().data(), search.params_.batch_size, search.params_.num_beams, search.params_.vocab_size,
                                         search.params_.max_length, search.GetSequenceLength(), penalty, search.params_.cuda_stream);
}

}  // namespace Processors_Cuda

}  // namespace Generators