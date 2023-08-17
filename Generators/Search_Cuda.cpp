#include "Generators.h"
#include "Search_Cuda.h"
#include "beam_search_scorer.h"
#include <queue>

namespace Generators {

#if 0
void softmax(gsl::span<float> values) {
  float max = *std::max_element(values.data(), values.data() + values.size());
  std::transform(values.begin(), values.end(), values.begin(), [max](float v) { return std::exp(v - max); });
  float sum = std::accumulate(values.begin(), values.end(), 0.0f);
  std::transform(values.begin(), values.end(), values.begin(), [sum](float v) { return v / sum; });
}

void log_softmax(gsl::span<float> values) {
  float max = *std::max_element(values.data(), values.data() + values.size());
  std::vector<float> scaled(values.begin(), values.end());
  std::transform(values.begin(), values.end(), scaled.begin(), [max](float v) { return std::exp(v - max); });

  float sum = std::accumulate(scaled.begin(), scaled.end(), 0.0f);
  float log_max = std::log(sum);
  std::transform(values.begin(), values.end(), values.begin(), [max, log_max](float v) { return v - max - log_max; });
}
#endif

void Launch_SoftMax(int32_t* next_tokens, const ScoreType* next_token_scores, int batch_size, int vocab_size, cudaStream_t stream);
void Launch_CheckForEOS(int32_t* next_tokens, int next_tokens_count, bool* eos_meet, int eos_token_id, int pad_token_id, bool* done_cpu, cudaStream_t stream);

Search_Cuda::Search_Cuda(SearchParams_Cuda& params)
    : params_{params},
      allocator_cpu_{Ort::Allocator::GetWithDefaultOptions()},
      allocator_cuda_{*params.p_allocator_cuda} {

  auto batch_beam_size = params.BatchBeamSize();

  int64_t sequences_dims[] = {batch_beam_size, params_.max_length};

  // below buffers are on cpu
  sequences_space_ = AllocateBuffer<int32_t>(&allocator_cpu_,
                                             sequences_space_buffer_,
                                             SafeInt<size_t>(2) * batch_beam_size * params_.max_length);
  memset(sequences_space_.data(), 0, sequences_space_.size_bytes());
  sequences_.Init(sequences_space_, static_cast<int>(batch_beam_size), params_.sequence_length, params_.max_length);

  sequence_lengths_ = AllocateBuffer<int32_t>(&allocator_cpu_, sequence_lengths_buffer_, batch_beam_size);
  eos_meet_ = AllocateBuffer<bool>(&allocator_cuda_, eos_meet_buffer_, batch_beam_size);
  cudaMemsetAsync(eos_meet_.data(), 0, eos_meet_.size_bytes(), params_.cuda_stream);

  // below buffers are on cpu or cuda
  size_t next_token_size = SafeInt<size_t>(batch_beam_size) * params_.vocab_size;
  next_token_scores_ = AllocateBuffer<ScoreType>(&allocator_cuda_, next_token_scores_buffer_, next_token_size);
  cudaMemsetAsync(next_token_scores_.data(), 0, next_token_scores_.size_bytes(), params_.cuda_stream);

  done_cpu_ = CudaMallocHostArray<bool>(1);

  SetInputSequence();
}

GreedySearch_Cuda::GreedySearch_Cuda(SearchParams_Cuda& params)
    : Search_Cuda{params} {
  next_tokens_ = AllocateBuffer<int32_t>(&allocator_cuda_, next_tokens_buffer_, SafeInt<size_t>(params.batch_size));
  cudaMemsetAsync(next_tokens_.data(), 0, next_tokens_.size_bytes(), params_.cuda_stream);

  next_tokens_cpu_ = CudaMallocHostArray<int32_t>(next_tokens_.size_bytes());
}

BeamSearch_Cuda::BeamSearch_Cuda(SearchParams_Cuda& params)
    : Search_Cuda{params} {
  assert(params_.num_beams > 1);  // If 1, use GreedySearch
  beam_scorer_ = std::make_unique<BeamSearchScorer>(params_, allocator_cpu_);
}

void Search_Cuda::SetInputSequence() {
  // The original inputs are not expanded, this expands them in place into the sequences
  gsl::span<int32_t> sequences_0 = sequences_space_;
  for (size_t batch = 0; batch < params_.batch_size; batch++) {
    for (size_t beam = 0; beam < params_.num_beams; beam++) {
      for (int j = 0; j < params_.sequence_length; j++) {
        sequences_0[SafeInt<gsl::index>(batch * params_.num_beams + beam) * params_.max_length + j] =
            static_cast<int32_t>(params_.input_ids[SafeInt<gsl::index>(batch) * params_.sequence_length + j]);
      }
    }
  }
}

void Search_Cuda::SetLogits(OrtValue& logits) {
  // Logits has shape (batch_size, input_length, vocab_size),
  // where input_length equals to parameters_->sequence_length for first subgraph call, and 1 for the remaining calls.
  auto logits_shape = logits.GetTensorTypeAndShapeInfo()->GetShape();
  assert(logits_shape.size() == 3);
  const ScoreType* logits_data = logits.GetTensorMutableData<ScoreType>();

  auto input_length = logits_shape[1];
  auto vocab_size = logits_shape[2];
  auto batch_beam_size = params_.BatchBeamSize();
  assert(vocab_size == params_.vocab_size);

  // Get logits for the last token:
  //    next_token_logits = logits[:, -1, :], and the result shape is (batch_size, vocab_size)
  // When input_length == 1, use logits directly in SoftmaxCPU below so it only need for input_length > 1.
  const ScoreType* current_logits = logits_data + (input_length - 1) * vocab_size;
  for (int i = 0; i < batch_beam_size; i++) {
    gsl::span<const ScoreType> source(current_logits, vocab_size);
    gsl::span<ScoreType> target = next_token_scores_.subspan(SafeInt<gsl::index>(i) * vocab_size,
                                                             static_cast<gsl::index>(vocab_size));
    cudaMemcpyAsync(target.data(), source.data(), source.size_bytes(), cudaMemcpyDeviceToDevice, params_.cuda_stream);
    current_logits += input_length * vocab_size;

//    log_softmax(target);
  }
}

#if 0
    if (do_sampling) {
      ORT_RETURN_IF_ERROR(SamplingCpuHelper::Sample(allocator,
                                                    thread_pool,
                                                    next_token_scores,
                                                    sampling_state,
                                                    greedy_state,
                                                    parameters,
                                                    dumper));
}
#endif

gsl::span<int32_t> GreedySearch_Cuda::GetNextTokens() {
  return next_tokens_;
}

gsl::span<int32_t> BeamSearch_Cuda::GetNextTokens() {
  return beam_scorer_->GetNextTokens();
}

gsl::span<int32_t> BeamSearch_Cuda::GetNextIndices() {
  return beam_scorer_->GetNextIndicesCPU();
}

int Search_Cuda::GetSequenceLength() {
  return sequences_.GetSequenceLength();
}

void BeamSearch_Cuda::NextTokensFromLogits() {
  auto beam_scores = beam_scorer_->GetNextScores();
  // Add beam score to next token scores. Corresponding python code is like:
  //    next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
  // TODO(tianleiwu): use thread pool to parallel
  int offset = 0;
  int batch_beam_index = 0;
  for (int i = 0; i < params_.batch_size; i++) {
    for (int j = 0; j < params_.num_beams; j++, batch_beam_index++) {
      for (int k = 0; k < params_.vocab_size; k++, offset++) {
        next_token_scores_[offset] += beam_scores[batch_beam_index];
      }
    }
  }

  // TODO: Write output scores?
  unsigned top_k = 2 * params_.num_beams;

  struct ScoreIndex {
    float score;
    int32_t index;
  };

  auto compare = [](const ScoreIndex& left, const ScoreIndex& right) { return left.score < right.score; };
  auto scores = std::make_unique<ScoreType[]>(top_k * params_.batch_size);
  auto indices = std::make_unique<int32_t[]>(top_k * params_.batch_size);
  auto tokens = std::make_unique<int32_t[]>(top_k * params_.batch_size);

  auto next_scores = gsl::make_span<float>(scores.get(), top_k * params_.batch_size);
  auto next_indices = gsl::make_span<int32_t>(indices.get(), top_k * params_.batch_size);
  auto next_tokens = gsl::make_span<int32_t>(tokens.get(), top_k * params_.batch_size);

  for (int batch_index = 0; batch_index < params_.batch_size; batch_index++) {
    std::priority_queue<ScoreIndex, std::vector<ScoreIndex>, decltype(compare)> queue;
    auto token_scores_sub = next_token_scores_.subspan(batch_index * params_.num_beams * params_.vocab_size, params_.num_beams * params_.vocab_size);
    for (int i = 0; i < token_scores_sub.size(); i++) {
      queue.push({token_scores_sub[i], i});
    }

    auto next_indices_sub = next_indices.subspan(top_k * batch_index, top_k);
    auto next_tokens_sub = next_tokens.subspan(top_k * batch_index, top_k);
    auto next_scores_sub = next_scores.subspan(top_k * batch_index, top_k);
    for (unsigned i = 0; i < top_k; i++) {
      auto v = queue.top();
      next_indices_sub[i] = v.index / params_.vocab_size;
      next_tokens_sub[i] = v.index % params_.vocab_size;
      next_scores_sub[i] = v.score;
      queue.pop();
    }
  }

  beam_scorer_->Process(sequences_, next_scores, next_tokens, next_indices);
  next_tokens_ = beam_scorer_->GetNextTokens();
}

void GreedySearch_Cuda::NextTokensFromLogits() {
  auto next_token_scores = next_token_scores_.data();
  Launch_SoftMax(next_tokens_.data(), next_token_scores, params_.batch_size, params_.vocab_size, params_.cuda_stream);
}

void Search_Cuda::CheckForEOS() {
  assert(next_tokens_.size()==eos_meet_.size());
  Launch_CheckForEOS(next_tokens_.data(), next_tokens_.size(), eos_meet_.data(), params_.eos_token_id, params_.pad_token_id, done_cpu_.get(), params_.cuda_stream);
}

void GreedySearch_Cuda::AppendNextTokensToSequences() {
  cudaMemcpy(next_tokens_cpu_.get(), next_tokens_.data(), next_tokens_.size_bytes(), cudaMemcpyDeviceToHost);
  sequences_.AppendNextTokenToSequences(gsl::span<const int32_t>(next_tokens_cpu_.get(), next_tokens_.size()));

  if (sequences_.GetSequenceLength() == params_.max_length)
    *done_cpu_ = true;
}

void BeamSearch_Cuda::AppendNextTokensToSequences() {
  sequences_.AppendNextTokenToSequences(beam_scorer_->GetNextIndicesCPU(), beam_scorer_->GetNextTokens());

  if (sequences_.GetSequenceLength() == params_.max_length)
    *done_cpu_ = true;
}

void BeamSearch_Cuda::Finalize(size_t num_return_sequences, gsl::span<int32_t> output, gsl::span<float> sequence_scores) {
  beam_scorer_->Finalize(sequences_, num_return_sequences, output, sequence_scores);
}

#if 0
// Not needed, for greedy can just grab the output sequence directly?
void GreedySearch::Finalize(size_t num_return_sequences, gsl::span<int32_t> output, gsl::span<float> sequence_scores) {
  auto shape=output_sequences_->GetTensorTypeAndShapeInfo()->GetShape();
  size_t shape_count = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());

  // Copy the sequences to output
  gsl::span<int32_t> output{ output_sequences_->GetTensorMutableData<int32_t>(), shape_count};
  for (int batch_id = 0; batch_id < params_.batch_size; ++batch_id) {
    auto batch_output = output.subspan(
        static_cast<size_t>(batch_id) * params_.max_length,
        params_.max_length);
    gsl::span<const int32_t> sequence_source = sequences_.GetSequence(batch_id);
    gsl::copy(sequence_source, batch_output);
  }
}
#endif

gsl::span<ScoreType> Search_Cuda::GetScores(int batch_beam_index) {
  assert(batch_beam_index >= 0 && batch_beam_index < params_.BatchBeamSize());
  return next_token_scores_.subspan(batch_beam_index * params_.vocab_size, params_.vocab_size);
}

namespace Processors_Cuda {

void MinLength(Search_Cuda& search, int min_length) {
  if (search.sequences_.GetSequenceLength() >= min_length)
    return;

  const int batch_beam_size = search.params_.BatchBeamSize();
  for (int i = 0; i < batch_beam_size; i++) {
    gsl::span<ScoreType> beam_token_scores = search.GetScores(i);
    beam_token_scores[search.params_.eos_token_id] = std::numeric_limits<ScoreType>::lowest();
  }
}

void RepetitionPenalty(Search_Cuda& search, ScoreType penalty) {
  const int batch_beam_size = search.params_.BatchBeamSize();
  for (int i = 0; i < batch_beam_size; i++) {
    gsl::span<ScoreType> beam_token_scores = search.GetScores(i);
    gsl::span<const int32_t> sequence = search.sequences_.GetSequence(i);

    // Find unique word IDs in sequence.
    std::unordered_set<int32_t> unique_word_ids;
    for (const auto& word_id : sequence) {
      unique_word_ids.insert(word_id);
    }

    for (const int32_t word_id : unique_word_ids) {
      ScoreType score = beam_token_scores[word_id];

      // If score < 0, then repetition penalty > 1.0 has to multiplied to reduce the previous token probability,
      // This assumes that scores are either positive (like ctrl) or negative (like GPT-2), but not a mixture.
      beam_token_scores[word_id] = (score < 0 ? score * penalty : score / penalty);
    }
  }
}

}  // namespace Processors

}  // namespace Generators