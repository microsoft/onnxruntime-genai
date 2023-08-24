#include "Generators.h"
#include "beam_search_scorer.h"
#include <queue>

namespace Generators {

void softmax(std::span<float> values) {
  float max = *std::max_element(values.data(), values.data() + values.size());
  std::transform(values.begin(), values.end(), values.begin(), [max](float v) { return std::exp(v - max); });
  float sum = std::accumulate(values.begin(), values.end(), 0.0f);
  std::transform(values.begin(), values.end(), values.begin(), [sum](float v) { return v / sum; });
}

void log_softmax(std::span<float> values) {
  float max = *std::max_element(values.data(), values.data() + values.size());
  std::vector<float> scaled(values.begin(), values.end());
  std::transform(values.begin(), values.end(), scaled.begin(), [max](float v) { return std::exp(v - max); });

  float sum = std::accumulate(scaled.begin(), scaled.end(), 0.0f);
  float log_max = std::log(sum);
  std::transform(values.begin(), values.end(), values.begin(), [max, log_max](float v) { return v - max - log_max; });
}

Search::Search(SearchParams params)
    : params_{params} {
  auto allocator = &Ort::Allocator::GetWithDefaultOptions();
  auto cpu_allocator = allocator;
  auto batch_beam_size = params.BatchBeamSize();

  int64_t sequences_dims[] = {batch_beam_size, params_.max_length};

  // below buffers are on cpu
  sequences_space_ = AllocateBuffer<int32_t>(cpu_allocator,
                                             sequences_space_buffer_,
                                             2 * batch_beam_size * params_.max_length);
  memset(sequences_space_.data(), 0, sequences_space_.size_bytes());
  sequences_.Init(sequences_space_, static_cast<int>(batch_beam_size), params_.sequence_length, params_.max_length);

  sequence_lengths_ = AllocateBuffer<int32_t>(cpu_allocator, sequence_lengths_buffer_, batch_beam_size);
  eos_meet_ = AllocateBuffer<bool>(cpu_allocator, eos_meet_buffer_, batch_beam_size);
  memset(eos_meet_.data(), 0, eos_meet_.size_bytes());

  // below buffers are on cpu or cuda
  size_t next_token_size = batch_beam_size * params_.vocab_size;
  next_token_scores_ = AllocateBuffer<ScoreType>(allocator, next_token_scores_buffer_, next_token_size);
  memset(next_token_scores_.data(), 0, next_token_scores_.size_bytes());

  SetInputSequence();
}

GreedySearch::GreedySearch(SearchParams params)
    : Search(params) {
  auto allocator = &Ort::Allocator::GetWithDefaultOptions();
  next_tokens_ = AllocateBuffer<int32_t>(allocator, next_tokens_buffer_, params.batch_size);
  memset(next_tokens_.data(), 0, next_tokens_.size_bytes());
}

BeamSearch::BeamSearch(SearchParams params)
    : Search(params) {
  assert(params_.num_beams > 1);  // If 1, use GreedySearch
  auto allocator = &Ort::Allocator::GetWithDefaultOptions();
  beam_scorer_ = std::make_unique<BeamSearchScorer>(params_, *allocator);
}

void Search::SetInputSequence() {
  // The original inputs are not expanded, this expands them in place into the sequences
  std::span<int32_t> sequences_0 = sequences_space_;
  for (size_t batch = 0; batch < params_.batch_size; batch++) {
    for (size_t beam = 0; beam < params_.num_beams; beam++) {
      for (int j = 0; j < params_.sequence_length; j++) {
        sequences_0[(batch * params_.num_beams + beam) * params_.max_length + j] =
            static_cast<int32_t>(params_.input_ids[batch * params_.sequence_length + j]);
      }
    }
  }
}

void Search::SetLogits(OrtValue& logits) {
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
    std::span<const ScoreType> source(current_logits, vocab_size);
    std::span<ScoreType> target = next_token_scores_.subspan(i * vocab_size, vocab_size);
    copy(source, target);
    current_logits += input_length * vocab_size;

    log_softmax(target);
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

std::span<int32_t> GreedySearch::GetNextTokens() {
  return next_tokens_;
}

std::span<int32_t> BeamSearch::GetNextTokens() {
  return beam_scorer_->GetNextTokens();
}

std::span<int32_t> BeamSearch::GetNextIndices() {
  return beam_scorer_->GetNextIndicesCPU();
}

int Search::GetSequenceLength() {
  return sequences_.GetSequenceLength();
}

void BeamSearch::NextTokensFromLogits() {
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

  auto next_scores = std::span<float>(scores.get(), top_k * params_.batch_size);
  auto next_indices = std::span<int32_t>(indices.get(), top_k * params_.batch_size);
  auto next_tokens = std::span<int32_t>(tokens.get(), top_k * params_.batch_size);

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

#if 0
  DumpMemory("Next Scores", next_scores);
  DumpMemory("Next Tokens", next_tokens);
  DumpMemory("Next Indices", next_indices);
#endif

  beam_scorer_->Process(sequences_, next_scores, next_tokens, next_indices);
  next_tokens_ = beam_scorer_->GetNextTokens();
}

void GreedySearch::NextTokensFromLogits() {
  auto next_token_scores = next_token_scores_.data();
  // next_tokens = torch.argmax(scores, dim=-1)
  for (size_t i = 0; i < params_.batch_size; i++) {
    int32_t best_token = 0;
    ScoreType best_score = next_token_scores[0];
    for (int32_t token = 1; token < params_.vocab_size; token++) {
      if (next_token_scores[token] > best_score) {
        best_score = next_token_scores[token];
        best_token = token;
      }
    }
    next_tokens_[i] = best_token;
    next_token_scores += params_.vocab_size;
  }
}

void Search::CheckForEOS() {
  // Look for EOS tokens, if seen set EOS flag and replace with pad token
  for (size_t batch_id = 0; batch_id < next_tokens_.size(); ++batch_id) {
    if (next_tokens_[batch_id] == params_.eos_token_id || eos_meet_[batch_id] == true) {
      eos_meet_[batch_id] = true;
      next_tokens_[batch_id] = params_.pad_token_id;
    }
  }

  // When all batches are finished, stop earlier to avoid wasting computation.
  // TODO: Merge this with the above so we don't have to double scan. Just keep track of 'batches left'
  {
    size_t batch_id = 0;
    while (batch_id < eos_meet_.size()) {
      if (eos_meet_[batch_id] == false) {
        break;
      }
      ++batch_id;
    }
    if (batch_id == eos_meet_.size()) {
      done_ = true;
      return;
    }
  }
}

void GreedySearch::AppendNextTokensToSequences() {
  sequences_.AppendNextTokenToSequences(next_tokens_);

  if (sequences_.GetSequenceLength() == params_.max_length)
    done_ = true;
}

void BeamSearch::AppendNextTokensToSequences() {
  sequences_.AppendNextTokenToSequences(beam_scorer_->GetNextIndicesCPU(), beam_scorer_->GetNextTokens());

  if (sequences_.GetSequenceLength() == params_.max_length)
    done_ = true;
}

void BeamSearch::Finalize(size_t num_return_sequences, std::span<int32_t> output, std::span<float> sequence_scores) {
  beam_scorer_->Finalize(sequences_, num_return_sequences, output, sequence_scores);
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

std::span<ScoreType> Search::GetScores(int batch_beam_index) {
  assert(batch_beam_index >= 0 && batch_beam_index < params_.BatchBeamSize());
  return next_token_scores_.subspan(batch_beam_index * params_.vocab_size, params_.vocab_size);
}

namespace Processors {

void MinLength(Search& search, int min_length) {
  if (search.sequences_.GetSequenceLength() >= min_length)
    return;

  const int batch_beam_size = search.params_.BatchBeamSize();
  for (int i = 0; i < batch_beam_size; i++) {
    std::span<ScoreType> beam_token_scores = search.GetScores(i);
    beam_token_scores[search.params_.eos_token_id] = std::numeric_limits<ScoreType>::lowest();
  }
}

void RepetitionPenalty(Search& search, ScoreType penalty) {
  const int batch_beam_size = search.params_.BatchBeamSize();
  for (int i = 0; i < batch_beam_size; i++) {
    std::span<ScoreType> beam_token_scores = search.GetScores(i);
    std::span<const int32_t> sequence = search.sequences_.GetSequence(i);

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

} // namespace Generators