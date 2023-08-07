#include "Generators.h"
#include "beam_search_scorer.h"
#include <queue>

void softmax(gsl::span<float> values) {
  float max = *std::max_element(values.data(), values.data()+values.size());
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

Search::Search(Model& model, SearchParams params)
    : model_{model}, params_{params} {
  auto allocator = &Ort::Allocator::GetWithDefaultOptions();
  auto cpu_allocator = allocator;
  auto batch_beam_size=params.BatchBeamSize();

  int64_t sequences_dims[] = {batch_beam_size, params_.max_length};
 
  // below buffers are on cpu
  search_state_.sequences_space = AllocateBuffer<int32_t>(cpu_allocator,
                                                          sequences_space_buffer_,
                                                          SafeInt<size_t>(2) * batch_beam_size * params_.max_length);
  memset(search_state_.sequences_space.data(), 0, search_state_.sequences_space.size_bytes());
  sequences_.Init(search_state_.sequences_space, static_cast<int>(batch_beam_size), params_.sequence_length, params_.max_length);

  search_state_.sequence_lengths = AllocateBuffer<int32_t>(cpu_allocator, sequence_lengths_buffer_, batch_beam_size);
  search_state_.eos_meet = AllocateBuffer<bool>(cpu_allocator, eos_meet_buffer_, batch_beam_size);
  memset(search_state_.eos_meet.data(), 0, search_state_.eos_meet.size_bytes());

  search_state_.next_tokens = AllocateBuffer<int32_t>(cpu_allocator, next_tokens_buffer_, SafeInt<size_t>(batch_beam_size));

  // below buffers are on cpu or cuda
  size_t next_token_size = SafeInt<size_t>(batch_beam_size) * model.GetVocabSize();
  search_state_.next_token_scores = AllocateBuffer<ScoreType>(allocator, next_token_scores_buffer_, next_token_size);
  search_state_.next_positions = AllocateBuffer<int32_t>(allocator, next_positions_buffer_, batch_beam_size);
  int64_t position_shape[] = {batch_beam_size, 1};
  position_ids_ = OrtValue::CreateTensor<int32_t>(allocator->GetInfo(), search_state_.next_positions.data(), search_state_.next_positions.size(), position_shape, std::size(position_shape));

  model_.CreateInputs(search_state_.sequence_lengths);

  {
    auto shape = model.GetInputIds().GetTensorTypeAndShapeInfo()->GetShape();
    size_t shape_elements = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());

    gsl::span<const int32_t> input_ids{model.GetInputIds().GetTensorMutableData<int32_t>(), shape_elements};
    SetSequence(input_ids);
  }

  memset(search_state_.next_token_scores.data(), 0, search_state_.next_token_scores.size_bytes());
  memset(search_state_.next_tokens.data(), 0, search_state_.next_tokens.size_bytes());
  memset(search_state_.next_positions.data(), 0, search_state_.next_positions.size_bytes());

  gsl::copy(search_state_.sequence_lengths, search_state_.next_positions);

  if (params_.num_beams > 1) {
    beam_scorer_ = std::make_unique<BeamSearchScorer>(params_, *allocator);
  }
}

void Search::SetSequence(gsl::span<const int32_t> input_ids_in_cpu) {
  auto batch_beam_size = params_.BatchBeamSize();
  gsl::span<int32_t> sequences_0 = search_state_.sequences_space;
  for (size_t i = 0; i < batch_beam_size; i++) {
    for (int j = 0; j < params_.sequence_length; j++) {
      sequences_0[SafeInt<gsl::index>(i) * params_.max_length + j] =
          static_cast<int32_t>(input_ids_in_cpu[SafeInt<gsl::index>(i) * params_.sequence_length + j]);
    }
  }
}

void Search::RunModel() {
  if(first_run_)
    first_run_=false;
  else
  {
    if (params_.num_beams>1) {
      model_.UpdateInputs(beam_scorer_->GetNextTokens(), *position_ids_, beam_scorer_->GetNextIndicesCPU(), sequences_.GetSequenceLength());
    }
    else {
      model_.UpdateInputs(search_state_.next_tokens, *position_ids_, {}, sequences_.GetSequenceLength());
    }
  }
  model_.Run();

  // Logits has shape (batch_size, input_length, vocab_size),
  // where input_length equals to parameters_->sequence_length for first subgraph call, and 1 for the remaining calls.
  auto logits_shape = model_.GetLogits().GetTensorTypeAndShapeInfo()->GetShape();
  assert(logits_shape.size() == 3);
  const ScoreType* logits_data = model_.GetLogits().GetTensorMutableData<ScoreType>();

  auto input_length = logits_shape[1];
  auto vocab_size = logits_shape[2];
  auto batch_beam_size = params_.BatchBeamSize();
  assert(vocab_size == vocab_size_);

  // Get logits for the last token:
  //    next_token_logits = logits[:, -1, :], and the result shape is (batch_size, vocab_size)
  // When input_length == 1, use logits directly in SoftmaxCPU below so it only need for input_length > 1.
  gsl::span<ScoreType> next_token_scores = search_state_.next_token_scores;
  const ScoreType* current_logits = logits_data + (input_length - 1) * vocab_size;
  for (int i = 0; i < batch_beam_size; i++) {
    gsl::span<const ScoreType> source(current_logits, vocab_size);
    gsl::span<ScoreType> target = next_token_scores.subspan(SafeInt<gsl::index>(i) * vocab_size,
                                                            static_cast<gsl::index>(vocab_size));
    gsl::copy(source, target);
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

void Search::NextTokensFromLogits() {

  if (beam_scorer_) {
    auto beam_scores = beam_scorer_->GetNextScores();
    // Add beam score to next token scores. Corresponding python code is like:
    //    next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
    // TODO(tianleiwu): use thread pool to parallel
    int offset = 0;
    int batch_beam_index = 0;
    for (int i = 0; i < params_.batch_size; i++) {
      for (int j = 0; j < params_.num_beams; j++, batch_beam_index++) {
        for (int k = 0; k < vocab_size_; k++, offset++) {
          search_state_.next_token_scores[offset] += beam_scores[batch_beam_index];
        }
      }
    }

    // TODO: Write output scores?
    unsigned top_k=2*params_.num_beams;

    struct ScoreIndex {
      float score;
      int32_t index;
    };

    
    auto compare = [](const ScoreIndex &left, const ScoreIndex &right) { return left.score<right.score; };
    auto scores = std::make_unique<ScoreType[]>(top_k * params_.batch_size);
    auto indices = std::make_unique<int32_t[]>(top_k * params_.batch_size);
    auto tokens = std::make_unique<int32_t[]>(top_k * params_.batch_size);

    auto next_scores = gsl::make_span<float>(scores.get(), top_k * params_.batch_size);
    auto next_indices = gsl::make_span<int32_t>(indices.get(), top_k * params_.batch_size);
    auto next_tokens = gsl::make_span<int32_t>(tokens.get(), top_k * params_.batch_size);

    for (int batch_index = 0; batch_index < params_.batch_size; batch_index++) {
      std::priority_queue<ScoreIndex, std::vector<ScoreIndex>, decltype(compare)> queue;
      auto token_scores_sub = search_state_.next_token_scores.subspan(batch_index * params_.num_beams * vocab_size_, params_.num_beams * vocab_size_);
      for (int i = 0; i < token_scores_sub.size(); i++) {
        queue.push({token_scores_sub[i], i});
      }

      auto next_indices_sub = next_indices.subspan(top_k * batch_index, top_k);
      auto next_tokens_sub = next_tokens.subspan(top_k * batch_index, top_k);
      auto next_scores_sub = next_scores.subspan(top_k * batch_index, top_k);
      for (unsigned i=0;i<top_k;i++) {
        auto v=queue.top();
        next_indices_sub[i] = v.index / vocab_size_;
        next_tokens_sub[i] = v.index % vocab_size_;
        next_scores_sub[i]=v.score;
        queue.pop();
      }
    }

    beam_scorer_->Process(sequences_, next_scores, next_tokens, next_indices);
    search_state_.next_tokens = beam_scorer_->GetNextTokens();
  }
  else {
    auto next_token_scores = search_state_.next_token_scores.data();
    // next_tokens = torch.argmax(scores, dim=-1)
    for (size_t i = 0; i < params_.batch_size; i++) {
      int32_t best_token = 0;
      ScoreType best_score = next_token_scores[0];
      for (int32_t token = 1; token < vocab_size_; token++) {
        if (next_token_scores[token] > best_score) {
          best_score = next_token_scores[token];
          best_token = token;
        }
      }
      search_state_.next_tokens[i] = best_token;
      next_token_scores += vocab_size_;
    }
  }
}

void Search::CheckForEOS() {

  // Look for EOS tokens, if seen set EOS flag and replace with pad token
  gsl::span<int32_t> next_tokens = search_state_.next_tokens;
  gsl::span<bool> eos_meet = search_state_.eos_meet;
  for (size_t batch_id = 0; batch_id < next_tokens.size(); ++batch_id) {
    if (next_tokens[batch_id] == params_.eos_token_id || eos_meet[batch_id] == true) {
      eos_meet[batch_id] = true;
      next_tokens[batch_id] = params_.pad_token_id;
    }
  }

  // When all batches are finished, stop earlier to avoid wasting computation.
  // TODO: Merge this with the above so we don't have to double scan. Just keep track of 'batches left'
  {
    gsl::span<bool> eos_meet = search_state_.eos_meet;
    size_t batch_id = 0;
    while (batch_id < eos_meet.size()) {
      if (eos_meet[batch_id] == false) {
        break;
      }
      ++batch_id;
    }
    if (batch_id == eos_meet.size()) {
      done_ = true;
      return;
    }
  }
}

void Search::AppendNextTokensToSequences() {
  if (beam_scorer_) {
    sequences_.AppendNextTokenToSequences(beam_scorer_->GetNextIndicesCPU(), beam_scorer_->GetNextTokens());
  } else {
    sequences_.AppendNextTokenToSequences(search_state_.next_tokens);
  }
  if (sequences_.GetSequenceLength() == params_.max_length)
    done_ = true;
}

void Search::Finalize(size_t num_return_sequences, gsl::span<int32_t> output, gsl::span<float> sequence_scores) {
  if (beam_scorer_)
    beam_scorer_->Finalize(sequences_, num_return_sequences, output, sequence_scores);
  else {
    assert(false); // Not needed, for greedy can just grab the output sequence directly?
#if 0
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
#endif
  }
}

gsl::span<ScoreType> Search::GetScores(int batch_beam_index) {
  assert(batch_beam_index >= 0 && batch_beam_index < params_.BatchBeamSize());
  return search_state_.next_token_scores.subspan(batch_beam_index * vocab_size_, vocab_size_);
}

namespace Processors {

void MinLength(Search &search, int min_length) {
  if (search.sequences_.GetSequenceLength() >= min_length)
    return;

  const int batch_beam_size = search.params_.BatchBeamSize();
  for (int i = 0; i < batch_beam_size; i++) {
    gsl::span<ScoreType> beam_token_scores = search.GetScores(i);
    beam_token_scores[search.params_.eos_token_id] = std::numeric_limits<ScoreType>::lowest();
  }
}

void RepetitionPenalty(Search& search, ScoreType penalty) {
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

}