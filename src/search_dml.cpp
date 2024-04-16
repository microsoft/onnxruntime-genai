#include <queue>
#include <algorithm>
#include "generators.h"
#include "softmax.h"
#include "search_dml.h"
#include "beam_search_scorer.h"

namespace Generators {

Search_Dml::Search_Dml(const GeneratorParams& params, DmlExecutionContext* dml_execution_context, DmlReadbackHeap* dml_readback_heap)
    : Search{params},
      sequences_{params.input_ids, params.batch_size, params.search.num_beams, params_->search.max_length},
      dml_execution_context_(dml_execution_context),
      dml_readback_heap_(dml_readback_heap) {
  auto batch_beam_size = params.BatchBeamSize();
  sequence_lengths_buffer_ = AllocateArray<int32_t>(batch_beam_size, &sequence_lengths_);
}

GreedySearch_Dml::GreedySearch_Dml(const GeneratorParams& params, DmlExecutionContext* dml_execution_context, DmlReadbackHeap* dml_readback_heap)
    : Search_Dml(params, dml_execution_context, dml_readback_heap), gen_(rd_()) {
  next_tokens_buffer_ = AllocateArray<int32_t>(params.batch_size, &next_tokens_);
  memset(next_tokens_.data(), 0, next_tokens_.size_bytes());

  eos_seen_buffer_ = AllocateArray<bool>(params.batch_size, &eos_seen_);
  memset(eos_seen_.data(), 0, eos_seen_.size_bytes());
}

BeamSearch_Dml::BeamSearch_Dml(const GeneratorParams& params, DmlExecutionContext* dml_execution_context, DmlReadbackHeap* dml_readback_heap)
    : Search_Dml(params, dml_execution_context, dml_readback_heap) {
  assert(params_->search.num_beams > 1);  // If 1, use GreedySearch
  beam_scorer_ = std::make_unique<BeamSearchScorer>(*params_);
}

BeamSearch_Dml::~BeamSearch_Dml() = default;

void Search_Dml::SetLogits(RoamingArray<float> logits_unk) {
  // TODO (pavignol): Optimize (ideally do the computation on the GPU)
  next_token_scores_ = logits_unk.GetCPU();
}

RoamingArray<int32_t> GreedySearch_Dml::GetNextTokens() {
  return next_tokens_;
}

RoamingArray<int32_t> BeamSearch_Dml::GetNextTokens() {
  return beam_scorer_->GetNextTokens();
}

RoamingArray<int32_t> BeamSearch_Dml::GetNextIndices() {
  return beam_scorer_->GetNextIndicesCPU();
}

int Search_Dml::GetSequenceLength() const {
  return sequences_.GetSequenceLength();
}

void BeamSearch_Dml::SelectTop() {
  auto beam_scores = beam_scorer_->GetNextScores();
  // Add beam score to next token scores. Corresponding python code is like:
  //    next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
  // TODO(tianleiwu): use thread pool to parallel
  int offset = 0;
  int batch_beam_index = 0;
  for (int i = 0; i < params_->batch_size; i++) {
    for (int j = 0; j < params_->search.num_beams; j++, batch_beam_index++) {
      for (int k = 0; k < params_->vocab_size; k++, offset++) {
        next_token_scores_[offset] += beam_scores[batch_beam_index];
      }
    }
  }

  // TODO: Write output scores?
  const size_t top_k = 2 * params_->search.num_beams;

  struct ScoreIndex {
    float score;
    int32_t index;

    bool operator<(const ScoreIndex& s) const { return score < s.score; }
  };

  auto scores = std::make_unique<float[]>(top_k * params_->batch_size);
  auto indices = std::make_unique<int32_t[]>(top_k * params_->batch_size);
  auto tokens = std::make_unique<int32_t[]>(top_k * params_->batch_size);

  auto next_scores = std::span<float>(scores.get(), top_k * params_->batch_size);
  auto next_indices = std::span<int32_t>(indices.get(), top_k * params_->batch_size);
  auto next_tokens = std::span<int32_t>(tokens.get(), top_k * params_->batch_size);

  for (size_t batch_index = 0; batch_index < static_cast<size_t>(params_->batch_size); batch_index++) {
    std::priority_queue<ScoreIndex, std::vector<ScoreIndex>> queue;
    auto token_scores_sub = next_token_scores_.subspan(batch_index * params_->search.num_beams * params_->vocab_size, static_cast<size_t>(params_->search.num_beams) * params_->vocab_size);
    for (int i = 0; i < token_scores_sub.size(); i++) {
      queue.push({token_scores_sub[i], i});
    }

    auto next_indices_sub = next_indices.subspan(top_k * batch_index, top_k);
    auto next_tokens_sub = next_tokens.subspan(top_k * batch_index, top_k);
    auto next_scores_sub = next_scores.subspan(top_k * batch_index, top_k);
    for (unsigned i = 0; i < top_k; i++) {
      auto v = queue.top();
      next_indices_sub[i] = v.index / params_->vocab_size;
      next_tokens_sub[i] = v.index % params_->vocab_size;
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

  AppendNextTokensToSequences();
}

void GreedySearch_Dml::SelectTop() {
  // next_tokens = torch.argmax(scores, dim=-1)
  for (size_t batch_id = 0; batch_id < params_->batch_size; batch_id++) {
    if (PadIfAlreadyEOS(batch_id)) {
      continue;
    }

    std::span<float> const scores = next_token_scores_.subspan(batch_id * params_->vocab_size, params_->vocab_size);
    auto const token = static_cast<int32_t>(std::distance(scores.begin(), std::max_element(scores.begin(), scores.end())));
    SetNextToken(batch_id, token);
  }

  AppendNextTokensToSequences();
}

static void SoftMax(std::span<float> scores, float temperature) {
  float const max_score = *std::max_element(scores.begin(), scores.end());

  // Subtract max score and scale by temperature
  std::transform(scores.begin(), scores.end(), scores.begin(), [max_score, temperature](float score) { return std::exp((score - max_score) / temperature); });

  // Compute sum of exponentials
  float const exp_sum = std::accumulate(scores.begin(), scores.end(), 0.0f);

  // Divide each score by the sum of exponentials
  std::transform(scores.begin(), scores.end(), scores.begin(), [exp_sum](float score) { return score / exp_sum; });
}

void GreedySearch_Dml::SampleTopK(int k, float temperature) {
  for (size_t batch_id = 0; batch_id < params_->batch_size; batch_id++) {
    std::span<float> const scores = next_token_scores_.subspan(batch_id * params_->vocab_size, params_->vocab_size);
    SoftMax(scores, temperature);
    // Find the top K scores
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), [scores = scores.data()](int i, int j) { return scores[i] > scores[j]; });
    // Sample a token from the top K
    std::discrete_distribution<> dis(scores.begin(), scores.begin() + k);
    SetNextToken(batch_id, indices[dis(gen_)]);
  }
  AppendNextTokensToSequences();
}

void GreedySearch_Dml::SampleTopP(float p, float temperature) {
  std::uniform_real_distribution<float> dis(0, p);
  for (size_t batch_id = 0; batch_id < params_->batch_size; batch_id++) {
    if (PadIfAlreadyEOS(batch_id)) {
      continue;
    }
    std::span<float> const scores = next_token_scores_.subspan(batch_id * params_->vocab_size, params_->vocab_size);
    SoftMax(scores, temperature);
    // Sort an array of indices into the scores
    std::vector<int32_t> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [scores = scores.data()](int32_t i, int32_t j) { return scores[i] > scores[j]; });
    // Sample a probability threshold
    float threshold = dis(gen_);
    int32_t token = 0;
    // Find the first token where the cumulative probability exceeds the threshold
    for (int i = 0; i < scores.size(); i++) {
      threshold -= scores[indices[i]];
      if (threshold > 0) {
        continue;
      }
      token = indices[i];
      break;
    }
    SetNextToken(batch_id, token);
  }
  AppendNextTokensToSequences();
}

void GreedySearch_Dml::SampleTopKTopP(int k, float p, float temperature) {
  std::uniform_real_distribution<float> dis(0, p);
  for (size_t batch_id = 0; batch_id < params_->batch_size; batch_id++) {
    if (PadIfAlreadyEOS(batch_id)) {
      continue;
    }
    std::span<float> const scores = next_token_scores_.subspan(batch_id * params_->vocab_size, params_->vocab_size);
    SoftMax(scores, temperature);
    // Find the top K scores
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), [scores = scores.data()](int i, int j) { return scores[i] > scores[j]; });
    // Sample a probability threshold
    float threshold = dis(gen_);
    int32_t token = indices[k - 1];
    // Find the first token where the cumulative probability exceeds the threshold
    for (int i = 0; i < k; i++) {
      threshold -= scores[indices[i]];
      if (threshold > 0) {
        continue;
      }
      token = indices[i];
      break;
    }
    SetNextToken(batch_id, token);
  }
  AppendNextTokensToSequences();
}

bool GreedySearch_Dml::PadIfAlreadyEOS(size_t batch_id) {
  // If this batch entry has already seen the EOS token, append the pad token
  if (!eos_seen_[batch_id]) {
    return false;
  }

  next_tokens_[batch_id] = params_->pad_token_id;
  return true;
}

void GreedySearch_Dml::SetNextToken(size_t batch_id, int32_t token) {
  next_tokens_[batch_id] = token;
  if (token == params_->eos_token_id) {
    eos_seen_[batch_id] = true;
    if (--not_done_count_ == 0) {
      done_ = true;
    }
  }
}

void GreedySearch_Dml::AppendNextTokensToSequences() {
  sequences_.AppendNextTokenToSequences(next_tokens_);

  if (sequences_.GetSequenceLength() == params_->search.max_length) {
    done_ = true;
  }
}

void BeamSearch_Dml::AppendNextTokensToSequences() {
  sequences_.AppendNextTokenToSequences(beam_scorer_->GetNextIndicesCPU(), beam_scorer_->GetNextTokens());

  if (sequences_.GetSequenceLength() == params_->search.max_length) {
    done_ = true;
  }
}

void BeamSearch_Dml::Finalize(size_t num_return_sequences, RoamingArray<int32_t> output, RoamingArray<float> sequence_scores) {
  beam_scorer_->Finalize(sequences_, num_return_sequences, output, sequence_scores);
}

std::span<float> Search_Dml::GetScores(int batch_beam_index) const {
  assert(batch_beam_index >= 0 && batch_beam_index < params_->BatchBeamSize());
  return next_token_scores_.subspan(static_cast<size_t>(batch_beam_index) * params_->vocab_size, params_->vocab_size);
}

void Search_Dml::ApplyMinLength(int min_length) {
  if (sequences_.GetSequenceLength() >= min_length) {
    return;
  }

  const int batch_beam_size = params_->BatchBeamSize();
  for (int i = 0; i < batch_beam_size; i++) {
    std::span<float> const beam_token_scores = GetScores(i);
    beam_token_scores[params_->eos_token_id] = std::numeric_limits<float>::lowest();
  }
}

void Search_Dml::ApplyRepetitionPenalty(float penalty) {
  if (penalty == 1.0f)
    return;

  const int batch_beam_size = params_->BatchBeamSize();
  for (int i = 0; i < batch_beam_size; i++) {
    std::span<float> const beam_token_scores = GetScores(i);
    std::span<const int32_t> const sequence = sequences_.GetSequence(i);

    // Find unique word IDs in sequence.
    std::unordered_set<int32_t> unique_word_ids;
    for (const auto& word_id : sequence) {
      unique_word_ids.insert(word_id);
    }

    for (const int32_t word_id : unique_word_ids) {
      float const score = beam_token_scores[word_id];

      // If score < 0, then repetition penalty > 1.0 has to multiplied to reduce the previous token probability,
      // This assumes that scores are either positive (like ctrl) or negative (like GPT-2), but not a mixture.
      beam_token_scores[word_id] = (score < 0 ? score * penalty : score / penalty);
    }
  }
}

}  // namespace Generators