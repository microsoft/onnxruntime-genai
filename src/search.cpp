#include "generators.h"
#include "softmax.h"
#include "search.h"
#include "beam_search_scorer.h"
#include <queue>
#include <algorithm>
#include <random>

namespace Generators {

Search_Cpu::Search_Cpu(SearchParams params)
    : params_{params},
      sequences_{params.input_ids, params.batch_size, params.num_beams, params_.max_length} {

  auto batch_beam_size = params.BatchBeamSize();

  sequence_lengths_buffer_ = AllocateArray<int32_t>(batch_beam_size, &sequence_lengths_);

  size_t next_token_size = batch_beam_size * params_.vocab_size;
  next_token_scores_buffer_ = AllocateArray<ScoreType>(next_token_size, &next_token_scores_);
  memset(next_token_scores_.data(), 0, next_token_scores_.size_bytes());
}

GreedySearch_Cpu::GreedySearch_Cpu(SearchParams params)
    : Search_Cpu(params) {
  next_tokens_buffer_ = AllocateArray<int32_t>(params.batch_size, &next_tokens_);
  memset(next_tokens_.data(), 0, next_tokens_.size_bytes());

  eos_seen_buffer_ = AllocateArray<bool>(params.batch_size, &eos_seen_);
  memset(eos_seen_.data(), 0, eos_seen_.size_bytes());
}

BeamSearch_Cpu::BeamSearch_Cpu(SearchParams params)
    : Search_Cpu(params) {
  assert(params_.num_beams > 1);  // If 1, use GreedySearch
  beam_scorer_ = std::make_unique<BeamSearchScorer>(params_);
}

BeamSearch_Cpu::~BeamSearch_Cpu() = default;

void Search_Cpu::SetLogits(RoamingArray<float> logits_unk) {
  cpu_span<float> logits=logits_unk;
  // Logits has shape (batch_size, input_length, vocab_size),
  // where input_length equals to parameters_->sequence_length for first subgraph call, and 1 for the remaining calls.

  auto batch_beam_size = params_.BatchBeamSize();
  auto input_length = logits.size() / (batch_beam_size * params_.vocab_size);
  assert(logits.size() % (batch_beam_size * params_.vocab_size) == 0);  // Should divide evenly

  // TODO: if input_length==1, use token scores directly

  // Get logits for the last token:
  //    next_token_logits = logits[:, -1, :], and the result shape is (batch_size, vocab_size)
  // When input_length == 1, use logits directly in SoftmaxCPU below so it only need for input_length > 1.
  const ScoreType* current_logits = logits.data() + (input_length - 1) * params_.vocab_size;
  for (int i = 0; i < batch_beam_size; i++) {
    std::span<const ScoreType> source(current_logits, params_.vocab_size);
    std::span<ScoreType> target = next_token_scores_.subspan(i * params_.vocab_size, params_.vocab_size);
    copy(source, target);
    current_logits += input_length * params_.vocab_size;

    log_softmax(target);
  }
}

RoamingArray<int32_t> GreedySearch_Cpu::GetNextTokens() {
  return next_tokens_;
}

RoamingArray<int32_t> BeamSearch_Cpu::GetNextTokens() {
  return beam_scorer_->GetNextTokens();
}

RoamingArray<int32_t> BeamSearch_Cpu::GetNextIndices() {
  return beam_scorer_->GetNextIndicesCPU();
}

int Search_Cpu::GetSequenceLength() const {
  return sequences_.GetSequenceLength();
}

void BeamSearch_Cpu::SelectTop() {
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

    bool operator<(const ScoreIndex& s) const { return score < s.score; }
  };

  auto scores = std::make_unique<ScoreType[]>(top_k * params_.batch_size);
  auto indices = std::make_unique<int32_t[]>(top_k * params_.batch_size);
  auto tokens = std::make_unique<int32_t[]>(top_k * params_.batch_size);

  auto next_scores = std::span<float>(scores.get(), top_k * params_.batch_size);
  auto next_indices = std::span<int32_t>(indices.get(), top_k * params_.batch_size);
  auto next_tokens = std::span<int32_t>(tokens.get(), top_k * params_.batch_size);

  for (int batch_index = 0; batch_index < params_.batch_size; batch_index++) {
    std::priority_queue<ScoreIndex, std::vector<ScoreIndex>> queue;
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

  AppendNextTokensToSequences();
}

void GreedySearch_Cpu::SelectTop() {
  // next_tokens = torch.argmax(scores, dim=-1)
  for (size_t batch_id = 0; batch_id < params_.batch_size; batch_id++) {
    if (PadIfAlreadyEOS(batch_id))
      continue;

    std::span<ScoreType> scores=next_token_scores_.subspan(batch_id*params_.vocab_size, params_.vocab_size);
    int32_t token = static_cast<int32_t>(std::distance(scores.begin(), std::max_element(scores.begin(), scores.end())));
    SetNextToken(batch_id, token);
  }

  AppendNextTokensToSequences();
}

void GreedySearch_Cpu::SampleTopK(int k, float temperature) {
#if 0
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, k);

  std::vector<int32_t> top_k;
  top_k.resize(k);
  for (size_t batch_id = 0; batch_id < params_.batch_size; batch_id++) {
    std::span<ScoreType> scores = next_token_scores_.subspan(batch_id * params_.vocab_size, params_.vocab_size);

    // Apply temperature and convert log probabilities to probabilities
    std::vector<float> prob(scores.size());
    std::transform(scores.begin(), scores.end(), prob.begin(), [temperature](float logp) { return std::exp(logp / temperature); });

    // Find the top K scores
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), [prob](int i, int j) { return prob[i] > prob[j]; });

    // Normalize the top K probabilities
    float total = std::accumulate(indices.begin(), indices.begin() + k, 0.0f, [prob](float sum, int i) { return sum + prob[i]; });
    std::transform(indices.begin(), indices.begin() + k, prob.begin(), [total](int i) { return prob[i] / total; });

    // Sample a token from the top K
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dis(prob.begin(), prob.begin() + k);

    SetNextToken(batch_id, indices[dis(gen)]);
  }
#endif

  AppendNextTokensToSequences();
}

void SoftMax(std::span<ScoreType> scores, float temperature) {
  ScoreType max_score = *std::max_element(scores.begin(), scores.end());

  // Subtract max score and scale by temperature
  std::transform(scores.begin(), scores.end(), scores.begin(), [max_score, temperature](ScoreType score) { return std::exp((score - max_score) / temperature); });

  // Compute sum of exponentials
  ScoreType exp_sum = std::accumulate(scores.begin(), scores.end(), 0.0f);

  // Divide each score by the sum of exponentials
  std::transform(scores.begin(), scores.end(), scores.begin(), [exp_sum](ScoreType score) { return score / exp_sum; });
}

void GreedySearch_Cpu::SampleTopP(float p, float temperature) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, p);

  for (size_t batch_id = 0; batch_id < params_.batch_size; batch_id++) {
    if (PadIfAlreadyEOS(batch_id))
      continue;

    std::span<ScoreType> scores = next_token_scores_.subspan(batch_id * params_.vocab_size, params_.vocab_size);

    SoftMax(scores, temperature);

    // Sort an array of indices into the scores
    std::vector<int32_t> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [scores = scores.data()](int32_t i, int32_t j) { return scores[i] > scores[j]; });

    // Sample a probability threshold
    float threshold = dis(gen);

    int32_t token=0;
    // Find the first token where the cumulative probability exceeds the threshold
    for (int i = 0; i < scores.size();i++) {
      threshold -= scores[indices[i]];
      if (threshold>0)
        continue;

      token=indices[i];
      break;
    }

    SetNextToken(batch_id, token);
  }

  AppendNextTokensToSequences();
}

bool GreedySearch_Cpu::PadIfAlreadyEOS(size_t batch_id) {
   // If this batch entry has already seen the EOS token, append the pad token
  if (!eos_seen_[batch_id])
    return false;

  next_tokens_[batch_id] = params_.pad_token_id;
  return true;
}


void GreedySearch_Cpu::SetNextToken(size_t batch_id, int32_t token) {
  next_tokens_[batch_id] = token;
  if (token == params_.eos_token_id) {
    eos_seen_[batch_id] = true;
    if (--not_done_count_ == 0)
      done_ = true;
  }
}

void GreedySearch_Cpu::AppendNextTokensToSequences() {
  sequences_.AppendNextTokenToSequences(next_tokens_);

  if (sequences_.GetSequenceLength() == params_.max_length)
    done_ = true;
}

void BeamSearch_Cpu::AppendNextTokensToSequences() {
  sequences_.AppendNextTokenToSequences(beam_scorer_->GetNextIndicesCPU(), beam_scorer_->GetNextTokens());

  if (sequences_.GetSequenceLength() == params_.max_length)
    done_ = true;
}

void BeamSearch_Cpu::Finalize(size_t num_return_sequences, RoamingArray<int32_t> output, RoamingArray<float> sequence_scores) {
  beam_scorer_->Finalize(sequences_, num_return_sequences, output, sequence_scores);
}

std::span<ScoreType> Search_Cpu::GetScores(int batch_beam_index) {
  assert(batch_beam_index >= 0 && batch_beam_index < params_.BatchBeamSize());
  return next_token_scores_.subspan(batch_beam_index * params_.vocab_size, params_.vocab_size);
}

namespace Processors {

void MinLength(Search_Cpu& search, int min_length) {
  if (search.sequences_.GetSequenceLength() >= min_length)
    return;

  const int batch_beam_size = search.params_.BatchBeamSize();
  for (int i = 0; i < batch_beam_size; i++) {
    std::span<ScoreType> beam_token_scores = search.GetScores(i);
    beam_token_scores[search.params_.eos_token_id] = std::numeric_limits<ScoreType>::lowest();
  }
}

void RepetitionPenalty(Search_Cpu& search, ScoreType penalty) {
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

}  // namespace Generators