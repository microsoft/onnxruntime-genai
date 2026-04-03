#include "generators.h"
#include "softmax.h"
#include "search.h"
#include "beam_search_scorer.h"
#include "cpu/interface.h"
#include <queue>
#include <algorithm>
#include <limits>

namespace Generators {

Search_Cpu::Search_Cpu(const GeneratorParams& params)
    : Search{params},
      cpu_device_{*GetCpuInterface()} {
  auto batch_beam_size = params.BatchBeamSize();

  sequence_lengths_ = cpu_device_.Allocate<int32_t>(batch_beam_size);
}

GreedySearch_Cpu::GreedySearch_Cpu(const GeneratorParams& params)
    : Search_Cpu(params) {
  if (params_->search.random_seed != -1)
    gen_.seed(params_->search.random_seed);
  else {
    std::random_device rd;
    std::array<uint32_t, decltype(gen_)::state_size> data;
    std::generate(data.begin(), data.end(), std::ref(rd));
    std::seed_seq seq(data.begin(), data.end());
    gen_.seed(seq);
  }

  next_tokens_ptr_ = cpu_device_.Allocate<int32_t>(params.search.batch_size);
  next_tokens_ptr_.Zero();
  next_tokens_ = cpu_span<int32_t>(next_tokens_ptr_.Span());

  eos_seen_buffer_ = AllocateArray<bool>(params.search.batch_size, &eos_seen_);
  memset(eos_seen_.data(), 0, eos_seen_.size_bytes());
}

BeamSearch_Cpu::BeamSearch_Cpu(const GeneratorParams& params)
    : Search_Cpu(params) {
  assert(params_->search.num_beams > 1);  // If 1, use GreedySearch
  beam_scorer_ = std::make_unique<BeamSearchScorer>(*params_);

  next_tokens_buffer_ = AllocateArray<int32_t>(params.BatchBeamSize(), &next_tokens_);
  memset(next_tokens_buffer_.get(), 0, next_tokens_.size_bytes());
}

BeamSearch_Cpu::~BeamSearch_Cpu() = default;

void Search_Cpu::ResetDone() {
  // Reset done count/state
  done_ = false;
}

void GreedySearch_Cpu::ResetDone() {
  Search_Cpu::ResetDone();
  not_done_count_ = params_->search.batch_size;
  memset(eos_seen_.data(), 0, eos_seen_.size_bytes());
}

DeviceSpan<float> Search_Cpu::GetLogits() const {
  return next_token_scores_;
}

void Search_Cpu::SetLogits(DeviceSpan<float> logits) {
  next_token_scores_ = logits;
  next_token_scores_.CopyDeviceToCpu();  // To the device->cpu copy once here as all later calls use CpuSpan()
}

DeviceSpan<int32_t> GreedySearch_Cpu::GetNextTokens() {
  return next_tokens_ptr_;
}

DeviceSpan<int32_t> BeamSearch_Cpu::GetNextTokens() {
  return beam_scorer_->GetNextTokens();
}

DeviceSpan<int32_t> BeamSearch_Cpu::GetNextIndices() {
  return beam_scorer_->GetNextIndices();
}

void BeamSearch_Cpu::SelectTop() {
  auto next_token_scores = next_token_scores_.CpuSpan();

  // Normalize next token scores
  for (size_t i = 0; i < params_->BatchBeamSize(); i++) {
    std::span<float> const scores = next_token_scores.subspan(i * static_cast<size_t>(params_->config.model.vocab_size), params_->config.model.vocab_size);
    LogSoftMax(scores, 1.0);
  }

  auto beam_scores = beam_scorer_->GetNextScores().Span();

  // Add beam score to next token scores. Corresponding python code is like:
  //    next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
  // TODO(aciddelgado): use thread pool to parallel
  int offset = 0;
  int batch_beam_index = 0;
  for (int i = 0; i < params_->search.batch_size; i++) {
    for (int j = 0; j < params_->search.num_beams; j++, batch_beam_index++) {
      for (int k = 0; k < params_->config.model.vocab_size; k++, offset++) {
        next_token_scores[offset] += beam_scores[batch_beam_index];
      }
    }
  }

  const size_t top_k = 2 * params_->search.num_beams;

  auto scores = std::make_unique<float[]>(top_k * params_->search.batch_size);     // Score of top_k tokens
  auto indices = std::make_unique<int32_t[]>(top_k * params_->search.batch_size);  // beam index of top_k tokens
  auto tokens = std::make_unique<int32_t[]>(top_k * params_->search.batch_size);   // token id of top_k tokens

  auto next_scores = std::span<float>(scores.get(), top_k * params_->search.batch_size);
  auto next_indices = std::span<int32_t>(indices.get(), top_k * params_->search.batch_size);
  auto next_tokens = std::span<int32_t>(tokens.get(), top_k * params_->search.batch_size);

  // Use partial_sort to find only the top 2*num_beams elements per batch,
  // instead of heapifying the entire vocab*beams array via priority_queue.
  const size_t total_elements = static_cast<size_t>(params_->search.num_beams) * params_->config.model.vocab_size;
  assert(total_elements >= top_k);
  std::vector<int32_t> idx(total_elements);

  for (size_t batch_index = 0; batch_index < static_cast<size_t>(params_->search.batch_size); batch_index++) {
    auto token_scores_sub = next_token_scores.subspan(batch_index * total_elements, total_elements);

    // Build index array and partial_sort to find top_k elements
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                      [&token_scores_sub](int32_t a, int32_t b) { return token_scores_sub[a] > token_scores_sub[b]; });

    auto next_indices_sub = next_indices.subspan(top_k * batch_index, top_k);
    auto next_tokens_sub = next_tokens.subspan(top_k * batch_index, top_k);
    auto next_scores_sub = next_scores.subspan(top_k * batch_index, top_k);
    for (size_t i = 0; i < top_k; i++) {
      next_indices_sub[i] = idx[i] / params_->config.model.vocab_size;
      next_tokens_sub[i] = idx[i] % params_->config.model.vocab_size;
      next_scores_sub[i] = token_scores_sub[idx[i]];
    }
  }

#if 0  // TODO(ryanhill): Use logging option
  DumpSpan(std::cout, next_tokens);
  DumpSpan(std::cout, next_indices_);
  DumpSpan(std::cout, next_scores_);
#endif

  beam_scorer_->Process(sequences_, next_scores, next_tokens, next_indices);
  next_tokens_ = cpu_span<int32_t>(beam_scorer_->GetNextTokens().Span());

  AppendNextTokensToSequences();
}

void GreedySearch_Cpu::SelectTop() {
  // next_tokens = torch.argmax(scores, dim=-1)
  for (size_t batch_id = 0; batch_id < params_->search.batch_size; batch_id++) {
    if (PadIfAlreadyEOS(batch_id)) {
      continue;
    }

    std::span<float> const scores = next_token_scores_.CpuSpan().subspan(batch_id * params_->config.model.vocab_size, params_->config.model.vocab_size);
    auto const token = static_cast<int32_t>(std::distance(scores.begin(), std::max_element(scores.begin(), scores.end())));
    SetNextToken(batch_id, token);
  }

  if (!done_)
    AppendNextTokensToSequences();
}

void GreedySearch_Cpu::SampleTopK(int k, float temperature) {
  const int vocab_size = params_->config.model.vocab_size;
  std::vector<int> indices(vocab_size);
  std::vector<float> top_k_scores(k);

  for (size_t batch_id = 0; batch_id < params_->search.batch_size; batch_id++) {
    if (PadIfAlreadyEOS(batch_id)) {
      continue;
    }
    std::span<float> const scores = next_token_scores_.CpuSpan().subspan(batch_id * vocab_size, vocab_size);
    // Find the top K scores
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), [scores = scores.data()](int i, int j) { return scores[i] > scores[j]; });
    for (int i = 0; i < k; i++)
      top_k_scores[i] = scores[indices[i]];
    // Sample a token from the top K
    Softmax(top_k_scores, temperature);
    std::discrete_distribution<> dis(top_k_scores.begin(), top_k_scores.end());
    SetNextToken(batch_id, indices[dis(gen_)]);
  }
  if (!done_)
    AppendNextTokensToSequences();
}

// Top-P (nucleus) sampling using adaptive partial sort.
//
// Instead of fully sorting the entire vocabulary O(V log V), this uses std::partial_sort
// with an adaptive K that starts small (256) and grows geometrically (4x) only when the
// cumulative probability of the top-K candidates hasn't yet reached the threshold p.
//
// Correctness: A single O(V) pass computes the global softmax partition function
// (max + sum-of-exp) before sorting. Cumulative probabilities are then measured
// against the true full-vocab distribution, ensuring the nucleus contains exactly
// the minimal set of tokens whose true probabilities sum to >= p.
//
// Complexity analysis (V = vocab_size, typically 128K-200K):
//   - Best case (common):  O(V) + O(V log 256) ≈ O(9V) — one pass for partition + partial sort
//   - Rare fallback:       O(V) + O(V log 1024)         — very flat distributions
//   - Worst case:          O(V) + O(V log V)             — uniform distribution (essentially never)
//
// Additional optimizations over the previous implementation:
//   - Buffers (indices, top_logits) are allocated once outside the batch loop,
//     eliminating ~800KB of heap allocation per batch item per decode step.
//   - Softmax for final sampling is computed only over the nucleus (not the full vocab).
//   - discrete_distribution is constructed over the nucleus (cutoff_index elements) instead
//     of the full vocabulary, avoiding an O(V) CDF construction on zeroed-out entries.
void GreedySearch_Cpu::SampleTopP(float p, float temperature) {
  const int vocab_size = params_->config.model.vocab_size;

  // Buffers allocated once outside the batch loop to avoid per-call heap allocations.
  // For vocab_size=200K, this saves ~800KB of malloc/free per batch item per token step.
  std::vector<int32_t> indices(vocab_size);
  std::vector<float> top_logits;

  for (size_t batch_id = 0; batch_id < params_->search.batch_size; batch_id++) {
    if (PadIfAlreadyEOS(batch_id)) {
      continue;
    }

    std::span<float> scores = next_token_scores_.CpuSpan().subspan(batch_id * vocab_size, vocab_size);

    // Step 1: Compute the global partition function (one O(V) pass) so that
    // cumulative probabilities measured over the sorted prefix are relative to
    // the true full-vocab distribution, not a renormalized subset.
    float inv_temp = 1.0f / temperature;
    float max_score = *std::max_element(scores.begin(), scores.end());
    float global_exp_sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
      global_exp_sum += std::exp((scores[i] - max_score) * inv_temp);
    }

    // Step 2: Adaptive partial sort to find nucleus candidates.
    //
    // Initialize index array [0, 1, 2, ..., V-1] for indirect sorting.
    // We partial_sort to place the top-K highest-scoring indices at the front,
    // then accumulate their true probabilities (using global_exp_sum) until
    // the cumulative mass reaches p.  If not, grow K by 4x and repeat.
    std::iota(indices.begin(), indices.end(), 0);

    int sorted_count = 0;               // Number of indices fully sorted so far
    int k = std::min(256, vocab_size);  // Initial candidate count (covers p>=0.99 for most LLM distributions)
    float cumulative_prob = 0.0f;
    int cutoff_index = 0;  // Number of tokens in the final nucleus

    while (k <= vocab_size) {
      // Partial sort: place the top-k elements (by score, descending) at indices[0..k-1].
      // Only the range [sorted_count, k) is newly sorted; [0, sorted_count) is already done.
      std::partial_sort(indices.begin() + sorted_count, indices.begin() + k, indices.end(),
                        [&scores](int32_t i, int32_t j) { return scores[i] > scores[j]; });

      // Accumulate true probabilities for the newly sorted elements [sorted_count, k).
      // Each probability = exp((score - max_score) * inv_temp) / global_exp_sum.
      cutoff_index = k;
      for (int i = sorted_count; i < k; ++i) {
        cumulative_prob += std::exp((scores[indices[i]] - max_score) * inv_temp) / global_exp_sum;
        if (cumulative_prob >= p) {
          cutoff_index = i + 1;
          break;
        }
      }
      sorted_count = k;

      // If we've accumulated enough probability, or we've exhausted the entire vocab, stop.
      if (cumulative_prob >= p || k == vocab_size)
        break;

      // The nucleus wasn't reached with the current K candidates. This means the
      // distribution is unusually flat. Grow K by 4x and try again.
      k = std::min(k * 4, vocab_size);
    }

    // Step 3: Re-normalize the nucleus probabilities and sample a token.
    // Build softmax over only the cutoff_index elements for proper sampling.
    top_logits.resize(cutoff_index);
    for (int i = 0; i < cutoff_index; ++i) {
      top_logits[i] = scores[indices[i]] * inv_temp;
    }
    Softmax(top_logits, 1.0f);

    std::discrete_distribution<> dist(top_logits.begin(), top_logits.end());
    int32_t sampled_index = dist(gen_);
    int32_t token = indices[sampled_index];

    SetNextToken(batch_id, token);
  }
  if (!done_)
    AppendNextTokensToSequences();
}

void GreedySearch_Cpu::SampleTopKTopP(int k, float p, float temperature) {
  assert(temperature > 0.0f);

  // --- Buffers allocated once to avoid re-allocations in the batch loop ---
  std::vector<int32_t> indices(params_->config.model.vocab_size);

  // Buffer for the top-k logits
  std::vector<float> top_k_logits;
  top_k_logits.reserve(k);

  // Buffer for probabilities used in Top-P filtering
  std::vector<float> temp_probs;
  temp_probs.reserve(k);

  for (size_t batch_id = 0; batch_id < params_->search.batch_size; batch_id++) {
    if (PadIfAlreadyEOS(batch_id)) {
      continue;
    }

    std::span<float> scores = next_token_scores_.CpuSpan().subspan(batch_id * params_->config.model.vocab_size, params_->config.model.vocab_size);

    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&scores](int32_t i, int32_t j) { return scores[i] > scores[j]; });

    // 2. Populate top K logits, applying temperature.
    top_k_logits.clear();
    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < k; ++i) {
      top_k_logits.push_back(scores[indices[i]] * inv_temp);
    }

    // 3. Top-p (nucleus) filtering.
    temp_probs.assign(top_k_logits.begin(), top_k_logits.end());
    Softmax(temp_probs, 1.0f);  // Temperature is already baked into top_k_logits

    float cumulative_prob = 0.0f;
    int cutoff_index = k;  // Default to keeping all top-k tokens
    for (int i = 0; i < k; ++i) {
      cumulative_prob += temp_probs[i];
      if (cumulative_prob >= p) {
        cutoff_index = i + 1;
        break;
      }
    }

    // 4. Resize logits to the nucleus, re-normalize, and sample.
    top_k_logits.resize(cutoff_index);
    Softmax(top_k_logits, 1.0f);

    std::discrete_distribution<> dist(top_k_logits.begin(), top_k_logits.end());
    int32_t token = indices[dist(gen_)];
    SetNextToken(batch_id, token);
  }
  if (!done_)
    AppendNextTokensToSequences();
}

bool GreedySearch_Cpu::PadIfAlreadyEOS(size_t batch_id) {
  // If this batch entry has already seen the EOS token, append the pad token
  if (!eos_seen_[batch_id]) {
    return false;
  }

  next_tokens_[batch_id] = params_->config.model.pad_token_id;
  return true;
}

void GreedySearch_Cpu::SetNextToken(size_t batch_id, int32_t token) {
  next_tokens_[batch_id] = token;
  if (contains(params_->config.model.eos_token_id, token)) {
    eos_seen_[batch_id] = true;
    if (g_log.enabled && g_log.hit_eos)
      Log("hit_eos", "EOS seen on batch " + std::to_string(batch_id));
    if (--not_done_count_ == 0) {
      done_ = true;
    }
  }
}

void GreedySearch_Cpu::AppendNextTokensToSequences() {
  // Append next token to each sequence.
  auto sequences_span = sequences_.GetSequences().CpuSpan();
  auto current_length = sequences_.GetSequenceLength();
  auto next_tokens = next_tokens_ptr_.Span();  // always on cpu
  auto batch_beam_size = params_->BatchBeamSize();
  for (int i = 0; i < batch_beam_size; i++) {
    sequences_span[i * sequences_.max_length_ + current_length] = next_tokens[i];
  }
  sequences_.GetSequences().CopyCpuToDevice();

  sequences_.AfterAppendNextTokens(next_tokens_ptr_, batch_beam_size);

  if (sequences_.GetSequenceLength() == params_->search.max_length) {
    if (g_log.enabled && g_log.hit_max_length)
      Log("hit_max_length", "greedy cpu hit");
    done_ = true;
  }
}

void GreedySearch_Cpu::AppendTokens(DeviceSpan<int32_t>& next_tokens) {
  // Set user-defined next tokens
  auto next_tokens_cpu = next_tokens.CpuSpan();
  auto batch_size = params_->search.batch_size;
  auto tokens_count_per_batch = next_tokens_cpu.size() / batch_size;
  for (size_t j = 0; j < tokens_count_per_batch; j++) {
    for (size_t i = 0; i < batch_size; i++) {
      SetNextToken(i, next_tokens_cpu[i * tokens_count_per_batch + j]);
    }
    AppendNextTokensToSequences();
  }

  ResetDone();
}

void GreedySearch_Cpu::RewindTo(size_t index) {
  ResetDone();

  // Set next tokens to the last tokens in the sequence
  if (index > 0) {
    for (int i = 0; i < params_->BatchBeamSize(); i++) {
      next_tokens_[i] = sequences_.GetSequences().Span()[(i * sequences_.max_length_) + index];
    }
  } else
    memset(next_tokens_.data(), 0, next_tokens_.size_bytes());
  sequences_.RewindTo(index);
}

void BeamSearch_Cpu::AppendTokens(DeviceSpan<int32_t>& next_tokens) {
  // Set user-defined next tokens
  auto next_tokens_cpu = next_tokens.CpuSpan();
  auto batch_beam_size = params_->BatchBeamSize();
  auto tokens_count_per_batch = next_tokens_cpu.size() / params_->search.batch_size;
  if (tokens_count_per_batch > sequences_.max_length_) {
    throw std::runtime_error("User-defined tokens exceed max_length.");
  }

  auto next_sequences_span = sequences_.GetNextSequences().Span();
  // Copy the user-defined tokens to the sequences
  for (ptrdiff_t i = 0; i < batch_beam_size; i++) {
    std::span<int32_t> target = next_sequences_span.subspan(i * sequences_.max_length_, tokens_count_per_batch);
    std::span<const int32_t> source = next_tokens_cpu.subspan((i / params_->search.num_beams) * tokens_count_per_batch, tokens_count_per_batch);
    copy(source, target);
  }
  sequences_.AfterAppendNextTokens(next_tokens, params_->search.batch_size);  // next_tokens is not expanded
}

bool BeamSearch_Cpu::IsDone() const {
  if (beam_scorer_->IsDone()) {
    return true;
  } else if (sequences_.GetSequenceLength() == params_->search.max_length) {
    return true;
  }
  return false;
}

void BeamSearch_Cpu::AppendNextTokensToSequences() {
  auto sequences_span = sequences_.GetSequences().CpuSpan();
  auto sequences_next_span = sequences_.GetNextSequences().CpuSpan();
  auto max_length = sequences_.max_length_;
  auto current_length = sequences_.GetSequenceLength();
  auto batch_beam_next_tokens = beam_scorer_->GetNextTokens().Span();
  auto batch_beam_indices = beam_scorer_->GetNextIndices().Span();
  auto batch_beam_size = params_->BatchBeamSize();

  for (ptrdiff_t i = 0; i < batch_beam_size; i++) {
    int batch_beam_index = batch_beam_indices[i];
    std::span<const int32_t> source = sequences_span.subspan(static_cast<size_t>(batch_beam_index) * max_length, current_length);
    std::span<int32_t> target = sequences_next_span.subspan(i * max_length, current_length);
    copy(source, target);

    // Append next token to each beam.
    sequences_next_span[i * max_length + current_length] = batch_beam_next_tokens[i];
  }
  auto next_tokens_device = beam_scorer_->GetNextTokens();
  sequences_.GetNextSequences().CopyCpuToDevice();
  sequences_.AfterAppendNextTokens(next_tokens_device, params_->BatchBeamSize());

  if (sequences_.GetSequenceLength() == params_->search.max_length) {
    if (g_log.enabled && g_log.hit_max_length)
      Log("hit_max_length", "beam cpu hit");
    done_ = true;
  }
}

void BeamSearch_Cpu::Finalize(size_t num_return_sequences) {
  if (finalized_)
    return;
  beam_scorer_->Finalize(sequences_, num_return_sequences);
  finalized_ = true;
}

DeviceSpan<int32_t> BeamSearch_Cpu::GetSequence(size_t index) {
  size_t batch_id = index / params_->search.num_return_sequences;
  size_t beam_id = index % params_->search.num_return_sequences;
  Finalize(params_->search.num_return_sequences);
  return beam_scorer_->GetBeamHypotheses(batch_id, beam_id);
}

DeviceSpan<int32_t> BeamSearch_Cpu::GetSequence(size_t batch_id, size_t beam_id) {
  Finalize(params_->search.num_return_sequences);
  return beam_scorer_->GetBeamHypotheses(batch_id, beam_id);
}

std::span<float> Search_Cpu::GetScores(int batch_beam_index) {
  assert(batch_beam_index >= 0 && batch_beam_index < params_->BatchBeamSize());
  return next_token_scores_.CpuSpan().subspan(static_cast<size_t>(batch_beam_index) * params_->config.model.vocab_size, params_->config.model.vocab_size);
}

void Search_Cpu::ApplyMinLength(int min_length) {
  if (sequences_.GetSequenceLength() >= min_length) {
    return;
  }

  const int batch_beam_size = params_->BatchBeamSize();
  for (int i = 0; i < batch_beam_size; i++) {
    std::span<float> const beam_token_scores = GetScores(i);
    for (auto token_id : params_->config.model.eos_token_id)
      beam_token_scores[token_id] = std::numeric_limits<float>::lowest();
  }
}

void Search_Cpu::ApplyRepetitionPenalty(float penalty) {
  if (penalty == 1.0f)
    return;

  const int batch_beam_size = params_->BatchBeamSize();
  for (int i = 0; i < batch_beam_size; i++) {
    std::span<float> const beam_token_scores = GetScores(i);
    std::span<const int32_t> const sequence = sequences_.GetSequence(i).CopyDeviceToCpu();

    if (repetition_penalty_visited_.empty())
      repetition_penalty_visited_.resize(params_->config.model.vocab_size, false);

    for (const auto& word_id : sequence) {
      if (word_id >= 0 && word_id < params_->config.model.vocab_size && !repetition_penalty_visited_[word_id]) {
        repetition_penalty_visited_[word_id] = true;
        float const score = beam_token_scores[word_id];
        beam_token_scores[word_id] = (score < 0 ? score * penalty : score / penalty);
      }
    }

    // Reset visited flags for tokens we touched (O(seq_len), not O(vocab_size))
    for (const auto& word_id : sequence) {
      if (word_id >= 0 && word_id < params_->config.model.vocab_size)
        repetition_penalty_visited_[word_id] = false;
    }
  }
}

}  // namespace Generators
