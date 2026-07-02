// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>
#include "span.h"

// Shared CPU sampling-distribution helpers used by both standard and speculative decoding.

namespace Generators {

// --- Logits penalties (min-length + repetition) -------------------------------------------
// Shared CPU logits-processing helpers used by both standard search (Search_Cpu) and speculative
// decoding, so the two paths apply identical penalties to a single logits row.

// Mask the end-of-stream tokens (set to lowest float) while the sequence is shorter than
// min_length, so EOS cannot be selected yet. Mirrors Search_Cpu::ApplyMinLength for one row.
// current_length is the sequence length *before* the token this row predicts is appended.
inline void ApplyMinLengthToLogits(std::span<float> logits, int current_length, int min_length,
                                   std::span<const int32_t> eos_token_ids) {
  if (current_length >= min_length)
    return;
  const int vocab_size = static_cast<int>(logits.size());
  for (auto eos : eos_token_ids) {
    if (eos >= 0 && eos < vocab_size)
      logits[static_cast<size_t>(eos)] = std::numeric_limits<float>::lowest();
  }
}

// Divide/multiply the logit of every token that already appears in prefix (matching HF's
// convention: score < 0 -> score * penalty, else score / penalty). Mirrors
// Search_Cpu::ApplyRepetitionPenalty for one row. visited is a reusable vocab-sized scratch
// buffer (grown on first use, left all-false on return so it can be reused across rows).
inline void ApplyRepetitionPenaltyToLogits(std::span<float> logits, std::span<const int32_t> prefix,
                                           float penalty, std::vector<bool>& visited) {
  if (penalty == 1.0f)
    return;
  const int vocab_size = static_cast<int>(logits.size());
  if (static_cast<int>(visited.size()) < vocab_size)
    visited.assign(static_cast<size_t>(vocab_size), false);

  for (const auto& word_id : prefix) {
    if (word_id >= 0 && word_id < vocab_size && !visited[word_id]) {
      visited[word_id] = true;
      const float score = logits[static_cast<size_t>(word_id)];
      logits[static_cast<size_t>(word_id)] = (score < 0 ? score * penalty : score / penalty);
    }
  }

  // Reset only the flags we touched (O(prefix), not O(vocab)).
  for (const auto& word_id : prefix) {
    if (word_id >= 0 && word_id < vocab_size)
      visited[word_id] = false;
  }
}

// Migrated from search.cpp so speculative decoding can share nucleus selection; only change: static -> inline.
//
// Find the minimal nucleus of tokens whose cumulative probability >= p using
// adaptive partial sort with a log-space tail bound. This avoids computing the
// global softmax partition function (O(V) exp() calls) by bounding the unsorted
// tail probability at each step.
//
// At sorted position i, all V-i-1 remaining elements have score <= scores[indices[i]]
// (by sort order within the prefix, and by partial_sort guarantee beyond it). So:
//   tail_sum <= (V - i - 1) * exp((scores[indices[i]] - max_score) * inv_temp)
// The cumulative probability lower bound is:
//   prefix_sum / (prefix_sum + tail_bound)
// When this exceeds p, the true cumulative (which is >= this bound) also exceeds p.
//
// The nucleus found may include a few extra tokens compared to exact global softmax,
// but is always a valid nucleus (true cumulative prob >= p). For typical peaked LLM
// distributions, the bound is tight and the result is identical.
inline int FindNucleus(std::span<const float> scores, std::span<int32_t> indices,
                       float p, float max_score, float inv_temp) {
  const int vocab_size = static_cast<int>(scores.size());

  // Start small (16) to minimize wasted work when the top few tokens dominate
  // (common at low temperature). Grow 4× to avoid O(V log V) full-sort fallback
  // that a 2× growth with a hard cap would require for flat distributions.
  constexpr int kInitialCandidateCount = 16;
  constexpr int kCandidateCountGrowthFactor = 4;

  std::iota(indices.begin(), indices.end(), 0);

  int sorted_count = 0;
  int k = std::min(kInitialCandidateCount, vocab_size);
  float prefix_exp_sum = 0.0f;
  int cutoff_index = 0;

  while (k <= vocab_size) {
    // Partial sort: place the top-k elements (by score, descending) at indices[0..k-1].
    // Only the range [sorted_count, k) is newly sorted; [0, sorted_count) is already done.
    std::partial_sort(indices.begin() + sorted_count, indices.begin() + k, indices.end(),
                      [&scores](int32_t i, int32_t j) { return scores[i] > scores[j]; });

    // Accumulate exp() for newly sorted elements and check the tail bound.
    cutoff_index = k;
    for (int i = sorted_count; i < k; ++i) {
      float exp_i = std::exp((scores[indices[i]] - max_score) * inv_temp);
      prefix_exp_sum += exp_i;

      // Upper bound on remaining elements: all V-i-1 unsorted entries have score <= current.
      float tail_bound = static_cast<float>(vocab_size - i - 1) * exp_i;
      if (prefix_exp_sum >= p * (prefix_exp_sum + tail_bound)) {
        cutoff_index = i + 1;
        break;
      }
    }
    sorted_count = k;

    if (cutoff_index < k || k == vocab_size)
      break;

    k = std::min(k * kCandidateCountGrowthFactor, vocab_size);
  }

  // When the vocabulary is small enough that the tail bound may be loose,
  // compute the exact cutoff using the true partition function.
  // For large V this path is never reached (tail bound is tight for peaked distributions).
  if (vocab_size <= kInitialCandidateCount * kCandidateCountGrowthFactor) {
    // Ensure all elements are sorted for exact computation
    if (sorted_count < vocab_size) {
      std::partial_sort(indices.begin() + sorted_count, indices.begin() + vocab_size, indices.end(),
                        [&scores](int32_t i, int32_t j) { return scores[i] > scores[j]; });
    }
    float global_exp_sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
      global_exp_sum += std::exp((scores[indices[i]] - max_score) * inv_temp);
    }
    float cum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
      cum += std::exp((scores[indices[i]] - max_score) * inv_temp);
      if (cum >= p * global_exp_sum) {
        return i + 1;
      }
    }
  }

  return cutoff_index;
}

// New struct holding a sparse truncated sampling distribution; added so the kept set can be
// returned to a caller (speculative decoding) instead of being sampled and discarded inline.
// Sparse truncated sampling distribution: kept token ids + renormalized probs (sum to 1).
// scratch is a reusable vocab-sized buffer to avoid per-token allocation.
struct SampledCategorical {
  std::vector<int32_t> indices;
  std::vector<float> probs;
  std::vector<int32_t> scratch;
};

// New function consolidating the top-k/top-p/top-k+top-p paths from search.cpp's SampleTop*;
// added so standard and speculative decoding build the distribution through one shared routine.
// Inputs: scores (logits), top_k, top_p, temperature; output struct out
// Output: sparse truncated distribution written into out (indices + renormalized probs)
inline void ComputeSampledCategorical(std::span<const float> scores, int top_k,
                                      float top_p, float temperature,
                                      SampledCategorical& out) {
  const int vocab_size = static_cast<int>(scores.size());
  const float inv_temp = 1.0f / temperature;
  out.scratch.resize(vocab_size);

  const bool apply_topk = (top_k > 1);
  const bool apply_topp = (top_p > 0.0f && top_p < 1.0f);

  if (apply_topk) {
    // Restrict to the top-k logits (matches SampleTopK / SampleTopKTopP).
    const int m = std::min(top_k, vocab_size);
    std::iota(out.scratch.begin(), out.scratch.end(), 0);
    std::partial_sort(out.scratch.begin(), out.scratch.begin() + m, out.scratch.end(),
                      [&scores](int32_t i, int32_t j) { return scores[i] > scores[j]; });

    const float max_score = scores[out.scratch[0]];
    std::vector<float> probs(m);
    float sum = 0.0f;
    for (int j = 0; j < m; j++) {
      probs[j] = std::exp((scores[out.scratch[j]] - max_score) * inv_temp);
      sum += probs[j];
    }
    if (sum > 0.0f)
      for (auto& v : probs) v /= sum;

    // Optional nucleus cutoff within the top-k (matches SampleTopKTopP).
    int keep = m;
    if (apply_topp) {
      float cum = 0.0f;
      for (int j = 0; j < m; j++) {
        cum += probs[j];
        if (cum >= top_p) {
          keep = j + 1;
          break;
        }
      }
    }

    // Renormalize the kept set (equivalent to re-softmax over the kept logits).
    float keep_sum = 0.0f;
    for (int j = 0; j < keep; j++) keep_sum += probs[j];
    out.indices.assign(out.scratch.begin(), out.scratch.begin() + keep);
    out.probs.resize(keep);
    const float inv = keep_sum > 0.0f ? 1.0f / keep_sum : 0.0f;
    for (int j = 0; j < keep; j++) out.probs[j] = probs[j] * inv;
    return;
  }

  // Nucleus over the full vocabulary (matches SampleTopP).
  float max_score = *std::max_element(scores.begin(), scores.end());
  int cutoff = FindNucleus(scores, {out.scratch.data(), out.scratch.size()}, top_p, max_score, inv_temp);

  std::vector<float> probs(cutoff);
  float sum = 0.0f;
  for (int j = 0; j < cutoff; j++) {
    probs[j] = std::exp((scores[out.scratch[j]] - max_score) * inv_temp);
    sum += probs[j];
  }
  out.indices.assign(out.scratch.begin(), out.scratch.begin() + cutoff);
  out.probs.resize(cutoff);
  const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
  for (int j = 0; j < cutoff; j++) out.probs[j] = probs[j] * inv;
}

// New function expanding a sparse SampledCategorical into a dense vector; added so speculative
// decoding can do per-token-id probability lookups for the accept/correction math.
// Inputs: sparse SampledCategorical c, vocab_size
// Output: dense full-vocab probability vector (zeros outside the kept set)
inline std::vector<float> ScatterToFullVocab(const SampledCategorical& c, int vocab_size) {
  std::vector<float> dense(vocab_size, 0.0f);
  for (size_t j = 0; j < c.indices.size(); j++)
    dense[c.indices[j]] = c.probs[j];
  return dense;
}

}  // namespace Generators
