// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "search.h"
#include "beam_search_scorer.h"

#include <cmath>

namespace Generators {

void BeamHypotheses::Init(float length_penalty, std::span<HypothesisScore> beams) {
  beams_ = beams;
  beams_used_ = 0;
  length_penalty_ = length_penalty;
  done_ = false;
}

void BeamHypotheses::Add(cpu_span<int32_t> hypothesis, float sum_logprobs) {
  auto length = hypothesis.size();
  float const score = sum_logprobs / std::pow(static_cast<float>(length), length_penalty_);

  size_t index = beams_used_;
  // If the array is full, don't add unless it's better than the worst element
  if (index == beams_.size()) {
    if (score <= beams_[--index].score) {
      return;
    }
  } else {
    beams_used_++;
  }

  // Rotate existing elements over while the new element scores higher
  for (; index > 0 && score > beams_[index - 1].score; index--) {
    beams_[index] = beams_[index - 1];
  }

  beams_[index] = HypothesisScore{hypothesis, score};
}

bool BeamHypotheses::CanImprove(float best_sum_logprobs, int current_length) const {
  float const current_score = best_sum_logprobs / std::pow(static_cast<float>(current_length), length_penalty_);
  return beams_.back().score < current_score;
}

BeamSearchScorer::BeamSearchScorer(const GeneratorParams& parameters)
    : batch_size_{parameters.batch_size},
      num_beams_{parameters.search.num_beams},
      max_length_{parameters.search.max_length},
      pad_token_id_{parameters.pad_token_id},
      eos_token_id_{parameters.eos_token_id},
      early_stopping_{parameters.search.early_stopping},
      not_done_count_{parameters.batch_size} {
  size_t const batch_beam_size = static_cast<size_t>(batch_size_) * num_beams_;

  std::span<HypothesisScore> beams;
  hypothesis_scores_ptr_ = AllocateArray<HypothesisScore>(batch_beam_size, &beams);
  beam_hyps_ptr_ = AllocateArray<BeamHypotheses>(batch_size_, &beam_hyps_);
  for (size_t i = 0; i < batch_size_; i++) {
    beam_hyps_[i].Init(parameters.search.length_penalty, beams.subspan(i * num_beams_, num_beams_));
  }

  next_beam_scores_ptr_ = AllocateArray<float>(batch_beam_size, &next_beam_scores_);
  next_beam_tokens_ptr_ = AllocateArray<int32_t>(batch_beam_size, &next_beam_tokens_);
  next_beam_indices_ptr_ = AllocateArray<int32_t>(batch_beam_size, &next_beam_indices_);

  // Space to store intermediate sequence with length sequence_length, sequence_length + 1, ..., max_sequence_length.
  size_t const per_beam = (max_length_ * (max_length_ + 1) - (parameters.sequence_length - 1) * parameters.sequence_length) / 2;
  hypothesis_buffer_ptr_ = AllocateArray<int32_t>(batch_beam_size * per_beam, &hypothesis_buffer_);

  memset(next_beam_scores_.data(), 0, next_beam_scores_.size_bytes());

  // Initialize score of first beam of each group with 0 and the rest with -1e9.
  // This ensures that the beams in the same group don't produce same tokens every time.
  std::span<float> const beam_scores = next_beam_scores_;
  for (int i = 0; i < parameters.batch_size; i++) {
    for (int j = 1; j < parameters.search.num_beams; j++) {
      beam_scores[i * parameters.search.num_beams + j] = -1e9;
    }
  }
}

void BeamSearchScorer::Process(Sequences& sequences,
                               std::span<const float> next_scores,
                               std::span<const int32_t> next_tokens,
                               std::span<const int32_t> next_indices) {
  // Sequences shape is (batch_size * num_beams, total_sequence_length)
  // It contains word ID of whole sequence generated so far.
  // It is different from subgraph input_ids, which only need one word when past state is not empty.

  size_t sequence_length = static_cast<size_t>(sequences.GetSequenceLength());

  assert(next_scores.size() == next_tokens.size());
  assert(next_scores.size() == next_indices.size());

  for (size_t batch = 0; batch < batch_size_; batch++) {
    BeamHypotheses& beam_hyp = beam_hyps_[batch];
    if (beam_hyp.done_) {
      assert(beam_hyp.beams_used_ == num_beams_);  // Batch can only be done if all beams have been generated

      // Pad the batch.
      for (size_t j = 0; j < num_beams_; j++) {
        next_beam_scores_[batch * num_beams_ + j] = 0.0f;
        next_beam_tokens_[batch * num_beams_ + j] = pad_token_id_;
        next_beam_indices_[batch * num_beams_ + j] = 0;
      }
      continue;
    }

    // Next tokens for this sentence.
    size_t beam_idx = 0;
    size_t const top_k = 2 * num_beams_;
    for (size_t j = 0; j < top_k; j++) {
      int32_t const next_token = next_tokens[batch * top_k + j];
      float const next_score = next_scores[batch * top_k + j];
      int32_t const next_index = next_indices[batch * top_k + j];

      int const batch_beam_idx = static_cast<int>(batch * num_beams_) + next_index;
      // Add to generated hypotheses if end of sentence.
      if ((eos_token_id_ >= 0) && (next_token == eos_token_id_)) {
        bool const is_beam_token_worse_than_top_num_beams = (j >= num_beams_);
        if (is_beam_token_worse_than_top_num_beams) {
          continue;
        }

        // Clone the sequence and append to buffer.
        cpu_span<const int32_t> const src{sequences.GetSequence(batch_beam_idx)};
        auto clone_span = hypothesis_buffer_.subspan(static_cast<size_t>(hypothesis_buffer_used_), sequence_length);
        cpu_span<int32_t> clone{clone_span.data(), sequence_length};

        copy(src, clone);
        hypothesis_buffer_used_ += static_cast<int>(sequence_length);
        beam_hyp.Add(clone, next_score);
      } else {
        // Add next predicted token since it is not eos_token.
        next_beam_scores_[batch * num_beams_ + beam_idx] = next_score;
        next_beam_tokens_[batch * num_beams_ + beam_idx] = next_token;
        next_beam_indices_[batch * num_beams_ + beam_idx] = batch_beam_idx;
        ++beam_idx;
      }

      // Once the beam for next step is full, don't add more tokens to it.
      if (beam_idx == num_beams_) {
        break;
      }
    }

    assert(beam_idx == num_beams_);
    assert(static_cast<size_t>(hypothesis_buffer_used_) <= hypothesis_buffer_.size());

    //  Check if we are done so that we can save a pad step if all(done)
    if (static_cast<size_t>(beam_hyp.beams_used_) < num_beams_) {
      continue;
    }

    if (!early_stopping_) {
      std::span<const float> const topk_scores = next_scores.subspan(batch * num_beams_, top_k);
      const auto best_sum_logprobs = std::max_element(topk_scores.begin(), topk_scores.end());
      if (beam_hyp.CanImprove(*best_sum_logprobs, static_cast<int>(sequence_length))) {
        continue;
      }
    }

    beam_hyp.done_ = true;
    not_done_count_--;
  }
}

void BeamSearchScorer::Finalize(Sequences& sequences,
                                size_t num_return_sequences) {
  // Finalize all open beam hypotheses and add to generated hypotheses.
  for (size_t batch_index = 0; batch_index < batch_size_; batch_index++) {
    BeamHypotheses& beam_hyp = beam_hyps_[batch_index];
    if (beam_hyp.done_) {
      continue;
    }

    for (size_t beam_index = 0; beam_index < num_beams_; beam_index++) {
      size_t const batch_beam_index = batch_index * num_beams_ + beam_index;
      float const final_score = next_beam_scores_[batch_beam_index];
      auto final_tokens = sequences.GetSequence(batch_beam_index);
      beam_hyp.Add(final_tokens, final_score);
    }
  }
}

}  // namespace Generators
