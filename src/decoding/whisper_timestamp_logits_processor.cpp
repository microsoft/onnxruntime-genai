// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "decoding/whisper_timestamp_logits_processor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "generator/generators.h"
#include "search.h"

namespace Generators {
namespace {

constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();

void Mask(std::span<float> logits) {
  std::fill(logits.begin(), logits.end(), kNegativeInfinity);
}

}  // namespace

WhisperTimestampLogitsProcessor::WhisperTimestampLogitsProcessor(WhisperTimestampLogitsConfig config)
    : config_{std::move(config)} {
  if (config_.timestamp_begin < 0)
    throw std::invalid_argument("Whisper timestamp_begin must be non-negative");
  if (config_.eot_token < 0)
    throw std::invalid_argument("Whisper eot_token must be non-negative");
  if (config_.max_initial_timestamp_index && *config_.max_initial_timestamp_index < 0)
    throw std::invalid_argument("Whisper max_initial_timestamp_index must be non-negative");
  if (config_.no_timestamps_token && *config_.no_timestamps_token < 0)
    throw std::invalid_argument("Whisper no_timestamps_token must be non-negative");
}

void WhisperTimestampLogitsProcessor::Apply(std::span<float> logits,
                                            std::span<const int32_t> tokens,
                                            size_t sample_begin) const {
  if (logits.empty())
    throw std::invalid_argument("Whisper timestamp logits cannot be empty");
  if (sample_begin > tokens.size())
    throw std::invalid_argument("Whisper timestamp sample_begin exceeds token history");
  if (config_.timestamp_begin >= static_cast<int>(logits.size()) ||
      config_.eot_token >= static_cast<int>(logits.size())) {
    throw std::invalid_argument("Whisper timestamp token metadata exceeds vocabulary size");
  }
  if (config_.eot_token >= config_.timestamp_begin)
    throw std::invalid_argument("Whisper eot_token must precede timestamp tokens");
  if (config_.no_timestamps_token &&
      *config_.no_timestamps_token >= static_cast<int>(logits.size())) {
    throw std::invalid_argument("Whisper no_timestamps_token exceeds vocabulary size");
  }
  if (config_.max_initial_timestamp_index &&
      *config_.max_initial_timestamp_index >
          static_cast<int>(logits.size()) - 1 - config_.timestamp_begin) {
    throw std::invalid_argument("Whisper max_initial_timestamp_index exceeds timestamp vocabulary");
  }

  ValidateTokens(tokens, sample_begin, logits.size());

  if (config_.no_timestamps_token)
    logits[*config_.no_timestamps_token] = kNegativeInfinity;
  Mask(logits.subspan(static_cast<size_t>(config_.eot_token + 1),
                      static_cast<size_t>(config_.timestamp_begin - config_.eot_token - 1)));

  const auto sampled_tokens = tokens.subspan(sample_begin, tokens.size() - sample_begin);
  const bool last_was_timestamp =
      !sampled_tokens.empty() && sampled_tokens.back() >= config_.timestamp_begin;
  const bool penultimate_was_timestamp =
      sampled_tokens.size() < 2 || sampled_tokens[sampled_tokens.size() - 2] >= config_.timestamp_begin;

  if (last_was_timestamp) {
    if (penultimate_was_timestamp) {
      Mask(logits.subspan(config_.timestamp_begin, logits.size() - config_.timestamp_begin));
    } else {
      Mask(logits.subspan(0, config_.eot_token));
    }
  }

  auto last_timestamp = -1;
  for (int32_t token : sampled_tokens) {
    if (token >= config_.timestamp_begin)
      last_timestamp = token;
  }
  if (last_timestamp >= config_.timestamp_begin) {
    const int first_allowed_timestamp =
        last_was_timestamp && !penultimate_was_timestamp ? last_timestamp : last_timestamp + 1;
    Mask(logits.subspan(config_.timestamp_begin,
                        static_cast<size_t>(first_allowed_timestamp - config_.timestamp_begin)));
  }

  if (sampled_tokens.empty()) {
    Mask(logits.subspan(0, config_.timestamp_begin));
    if (config_.max_initial_timestamp_index) {
      const int first_disallowed_timestamp =
          config_.timestamp_begin + *config_.max_initial_timestamp_index + 1;
      Mask(logits.subspan(first_disallowed_timestamp, logits.size() - first_disallowed_timestamp));
    }
  }

  const auto timestamp_logits =
      logits.subspan(config_.timestamp_begin, logits.size() - config_.timestamp_begin);
  const auto text_logits = logits.subspan(0, config_.timestamp_begin);
  float maximum = kNegativeInfinity;
  float max_text_logit = kNegativeInfinity;
  for (float value : timestamp_logits) {
    if (std::isfinite(value))
      maximum = std::max(maximum, value);
  }
  for (float value : text_logits) {
    if (std::isfinite(value)) {
      maximum = std::max(maximum, value);
      max_text_logit = std::max(max_text_logit, value);
    }
  }

  if (std::isfinite(maximum)) {
    double timestamp_mass = 0.0;
    for (float value : timestamp_logits) {
      if (std::isfinite(value))
        timestamp_mass += std::exp(static_cast<double>(value) - maximum);
    }
    const double max_text_mass =
        std::isfinite(max_text_logit)
            ? std::exp(static_cast<double>(max_text_logit) - maximum)
            : 0.0;
    if (timestamp_mass > max_text_mass)
      Mask(text_logits);
  }

  bool has_finite_logit = false;
  for (float value : logits) {
    has_finite_logit = has_finite_logit || std::isfinite(value);
  }
  if (!has_finite_logit)
    throw std::runtime_error("Whisper timestamp rules masked every token");
}

void WhisperTimestampLogitsProcessor::ValidateTokens(std::span<const int32_t> tokens,
                                                     size_t sample_begin,
                                                     size_t vocab_size) const {
  if (sample_begin > tokens.size())
    throw std::invalid_argument("Whisper timestamp sample_begin exceeds token history");
  for (int32_t token : tokens) {
    if (token < 0 || token >= static_cast<int32_t>(vocab_size))
      throw std::invalid_argument("Whisper timestamp token history contains an invalid token");
  }
  if (config_.no_timestamps_token) {
    for (int32_t token : tokens.subspan(0, sample_begin)) {
      if (token == *config_.no_timestamps_token) {
        throw std::runtime_error(
            "Whisper timestamp decoding cannot be enabled when the prompt contains no_timestamps_token_id");
      }
    }
  }
}

void ApplyWhisperTimestampRulesToSearch(Search& search,
                                        const WhisperTimestampLogitsProcessor& processor,
                                        size_t sample_begin) {
  const size_t batch_beam_size = static_cast<size_t>(search.params_->BatchBeamSize());
  const size_t vocab_size = static_cast<size_t>(search.params_->config.model.vocab_size);
  auto logits = search.GetLogits();
  if (logits.size() != batch_beam_size * vocab_size) {
    throw std::runtime_error("Whisper timestamp logits size does not match batch_beam_size * vocab_size");
  }

  auto& device = *search.params_->p_device;
  if (device.GetType() != DeviceType::CPU) {
    const auto& config = processor.GetConfig();
    const bool applied = device.ApplyWhisperTimestampRules(
        logits.Span().data(),
        search.sequences_.GetSequences().Span().data(),
        search.GetSequenceDoneDevice(),
        static_cast<int>(batch_beam_size),
        static_cast<int>(vocab_size),
        search.sequences_.max_length_,
        search.sequences_.GetSequenceLength(),
        static_cast<int>(sample_begin),
        config.timestamp_begin,
        config.eot_token,
        config.no_timestamps_token.value_or(-1),
        config.max_initial_timestamp_index.value_or(-1));
    if (!applied)
      throw std::runtime_error(
          "Whisper timestamp decoding is not supported by the selected execution provider");
    return;
  }

  auto cpu_logits = logits.CpuSpan();

  for (size_t row = 0; row < batch_beam_size; ++row) {
    if (search.IsSequenceDone(row))
      continue;

    auto row_logits = cpu_logits.subspan(row * vocab_size, vocab_size);
    for (float& value : row_logits) {
      if (std::isnan(value) || value == std::numeric_limits<float>::infinity())
        throw std::runtime_error("Whisper timestamp logits contain NaN or positive infinity");
      if (value == std::numeric_limits<float>::lowest())
        value = kNegativeInfinity;
    }
    processor.Apply(row_logits, search.sequences_.GetSequence(row).CpuSpan(), sample_begin);
  }
}

}  // namespace Generators
