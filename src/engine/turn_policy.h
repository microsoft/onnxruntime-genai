// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "../config.h"
#include "../smartptrs.h"

/**
 * @file turn_policy.h
 * @brief The generation policy one Engine turn runs under.
 *
 * Every value here is resolved anew for each turn from the model-authored search defaults plus the
 * fields the caller explicitly set on that turn's options. Nothing carries over implicitly from the
 * preceding turn: a turn that overrides nothing decodes with model defaults again.
 */

namespace Generators {

struct TurnOptions;

/**
 * @brief The complete, already-resolved generation policy of one turn.
 *
 * Deliberately has no member initializers for the scalars: `ResolveTurnPolicy()` writes every one
 * of them from the model defaults plus that turn's overrides, so a default here would be a second,
 * silent source of policy that never applies. Anything that builds a policy outside
 * `ResolveTurnPolicy()` has to state every field.
 */
struct EffectiveTurnPolicy {
  bool do_sample;
  float temperature;
  float top_p;
  int top_k;
  float repetition_penalty;
  int no_repeat_ngram_size;
  size_t min_generated_tokens;
  std::optional<size_t> max_generated_tokens;

  /**
   * @brief The single classification every Engine execution path shares.
   *
   * Ordinary sampling, batched sampling, sampled draft verification, and draft eligibility all
   * decide "greedy or sampled" here, so a policy can never be greedy on one path and sampled on
   * another.
   */
  bool IsGreedy() const noexcept {
    return !do_sample || top_k == 1 || temperature == 0;
  }
};

/**
 * @brief True when a Search built on this scoring device implements no_repeat_ngram_size.
 *
 * Core-owned and keyed by the device type rather than asked of Search or DeviceInterface, so an
 * unsupported request is rejected at admission instead of throwing after the model has already run.
 */
bool SupportsNoRepeatNgram(DeviceType scoring_device_type) noexcept;

/**
 * @brief Resolves one turn's policy from model defaults plus that turn's explicit overrides.
 *
 * Performs no validation and touches no Request state.
 */
EffectiveTurnPolicy ResolveTurnPolicy(const Config::Search& model_defaults,
                                      const TurnOptions& options);

/**
 * @brief Rejects a resolved policy the Engine cannot honor exactly as written.
 *
 * Called before any Request or Engine mutation. Besides ordinary range checks it rejects two
 * inconsistencies between what the caller asked for and how the turn actually selects tokens:
 *
 * - An explicit `do_sample = true` that a model search default silently overrides, because the
 *   model supplies `top_k == 1` or `temperature == 0`. The caller cannot see those defaults through
 *   the options object, so the error names the model-supplied cause and the turn must override that
 *   exact field to sample. A caller who spelled greedy out themselves (`top_k == 1` or
 *   `temperature == 0` set on this turn) chose it and is accepted.
 * - An explicitly set distribution scalar that contradicts a greedy resolution -- a `temperature`
 *   other than 0 or 1, a nucleus `top_p` strictly between 0 and 1, or a `top_k` above 1 -- so a
 *   caller never believes a turn sampled when it actually selected the top logit. Values that are
 *   themselves consistent with greedy selection (`temperature == 0`, `top_k == 1`) or that restrict
 *   nothing (`top_k == 0`, `top_p` of 0 or 1, `temperature` of 1) are accepted.
 *
 * A zero `max_generated_tokens` is rejected earlier by Request::ValidateTurnAdmission(), which
 * every admission passes through first.
 *
 * @param policy The resolved policy.
 * @param options The turn options, read only to tell an explicit override from a model default.
 * @param scoring_device_type Device that will build this Request's Search.
 * @param vocab_size Number of tokens in the model vocabulary.
 * @param turn_prompt_length Sequence length the turn's generated output starts at.
 * @param max_session_tokens The Request's one session limit.
 */
void ValidateTurnPolicy(const EffectiveTurnPolicy& policy,
                        const TurnOptions& options,
                        DeviceType scoring_device_type,
                        int vocab_size,
                        size_t turn_prompt_length,
                        size_t max_session_tokens);

}  // namespace Generators
