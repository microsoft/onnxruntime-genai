// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "turn_policy.h"

#include <cassert>
#include <cmath>
#include <stdexcept>

#include "request.h"

namespace Generators {

namespace {

// Names the explicitly-set scalars that contradict a greedy resolution, so the error tells the
// caller exactly which of their values asked for a distribution the turn will never draw from.
void RejectContradictoryScalars(const std::string& greedy_cause,
                                const std::vector<const char*>& contradictory_fields) {
  std::string message =
      "The resolved turn policy selects the top logit (" + greedy_cause +
      "), so these explicitly set sampling options contradict it:";
  for (const char* field : contradictory_fields) {
    message += ' ';
    message += field;
  }
  message +=
      ". Either remove them or request sampling with do_sample=true, top_k != 1, and a nonzero "
      "temperature.";
  throw std::runtime_error(std::move(message));
}

// Names the model-supplied condition that keeps a turn greedy even though the caller explicitly
// asked to sample. The caller cannot see the model's search defaults from the options object, so
// the error names the field to override rather than a value to remove.
void RejectModelSuppliedGreedy(const std::vector<const char*>& model_causes) {
  std::string message =
      "do_sample=true was explicitly set, but this turn still selects the top logit because the "
      "model's own search defaults supply";
  for (size_t i = 0; i < model_causes.size(); ++i) {
    message += i == 0 ? " " : " and ";
    message += model_causes[i];
  }
  message +=
      ". Set that field explicitly on the turn -- top_k above 1, or a nonzero temperature -- to "
      "request sampling, or drop do_sample to accept top-logit selection.";
  throw std::runtime_error(std::move(message));
}

}  // namespace

bool SupportsNoRepeatNgram(DeviceType scoring_device_type) noexcept {
  // Only Search_Cpu implements ApplyNoRepeatNgram; the Search base class throws for every other
  // scoring device. Keyed by device type because the scoring device is what builds the Search.
  return scoring_device_type == DeviceType::CPU;
}

EffectiveTurnPolicy ResolveTurnPolicy(const Config::Search& model_defaults,
                                      const TurnOptions& options) {
  // Value-initialized so nothing is ever indeterminate; every field is then written below.
  EffectiveTurnPolicy policy{};
  policy.do_sample = options.do_sample.value_or(model_defaults.do_sample);
  policy.temperature = options.temperature.value_or(model_defaults.temperature);
  policy.top_p = options.top_p.value_or(model_defaults.top_p);
  policy.top_k = options.top_k.value_or(model_defaults.top_k);
  policy.repetition_penalty =
      options.repetition_penalty.value_or(model_defaults.repetition_penalty);
  policy.no_repeat_ngram_size =
      options.no_repeat_ngram_size.value_or(model_defaults.no_repeat_ngram_size);
  // Deliberately not defaulted from the model's session-absolute search.min_length: a per-turn
  // floor and a whole-session floor are different quantities, and Engine Request creation rejects
  // a model that still carries a nonzero legacy value.
  policy.min_generated_tokens = options.min_generated_tokens.value_or(0);
  policy.max_generated_tokens = options.max_generated_tokens;
  return policy;
}

void ValidateTurnPolicy(const EffectiveTurnPolicy& policy,
                        const TurnOptions& options,
                        DeviceType scoring_device_type,
                        size_t turn_prompt_length,
                        size_t max_session_tokens) {
  if (!std::isfinite(policy.temperature) || policy.temperature < 0.0f) {
    throw std::runtime_error(
        "temperature (" + std::to_string(policy.temperature) +
        ") must be finite and 0 or greater.");
  }
  if (!std::isfinite(policy.top_p) || policy.top_p < 0.0f || policy.top_p > 1.0f) {
    throw std::runtime_error(
        "top_p (" + std::to_string(policy.top_p) +
        ") must be finite and between 0.0 and 1.0.");
  }
  if (policy.top_k < 0) {
    throw std::runtime_error(
        "top_k (" + std::to_string(policy.top_k) + ") must be 0 or greater.");
  }
  if (!std::isfinite(policy.repetition_penalty) || policy.repetition_penalty <= 0.0f) {
    throw std::runtime_error(
        "repetition_penalty (" + std::to_string(policy.repetition_penalty) +
        ") must be finite and greater than zero.");
  }
  if (policy.no_repeat_ngram_size < 0) {
    throw std::runtime_error(
        "no_repeat_ngram_size (" + std::to_string(policy.no_repeat_ngram_size) +
        ") must be 0 or greater.");
  }
  if (policy.no_repeat_ngram_size > 0 && !SupportsNoRepeatNgram(scoring_device_type)) {
    throw std::runtime_error(
        "no_repeat_ngram_size is not supported on the selected scoring device type (" +
        to_string(scoring_device_type) + ").");
  }

  // A greedy resolution draws from no distribution at all. An explicitly set scalar that only
  // makes sense for a distribution is a caller mistake, not a preference: reject it instead of
  // silently selecting the top logit. Values that are consistent with greedy selection
  // (temperature 0, top_k 1) or that request no restriction at all (top_k 0, top_p 0 or 1,
  // temperature 1) are accepted, because honoring them exactly is what the turn already does.
  if (policy.IsGreedy()) {
    // A caller who spells greedy out themselves has chosen it; the model's defaults did not choose
    // it for them. Only these two spellings resolve to greedy on their own.
    const bool caller_requested_greedy =
        (options.top_k.has_value() && policy.top_k == 1) ||
        (options.temperature.has_value() && policy.temperature == 0.0f);
    // An explicit do_sample=true that a model default silently overrides is the one greedy
    // resolution the caller cannot have meant, so it is rejected before anything is mutated.
    // Requesting sampling means overriding the field the model set, not just setting do_sample.
    if (options.do_sample.value_or(false) && !caller_requested_greedy) {
      std::vector<const char*> model_causes;
      if (policy.top_k == 1) {
        model_causes.push_back("search.top_k = 1");
      }
      if (policy.temperature == 0.0f) {
        model_causes.push_back("search.temperature = 0");
      }
      // do_sample is true here, so top_k == 1 or temperature == 0 is the only way to be greedy,
      // and neither of them came from this turn's options.
      assert(!model_causes.empty());
      RejectModelSuppliedGreedy(model_causes);
    }

    std::vector<const char*> contradictory;
    // Every condition that made this policy greedy, so the error names the resolution the caller
    // actually got rather than one arbitrarily chosen reason.
    std::string cause;
    const auto add_cause = [&cause](const char* reason) {
      if (!cause.empty()) cause += " and ";
      cause += reason;
    };
    if (!policy.do_sample) add_cause("do_sample is false");
    if (policy.top_k == 1) add_cause("top_k is 1");
    if (policy.temperature == 0.0f) add_cause("temperature is 0");

    // Temperature 1 rescales nothing, exactly like top_p 1 and top_k 0, so it is neutral under a
    // greedy resolution rather than a request for a distribution.
    if (options.temperature.has_value() && policy.temperature != 0.0f &&
        policy.temperature != 1.0f) {
      contradictory.push_back("temperature");
    }
    if (options.top_p.has_value() && policy.top_p > 0.0f && policy.top_p < 1.0f) {
      contradictory.push_back("top_p");
    }
    if (options.top_k.has_value() && policy.top_k > 1) {
      contradictory.push_back("top_k");
    }
    if (!contradictory.empty()) {
      RejectContradictoryScalars(cause, contradictory);
    }
  } else if (policy.top_k == 0 && policy.top_p == 0.0f) {
    throw std::runtime_error(
        "A sampled turn requires a positive top_k or a positive top_p; both are 0.");
  }

  if (policy.min_generated_tokens != 0) {
    if (policy.max_generated_tokens &&
        policy.min_generated_tokens > *policy.max_generated_tokens) {
      throw std::runtime_error(
          "min_generated_tokens (" + std::to_string(policy.min_generated_tokens) +
          ") must not exceed max_generated_tokens (" +
          std::to_string(*policy.max_generated_tokens) + ").");
    }
    // The floor is an absolute sequence position, so it has to fit inside the session limit, which
    // Engine Request creation has already proven fits the internal Search length type.
    const size_t remaining_session_tokens =
        turn_prompt_length < max_session_tokens ? max_session_tokens - turn_prompt_length : 0;
    if (policy.min_generated_tokens > remaining_session_tokens) {
      throw std::runtime_error(
          "min_generated_tokens (" + std::to_string(policy.min_generated_tokens) +
          ") does not fit between the turn's prompt length (" +
          std::to_string(turn_prompt_length) + ") and max_session_tokens (" +
          std::to_string(max_session_tokens) + ").");
    }
  }
}

}  // namespace Generators
