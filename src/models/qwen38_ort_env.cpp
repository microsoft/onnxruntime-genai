// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qwen38_ort_env.h"

#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace Generators {

namespace {

std::recursive_mutex& EnvironmentMutex() {
  static std::recursive_mutex mutex;
  return mutex;
}

// A null `value` removes the variable.
void WriteEnvironmentVariable(const std::string& name, const std::string* value) {
#if defined(_WIN32)
  // The CRT removes a variable that is set to the empty string.
  const bool written = _putenv_s(name.c_str(), value ? value->c_str() : "") == 0;
#else
  const bool written = (value ? setenv(name.c_str(), value->c_str(), 1) : unsetenv(name.c_str())) == 0;
#endif
  if (!written) throw std::runtime_error("qwen38_graph: cannot update the environment variable " + name);
}

}  // namespace

Qwen38OrtEnvScope::Qwen38OrtEnvScope(const std::vector<Qwen38OrtEnvVariable>& variables) : lock_{EnvironmentMutex()} {
  try {
    saved_.reserve(variables.size());
    for (const Qwen38OrtEnvVariable& variable : variables) {
      const char* current = std::getenv(variable.name.c_str());
      if (variable.only_if_unset && current != nullptr) continue;
      saved_.push_back(Saved{variable.name, variable.value, current ? std::optional<std::string>{current} : std::nullopt});
      WriteEnvironmentVariable(variable.name, &variable.value);
    }
  } catch (...) {
    Restore();
    throw;
  }
}

Qwen38OrtEnvScope::~Qwen38OrtEnvScope() { Restore(); }

bool Qwen38OrtEnvScope::StillPublished(const char* name) const {
  for (auto it = saved_.rbegin(); it != saved_.rend(); ++it) {
    if (it->name != name) continue;
    const char* current = std::getenv(name);
    return current != nullptr && it->published == current;
  }
  return false;
}

void Qwen38OrtEnvScope::Restore() noexcept {
  for (auto it = saved_.rbegin(); it != saved_.rend(); ++it) {
    try {
      WriteEnvironmentVariable(it->name, it->previous ? &*it->previous : nullptr);
    } catch (...) {
      // Nothing sensible is left to do when the environment cannot be written back.
    }
  }
  saved_.clear();
}

std::vector<Qwen38OrtEnvVariable> Qwen38OrtRunHints::Collect(const Qwen38PromptRunHints* prompt) const {
  std::vector<Qwen38OrtEnvVariable> variables;
  if (new_turn_) variables.push_back({Qwen38OrtEnv::kNewTurn, "1", false});
  if (handoff_done_) variables.push_back({Qwen38OrtEnv::kHandoffDone, "1", false});
  if (prompt != nullptr) {
    if (prompt->total_tokens > 0)
      variables.push_back({Qwen38OrtEnv::kPrefillTotalTokens, std::to_string(prompt->total_tokens), false});
    if (prompt->max_compact_experts > 0) {
      variables.push_back(
          {Qwen38OrtEnv::kCurrentMaxCompactExperts, std::to_string(prompt->max_compact_experts), false});
    }
  }
  if (warmup_) variables.push_back({Qwen38OrtEnv::kInWarmup, "1", false});
  if (deterministic_moe_) variables.push_back({Qwen38OrtEnv::kForceDeterministicMoe, "1", true});
  return variables;
}

}  // namespace Generators
