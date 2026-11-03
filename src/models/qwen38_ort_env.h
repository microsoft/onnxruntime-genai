// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Qwen3.8-Flash-Next NVFP4 (model.type "qwen38_flash_graph"): the per-Run hints the engine hands to ORT's QMoE kernels.
//
// Those kernels (ORT contrib_ops/cuda/moe and the CUDA plugin EP) read their hints from process environment variables;
// ORT has no per-run config entry for them yet. The follow-up is to pass them through OrtRunOptions::AddConfigEntry,
// as the engine already does for `gpu_graph_id` and `memory.enable_memory_arena_shrinkage`; this file then goes away.
//   ORT_QMOE_PREFILL_NEW_TURN             "1" for the first Run of a turn (ORT resets it to "0" once it consumed it)
//   ORT_QMOE_PREFILL_TOTAL_TOKENS         prompt length of the turn, while the prompt is ingested
//   ORT_QMOE_CURRENT_MAX_COMPACT_EXPERTS  expert-compaction wave of one dense prefill chunk
//   ORT_QMOE_PREFILL_HANDOFF_DONE         "1" for the first Run after the prompt state moved to the decode graphs (ORT
//                                         resets it to "0" once it consumed it)
//   ORT_QMOE_IN_WARMUP                    "1" for the Runs of Warmup()
//   ORT_FORCE_DETERMINISTIC_MOE           "1" for the Runs from the first Warmup() on unless the process sets it
//                                         (deterministic tactics instead of GEMM profiling)
//
// Two rules keep that process-wide channel contained:
//  * The engine owns the values (Qwen38OrtRunHints) and never reads them back from the environment, so nothing another
//    engine or the caller writes there can change what it publishes.
//  * A value is published only by Qwen38OrtEnvScope, which holds one process-wide lock from the publish through the
//    Run to the restore of the previous values: engines never interleave their writes and nothing outlives the Run.
// The lock does not protect against threads outside the engines (other ORT sessions in the process) that read the
// environment while a Run publishes, nor against ORT's own writes (it resets the one-shot notices itself) racing such
// readers; only the per-run config entries remove that.
#pragma once

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace Generators {

namespace Qwen38OrtEnv {
inline constexpr const char* kNewTurn = "ORT_QMOE_PREFILL_NEW_TURN";
inline constexpr const char* kPrefillTotalTokens = "ORT_QMOE_PREFILL_TOTAL_TOKENS";
inline constexpr const char* kCurrentMaxCompactExperts = "ORT_QMOE_CURRENT_MAX_COMPACT_EXPERTS";
inline constexpr const char* kHandoffDone = "ORT_QMOE_PREFILL_HANDOFF_DONE";
inline constexpr const char* kInWarmup = "ORT_QMOE_IN_WARMUP";
inline constexpr const char* kForceDeterministicMoe = "ORT_FORCE_DETERMINISTIC_MOE";
}  // namespace Qwen38OrtEnv

struct Qwen38OrtEnvVariable {
  std::string name;
  std::string value;
  bool only_if_unset{false};  // leave a value that is already set (by the user) untouched
};

// Publishes `variables` for its lifetime and restores the previous values (unset stays unset) in reverse order on every
// exit path. Holds a process-wide recursive lock for its lifetime, so scopes of different threads never overlap.
class Qwen38OrtEnvScope {
 public:
  explicit Qwen38OrtEnvScope(const std::vector<Qwen38OrtEnvVariable>& variables);
  ~Qwen38OrtEnvScope();
  Qwen38OrtEnvScope(const Qwen38OrtEnvScope&) = delete;
  Qwen38OrtEnvScope& operator=(const Qwen38OrtEnvScope&) = delete;

  // True while `name` still holds the value this scope published for it: ORT resets a one-shot notice it consumed.
  bool StillPublished(const char* name) const;

 private:
  struct Saved {
    std::string name;
    std::string published;
    std::optional<std::string> previous;
  };

  void Restore() noexcept;

  std::unique_lock<std::recursive_mutex> lock_;
  std::vector<Saved> saved_;
};

// What a prompt chunk tells ORT beyond the turn state.
struct Qwen38PromptRunHints {
  size_t total_tokens{};      // prompt length of the turn; 0 = not announced
  int max_compact_experts{};  // expert-compaction wave of the chunk; 0 = ORT's default
};

// The hints of one model turn. The engine owns one; it is only touched from the engine's driving thread.
class Qwen38OrtRunHints {
 public:
  // Reset(): a new turn starts. The first Run after it says so; nothing of the previous turn survives.
  void BeginTurn() noexcept {
    new_turn_ = true;
    handoff_done_ = false;
    expected_prompt_tokens_ = 0;
  }

  // The executor's total prompt length of the turn (0 = unknown), see Qwen38DecodeDevice::ExpectPrefillTokens().
  void ExpectPrefillTokens(size_t total_tokens) noexcept { expected_prompt_tokens_ = total_tokens; }
  size_t ExpectedPrefillTokens() const noexcept { return expected_prompt_tokens_; }

  // A prompt chunk of `tokens` follows `ingested_tokens` earlier ones. Returns the prompt length ORT is told: the
  // announced one, never less than what the turn has ingested so far. The first chunk of a turn re-arms the new-turn
  // notice.
  size_t BeginPromptChunk(size_t ingested_tokens, size_t tokens) noexcept {
    if (ingested_tokens == 0) new_turn_ = true;
    handoff_done_ = false;
    return std::max(ingested_tokens + tokens, expected_prompt_tokens_);
  }

  // The prompt state moved to the decode graphs: the next Run carries the notice; the announced length is spent.
  void MarkHandoff() noexcept {
    handoff_done_ = true;
    expected_prompt_tokens_ = 0;
  }

  // Warmup() starts or ends: its Runs carry ORT_QMOE_IN_WARMUP. From the first warmup on, every Run of this engine
  // asks for deterministic MoE tactics (no GEMM profiling, which may not run while a graph is captured and would make
  // the tactics, hence the numerics, differ between Runs and processes) unless the process sets the variable itself.
  void SetWarmup(bool warmup) noexcept {
    warmup_ = warmup;
    if (warmup) deterministic_moe_ = true;
  }
  bool InWarmup() const noexcept { return warmup_; }

  // The variables to publish around the next Run; `prompt` is set for the Runs that ingest prompt tokens.
  std::vector<Qwen38OrtEnvVariable> Collect(const Qwen38PromptRunHints* prompt) const;

  // A Run returned. A one-shot notice that ORT did not reset was not consumed (the session has no QMoE layer that reads
  // it): it stays pending for the next Run, as the process-wide variable it replaces did.
  void RunReturned(const Qwen38OrtEnvScope& scope) {
    if (new_turn_ && !scope.StillPublished(Qwen38OrtEnv::kNewTurn)) new_turn_ = false;
    if (handoff_done_ && !scope.StillPublished(Qwen38OrtEnv::kHandoffDone)) handoff_done_ = false;
  }

 private:
  bool new_turn_{};
  bool handoff_done_{};
  bool warmup_{};
  bool deterministic_moe_{};
  size_t expected_prompt_tokens_{};
};

// Runs `run` (one ORT Run) with the hints published and the environment restored afterwards. A Run that throws leaves
// the one-shot notices pending.
template <typename RunFn>
void RunWithQwen38OrtHints(Qwen38OrtRunHints& hints, const Qwen38PromptRunHints* prompt, RunFn&& run) {
  Qwen38OrtEnvScope scope(hints.Collect(prompt));
  run();
  hints.RunReturned(scope);
}

}  // namespace Generators
