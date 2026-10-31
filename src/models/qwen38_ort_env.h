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
// Three rules keep that process-wide channel contained:
//  * The engine owns the values (Qwen38OrtRunHints) and never reads them back from the environment, so nothing another
//    engine or the caller writes there can change what it publishes.
//  * A value is published only by Qwen38OrtEnvScope, which holds one process-wide lock from the publish through the
//    Run to the restore of the previous values: engines never interleave their writes and nothing outlives the Run.
//  * The turn state belongs to a slot of the engine, not to the engine: the notices and the prompt length of a slot are
//    published only by the Runs that serve that slot, and what a prompt Run publishes is computed from that Run alone
//    (Qwen38PromptRunHints), never kept or merged across Runs. Slots that ingest their prompts in interleaved chunks, or
//    that decode in one batched Run while another is prefilling, therefore never see each other's values.
// The lock does not protect against threads outside the engines (other ORT sessions in the process) that read the
// environment while a Run publishes, nor against ORT's own writes (it resets the one-shot notices itself) racing such
// readers; only the per-run config entries remove that.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
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

// The slots of the engine one ORT Run serves: bit s is slot s. A prompt chunk Run serves one slot, a batched decode Run
// the slots that decode in it, a warm-up Run the slots it exercises.
using Qwen38SlotMask = uint32_t;
constexpr Qwen38SlotMask Qwen38SlotBit(size_t slot) noexcept { return Qwen38SlotMask{1} << slot; }
// The first `count` slots (0..count-1), for a Run that exercises that many.
constexpr Qwen38SlotMask Qwen38FirstSlots(size_t count) noexcept { return Qwen38SlotBit(count) - 1; }
template <typename Slots>
Qwen38SlotMask Qwen38SlotMaskOf(const Slots& slots) noexcept {
  Qwen38SlotMask mask = 0;
  for (const size_t slot : slots) mask |= Qwen38SlotBit(slot);
  return mask;
}

// The hints of one engine: the turn state of each of its slots and the warm-up state. The engine owns one; it is only
// touched from the engine's driving thread. The forms without a slot argument address slot 0, which is the only slot of
// a single-slot engine.
class Qwen38OrtRunHints {
 public:
  explicit Qwen38OrtRunHints(size_t slot_count = 1) { SetSlotCount(slot_count); }

  // The engine has `slot_count` slots (at least one); the turn state of the ones that exist is kept.
  void SetSlotCount(size_t slot_count) {
    if (slot_count == 0 || slot_count > kMaxSlots) {
      throw std::invalid_argument("qwen38_graph: the run hints serve 1.." + std::to_string(kMaxSlots) + " slots, got " +
                                  std::to_string(slot_count));
    }
    slots_.resize(slot_count);
  }
  size_t SlotCount() const noexcept { return slots_.size(); }

  // Reset(): every slot starts a new turn. ResetSlot(): the slot does and the others are untouched. The first Run that
  // serves the slot after it says so; nothing of the previous turn of the slot survives.
  void BeginAllTurns() noexcept {
    for (SlotHints& slot : slots_) BeginTurn(slot);
  }
  void BeginTurn() noexcept { BeginTurn(slots_.front()); }
  void BeginTurn(size_t slot) { BeginTurn(At(slot)); }

  // The executor's total prompt length of the turn of the slot (0 = unknown), see
  // Qwen38DecodeDevice::ExpectPrefillTokensSlot().
  void ExpectPrefillTokens(size_t total_tokens) noexcept { slots_.front().expected_prompt_tokens = total_tokens; }
  void ExpectPrefillTokens(size_t slot, size_t total_tokens) { At(slot).expected_prompt_tokens = total_tokens; }
  size_t ExpectedPrefillTokens() const noexcept { return slots_.front().expected_prompt_tokens; }
  size_t ExpectedPrefillTokens(size_t slot) const { return At(slot).expected_prompt_tokens; }

  // A prompt chunk of `tokens` follows `ingested_tokens` earlier ones of the slot. Returns the prompt length ORT is told:
  // the announced one of the slot, never less than what its turn has ingested so far. The first chunk of a turn re-arms
  // the new-turn notice of the slot.
  size_t BeginPromptChunk(size_t ingested_tokens, size_t tokens) noexcept {
    return BeginPromptChunk(slots_.front(), ingested_tokens, tokens);
  }
  size_t BeginPromptChunk(size_t slot, size_t ingested_tokens, size_t tokens) {
    return BeginPromptChunk(At(slot), ingested_tokens, tokens);
  }

  // The prompt state of the slot moved to the decode graphs: the next Run that serves it carries the notice; its announced
  // length is spent.
  void MarkHandoff() noexcept { MarkHandoff(slots_.front()); }
  void MarkHandoff(size_t slot) { MarkHandoff(At(slot)); }

  // Warmup() starts or ends: its Runs carry ORT_QMOE_IN_WARMUP. From the first warmup on, every Run of this engine
  // asks for deterministic MoE tactics (no GEMM profiling, which may not run while a graph is captured and would make
  // the tactics, hence the numerics, differ between Runs and processes) unless the process sets the variable itself.
  void SetWarmup(bool warmup) noexcept {
    warmup_ = warmup;
    if (warmup) deterministic_moe_ = true;
  }
  bool InWarmup() const noexcept { return warmup_; }

  // Every Run of this engine asks for deterministic MoE tactics from now on, without being a warmup Run: the DFlash
  // pipeline verifies at S=8 under CUDA graphs from its first decode on, whether or not the model warms up.
  void RequireDeterministicMoe() noexcept { deterministic_moe_ = true; }

  // The variables to publish around the next Run, which serves `participants`: the pending notices of those slots only
  // (a notice that several of them have is published once), and `prompt` for the Runs that ingest prompt tokens.
  std::vector<Qwen38OrtEnvVariable> Collect(const Qwen38PromptRunHints* prompt,
                                            Qwen38SlotMask participants = Qwen38SlotBit(0)) const;

  // A Run that served `participants` returned. A one-shot notice that ORT did not reset was not consumed (the session
  // has no QMoE layer that reads it): it stays pending for the next Run that serves the slot, as the process-wide
  // variable it replaces did. The notices of slots that did not take part are never touched.
  void RunReturned(const Qwen38OrtEnvScope& scope, Qwen38SlotMask participants = Qwen38SlotBit(0));

 private:
  static constexpr size_t kMaxSlots = 8;

  struct SlotHints {
    bool new_turn{};
    bool handoff_done{};
    size_t expected_prompt_tokens{};
  };

  static void BeginTurn(SlotHints& slot) noexcept {
    slot.new_turn = true;
    slot.handoff_done = false;
    slot.expected_prompt_tokens = 0;
  }
  static size_t BeginPromptChunk(SlotHints& slot, size_t ingested_tokens, size_t tokens) noexcept {
    if (ingested_tokens == 0) slot.new_turn = true;
    slot.handoff_done = false;
    return std::max(ingested_tokens + tokens, slot.expected_prompt_tokens);
  }
  static void MarkHandoff(SlotHints& slot) noexcept {
    slot.handoff_done = true;
    slot.expected_prompt_tokens = 0;
  }

  SlotHints& At(size_t slot) {
    if (slot >= slots_.size()) throw std::out_of_range("qwen38_graph: run hints slot index out of range");
    return slots_[slot];
  }
  const SlotHints& At(size_t slot) const {
    if (slot >= slots_.size()) throw std::out_of_range("qwen38_graph: run hints slot index out of range");
    return slots_[slot];
  }
  void CheckParticipants(Qwen38SlotMask participants) const {
    if (participants == 0 || (participants >> slots_.size()) != 0) {
      throw std::out_of_range("qwen38_graph: a Run serves slots the run hints do not know");
    }
  }

  std::vector<SlotHints> slots_;
  bool warmup_{};
  bool deterministic_moe_{};
};

// Runs `run` (one ORT Run) with the hints published and the environment restored afterwards. A Run that throws leaves
// the one-shot notices pending.
template <typename RunFn>
void RunWithQwen38OrtHints(Qwen38OrtRunHints& hints, const Qwen38PromptRunHints* prompt, RunFn&& run,
                           Qwen38SlotMask participants = Qwen38SlotBit(0)) {
  Qwen38OrtEnvScope scope(hints.Collect(prompt, participants));
  run();
  hints.RunReturned(scope, participants);
}

}  // namespace Generators
