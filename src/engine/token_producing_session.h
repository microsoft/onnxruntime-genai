// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <atomic>
#include <limits>
#include <stdexcept>
#include <vector>

#include "executor_result.h"

namespace Generators {

// Owner-thread host bookkeeping for token-producing backends. It contains no device state or graph layout.
// An identity namespace is never reused across Engine leases, including when local submission counters restart.
class TokenProducingSession {
 public:
  explicit TokenProducingSession(size_t slots) : context_id_{NextIdentity()}, slots_(slots) {
    retired_.reserve(slots);
  }

  BackendPlannedRow Plan(size_t request_row, size_t slot, const void* owner, uint64_t turn_id,
                         size_t max_prompt_tokens, bool allow_phase_completion) {
    auto& state = slots_.at(slot);
    for (const auto& retired : retired_) {
      if (retired.slot.index == slot) throw std::logic_error("A retiring backend slot is not physically reusable.");
    }
    if (state.pending) throw std::logic_error("A backend completion must be settled before another submission.");
    if (state.owner != owner || state.turn_id != turn_id) {
      Retire(slot);
      state.owner = owner;
      state.turn_id = turn_id;
    }
    if (!owner) throw std::logic_error("A backend submission requires a live owner.");
    if (state.submission == std::numeric_limits<uint64_t>::max()) throw std::overflow_error("Backend submission identity exhausted.");
    state.pending = true;
    state.key = CompletionKey{context_id_, 1, SlotHandle{static_cast<uint32_t>(slot), state.epoch}, turn_id, ++state.submission};
    return BackendPlannedRow{request_row, state.key, max_prompt_tokens, allow_phase_completion};
  }

  // False means a tracked retiring submission settled without public delivery.
  bool Collected(const BackendRowResult& row) {
    for (auto it = retired_.begin(); it != retired_.end(); ++it) {
      if (*it == row.key) {
        retired_.erase(it);
        return false;
      }
    }
    auto& state = RequirePending(row.key);
    if (state.collected) throw std::logic_error("Duplicate backend completion.");
    state.collected = true;
    state.candidates = row.selected_tokens ? row.selected_tokens->candidate_count : 0;
    return true;
  }

  void Acknowledge(const TokenCommitAck& ack) {
    auto& state = RequirePending(ack.key);
    if (!state.collected || ack.candidates_consumed > state.candidates ||
        ack.tokens_appended > ack.candidates_consumed || ack.tokens_staged_for_events > ack.tokens_appended ||
        (ack.candidates_consumed != state.candidates && ack.disposition == ContinuationDisposition::Continue) ||
        (ack.turn_finished && ack.disposition == ContinuationDisposition::Continue)) {
      throw std::logic_error("Invalid backend commit acknowledgement.");
    }
    state.pending = false;
    state.collected = false;
    state.candidates = 0;
  }

  void Retire(size_t slot) {
    auto& state = slots_.at(slot);
    if (state.epoch == std::numeric_limits<uint64_t>::max()) throw std::overflow_error("Backend slot epoch exhausted.");
    if (state.pending && !state.collected) {
      if (retired_.size() >= slots_.size()) throw std::logic_error("Backend retirement capacity exhausted.");
      retired_.push_back(state.key);
    }
    const uint64_t epoch = state.epoch + 1;
    state = {};
    state.epoch = epoch;
  }

 private:
  struct Slot {
    const void* owner{};
    uint64_t turn_id{};
    uint64_t epoch{};
    uint64_t submission{};
    CompletionKey key{};
    bool pending{};
    bool collected{};
    size_t candidates{};
  };

  static uint64_t NextIdentity() {
    static std::atomic<uint64_t> next{1};
    const uint64_t id = next.fetch_add(1, std::memory_order_relaxed);
    if (id == 0) throw std::overflow_error("Backend context identity exhausted.");
    return id;
  }

  Slot& RequirePending(const CompletionKey& key) {
    auto& state = slots_.at(key.slot.index);
    if (!state.pending || !(state.key == key)) throw std::logic_error("Unknown or stale backend completion identity.");
    return state;
  }

  uint64_t context_id_;
  std::vector<Slot> slots_;
  std::vector<CompletionKey> retired_;
};

}  // namespace Generators
