// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Device-side contract of the Qwen3.8-Flash CUDA-graph decode loop, independent of CUDA, ORT and the Engine. The real
// implementation is Qwen38GraphEngine (qwen38_graph.h); the Engine-hosted executor (src/engine/qwen38_graph_executor.h)
// and its tests only see this interface.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "span.h"

namespace Generators {

// What one draft -> verify -> commit cycle published. The verify step ran the target on [pending token, draft0, ...,
// draft(R-2)], R = 2 .. 8 rows: pred0..pred7 are the target's greedy predictions at each verify row (those past row R - 1 are
// unused). The longest matching prefix of drafts is accepted (accepted_count in 0 .. R - 1; 1 + accepted_count tokens are
// committed, and the bank ring advances by 1 + accepted_count, or by one bank when the compact replay ping-pongs between two).
struct Qwen38CycleRecord {
  int64_t pred0{};
  int64_t pred1{};
  int64_t pred2{};
  int64_t pred3{};
  int64_t pred4{};
  int64_t pred5{};
  int64_t pred6{};
  int64_t pred7{};
  bool accepted{};
  int accepted_count{};
  int proposed_count{1};
  int64_t new_pending{};  // device view after the commit (checked against the host ledger)
  int64_t new_length{};

  size_t TokenCount() const {
    return accepted_count > 0 ? static_cast<size_t>(1 + accepted_count) : (accepted ? 2 : 1);
  }
};

struct Qwen38DecodeStats {
  size_t cycles{};
  size_t accepted{};
  size_t tokens{};
};

// The loop the Engine executor drives. Single-threaded: every method runs on the Engine owner thread.
class Qwen38DecodeDevice {
 public:
  virtual ~Qwen38DecodeDevice() = default;

  // Fatal execution invalidates mutable state across Engine leases; reset does not recover it.
  virtual void MarkContextFailed() noexcept {}
  virtual bool ContextIsHealthy() const noexcept { return true; }

  // Largest sequence (prompt + generated) the fixed KV store holds is MaxSequenceLength() = Capacity() - ReservedTail().
  virtual int Capacity() const = 0;
  virtual bool MtpEnabled() const = 0;

  // Waits for every enqueued cycle, then clears all state: the device is empty (length 0) afterwards.
  virtual void Reset() = 0;
  // Waits for every enqueued cycle without clearing anything; must not throw.
  virtual void Drain() noexcept = 0;

  // Ingests prompt tokens and leaves PendingToken() = the greedy token after the last ingested one. Consecutive calls
  // continue the same prompt; every chunk but the last has an even size unless PrefillChunkTokens() says otherwise. Only
  // valid before BeginDecode. Synchronous.
  virtual void PrefillChunk(std::span<const int32_t> tokens) = 0;
  virtual int32_t PendingToken() const = 0;
  // Size of the prompt chunks the device ingests most efficiently (the fast prefill keeps one dense chunk resident); 0 when
  // it has no preference. The executor then delivers prompts in chunks of this size, any chunk size being valid.
  virtual size_t PrefillChunkTokens() const { return 0; }
  // Announces the total prompt length (0 = unknown) of the turn that the following PrefillChunk() calls ingest in
  // chunks. Call it after Reset() and before the first chunk; Reset() and BeginDecode() drop it. The device hands it to
  // the ORT QMoE kernels with every prompt chunk instead of the length seen so far. This replaces publishing
  // ORT_QMOE_PREFILL_TOTAL_TOKENS from the caller: the device never reads that variable, and the caller never touches
  // the process environment.
  virtual void ExpectPrefillTokens(size_t total_tokens) { (void)total_tokens; }
  // The same for the turn of `slot` of a multi-slot device: the length is that slot's own state, so the prompts of
  // several slots that are ingested in interleaved chunks never see each other's length. Call it after ResetSlot(slot)
  // and before the first PrefillChunkSlot(slot, ...); ResetSlot(slot) and BeginDecodeSlot(slot, ...) drop it. The device
  // hands it to ORT only with the Runs that ingest a chunk of that slot. Slot 0 of a single-slot device is
  // ExpectPrefillTokens().
  virtual void ExpectPrefillTokensSlot(size_t slot, size_t total_tokens) {
    if (slot != 0) throw std::logic_error("Single-slot device only supports slot 0.");
    ExpectPrefillTokens(total_tokens);
  }

  // Starts decoding after the prompt: hands the prompt state over to the decode state (fast prefill), prepares the head and,
  // when `run_cycle`, enqueues the first cycle (so that it is running before the first token is returned to the caller).
  virtual void BeginDecode(bool run_cycle) = 0;
  // Spins on the pinned record of the cycle in flight. Throws when the device reports a validation failure.
  virtual Qwen38CycleRecord WaitCycle() = 0;
  // Commits `record` on the host and, when `want_more`, enqueues the next cycle (look-ahead).
  virtual void Advance(const Qwen38CycleRecord& record, bool want_more) = 0;
  // Throws when the device accumulated a validation error over the decode: from the target and, outside the fold
  // stack, from the standalone head, whose status the cycle record does not carry. Valid once BeginDecode ran, also
  // right after Drain() while the record of the last cycle is still unconsumed (cancellation, stop string): it
  // synchronizes the stream itself, so the error of a cycle nobody waited for is reported here instead of being
  // cleared by the next Reset() or ResetSlot(). A device with several slots reports the error of every slot that
  // started a decode, not only slot 0's; the executor calls it, after Drain(), before it resets a slot.
  virtual void CheckDecodeStatus() {}

  // Positions at the end of the KV store that are never filled: a verify step of R rows writes R rows past the committed length,
  // and no device keeps fewer than kReservedTail free. A device that verifies more rows reports a larger ReservedTail().
  static constexpr int kReservedTail = 4;
  virtual int ReservedTail() const { return kReservedTail; }
  // The tail can leave nothing (a store smaller than the rows one verify step writes): 0 then, never a wrapped negative count.
  size_t MaxSequenceLength() const {
    const int room = Capacity() - ReservedTail();
    return room > 0 ? static_cast<size_t>(room) : 0;
  }

  // Multi-slot / batched execution contract (defaults to single-slot slot 0).
  virtual size_t MaxBatchSize() const { return 1; }
  virtual bool SupportsDynamicBatching() const { return false; }

  virtual void ResetSlot(size_t slot) {
    if (slot != 0) throw std::logic_error("Single-slot device only supports slot 0.");
    Reset();
  }
  virtual void PrefillChunkSlot(size_t slot, std::span<const int32_t> tokens) {
    if (slot != 0) throw std::logic_error("Single-slot device only supports slot 0.");
    PrefillChunk(tokens);
  }
  virtual int32_t PendingTokenSlot(size_t slot) const {
    if (slot != 0) throw std::logic_error("Single-slot device only supports slot 0.");
    return PendingToken();
  }
  virtual void BeginDecodeSlot(size_t slot, bool want_more) {
    if (slot != 0) throw std::logic_error("Single-slot device only supports slot 0.");
    BeginDecode(want_more);
  }
  virtual void DraftDeferredSlots(std::span<const size_t> active_slots) {
    (void)active_slots;
  }
  virtual void EnqueueBatchDecode(std::span<const size_t> active_slots) {
    (void)active_slots;
  }
  virtual void WaitBatchDecode(std::span<const size_t> active_slots,
                               std::span<Qwen38CycleRecord> records) {
    if (active_slots.size() == 1 && active_slots[0] == 0 && !records.empty()) {
      records[0] = WaitCycle();
      return;
    }
    throw std::logic_error("Single-slot device does not implement WaitBatchDecode for multiple slots.");
  }
  virtual void AdvanceBatchDecode(std::span<const size_t> active_slots,
                                  std::span<const Qwen38CycleRecord> records,
                                  std::span<const uint8_t> want_more,
                                  bool defer_next_cycle = false) {
    (void)defer_next_cycle;
    if (active_slots.size() == 1 && active_slots[0] == 0 && !records.empty() && !want_more.empty()) {
      Advance(records[0], want_more[0] != 0);
      return;
    }
    throw std::logic_error("Single-slot device does not implement AdvanceBatchDecode for multiple slots.");
  }
};

// Host mirror of the device loop's committed state: how many tokens are committed and which recurrent bank of the ring the
// next target step reads. A target step reading bank k writes its committed state into bank (k + step) % num_banks (step = rows
// committed, 1 .. max_step), so a ring of max_step + 1 banks holds the bank being read, the mid-states of the step and the
// result (kBanks = 3 for the S=2 verify). The compact replay (SetCompactPingPong) instead reads the committed state of one bank
// and writes the state after the accepted rows into the other, so two banks that swap on every step suffice for any row count.
class Qwen38DecodeLedger {
 public:
  static constexpr int kBanks = 3;

  explicit Qwen38DecodeLedger(int capacity, int num_banks = kBanks)
      : capacity_{capacity}, num_banks_{num_banks}, max_step_{static_cast<size_t>(num_banks - 1)} {}

  void SetNumBanks(int num_banks) {
    num_banks_ = num_banks;
    max_step_ = static_cast<size_t>(num_banks - 1);
    ping_pong_ = false;
  }
  void SetCompactPingPong(bool enabled, int max_step) {
    ping_pong_ = enabled;
    num_banks_ = enabled ? 2 : (max_step + 1);
    max_step_ = static_cast<size_t>(max_step);
  }
  int NumBanks() const { return num_banks_; }

  void Reset() {
    bank_ = 0;
    length_ = 0;
  }

  int Bank() const { return bank_; }
  size_t Length() const { return length_; }

  void SetBank(int bank) {
    if (bank < 0 || bank >= num_banks_)
      throw std::out_of_range("qwen38_graph: invalid ledger bank " + std::to_string(bank));
    bank_ = bank;
  }

  // Commits `step` rows (1 .. max_step_).
  void Advance(size_t step) {
    if (step < 1 || step > max_step_)
      throw std::logic_error("qwen38_graph: a step commits 1 to " + std::to_string(max_step_) + " rows, not " +
                             std::to_string(step));
    bank_ = ping_pong_
                ? static_cast<int>((static_cast<size_t>(bank_) + 1) % static_cast<size_t>(num_banks_))
                : static_cast<int>((static_cast<size_t>(bank_) + step) % static_cast<size_t>(num_banks_));
    length_ += step;
  }

  // Commits `rows` prompt rows at once (the dense prefill: no bank rotation, the hand-off writes bank 0).
  void AdvancePrompt(size_t rows) {
    if (bank_ != 0) throw std::logic_error("qwen38_graph: a dense prompt chunk needs the committed bank 0");
    length_ += rows;
  }

  // Throws unless `rows` more positions fit. `tail` extra positions must stay free (see kReservedTail).
  void RequireRoom(size_t rows, const char* what, size_t tail = 0) const {
    if (length_ + rows + tail > static_cast<size_t>(capacity_))
      throw std::runtime_error(std::string("qwen38_graph: ") + what + " exceeds the KV capacity of " + std::to_string(capacity_));
  }

  // The device published its own committed length; it must equal the host's.
  void CheckDeviceLength(int64_t device_length) const {
    if (static_cast<int64_t>(length_) != device_length)
      throw std::runtime_error("qwen38_graph: device length " + std::to_string(device_length) + " differs from host length " +
                               std::to_string(length_));
  }

 private:
  int capacity_;
  int num_banks_{kBanks};
  size_t max_step_{kBanks - 1};
  bool ping_pong_{false};
  int bank_{};
  size_t length_{};
};

}  // namespace Generators
