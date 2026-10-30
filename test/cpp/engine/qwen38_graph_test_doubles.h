// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Generators::test {

// A scripted stand-in for the CUDA-graph decode loop. It reproduces the loop's observable contract without a model:
//
//  - the "model" is a function of the position: the greedy token at sequence position p is TokenAt(p), so a Request that
//    runs it sees the sequence prompt ++ TokenAt(H), TokenAt(H + 1), ... (H = prompt length) regardless of how the steps
//    were chunked, and re-ingesting a longer history (a continued turn) keeps producing TokenAt of the next position;
//  - the drafter is right on the cycles named by `accept_pattern` (cycled), so cycles commit 1 or 2 tokens; SetDrafts()
//    switches to the fold's multi-draft cycles (2..3 drafts per verify, any accepted prefix 0..drafts per cycle);
//  - `dense_prefill` models the fast prefill: chunks of any size advance the ledger at bank 0 (no S=2 pairs), the hand-off
//    to the decode state happens in BeginDecode(run_cycle = true) and only once all chunks arrived, `preferred_chunk` is
//    the chunk size the device asks the executor for;
//  - the pending token / in-flight cycle / ledger bookkeeping follows Qwen38GraphEngine, and misuse (a wait without a
//    cycle in flight, an advance of something that was not waited for, prefill after decoding started, ...) throws, so the
//    tests also prove the executor drives the device in a legal order;
//  - a cycle runs asynchronously: it is enqueued by BeginDecode/Advance and completes once something synchronizes the
//    stream (WaitCycle, Drain, CheckDecodeStatus, Reset). Like the real device, Drain() completes it without consuming its
//    record (the cycle stays "in flight" until WaitCycle), and CheckDecodeStatus() synchronizes itself and reports the
//    error bits accumulated by every completed cycle, consumed or not, until Reset() clears them. `fail_cycle` makes a cycle
//    accumulate such bits when it completes, so a failure only becomes visible after the stream was drained.
class ScriptedQwen38Device final : public Qwen38DecodeDevice {
 public:
  explicit ScriptedQwen38Device(int capacity = 68, bool mtp = true, size_t max_batch_size = 1)
      : capacity_{capacity}, mtp_{mtp}, max_batch_size_{max_batch_size}, ledger_{capacity} {
    slot_ledgers_.assign(max_batch_size_, Qwen38DecodeLedger{capacity});
    slot_ingested_.resize(max_batch_size_);
    slot_decoding_.assign(max_batch_size_, false);
    slot_pending_.assign(max_batch_size_, -1);
    slot_cycles_.assign(max_batch_size_, 0);
  }

  // ---- script ----
  std::vector<bool> accept_pattern{true, false};
  std::map<size_t, int32_t> overrides;  // position -> token (for example the EOS id)
  int32_t TokenAt(size_t position) const {
    const auto found = overrides.find(position);
    if (found != overrides.end()) return found->second;
    return static_cast<int32_t>(10 + (position * 7) % 40);  // never the EOS id 1 or a stop-string token (< 10)
  }

  bool dense_prefill{};
  size_t preferred_chunk{};

  // Called at the end of ExpectPrefillTokens(), that is between the publication of a turn's prompt length and its first
  // PrefillChunk(): a test that drives several Engines blocks here to interleave their owner threads the worst way.
  std::function<void(size_t)> on_expect_prefill;

  // Every verify proposes `drafts` tokens and accepts the next entry of `accepted_counts` (cycled, each 0..drafts) of them
  // in order, so the cycle commits that many tokens + 1 (the correction, or the bonus when all were accepted). The record
  // then reports them like the production device does, through `accepted_count`. Call it before the first Reset().
  void SetDrafts(int drafts, std::vector<int> accepted_counts) {
    if (drafts < 1 || drafts > 3) throw std::invalid_argument("a verify proposes 1..3 drafts");
    for (const int count : accepted_counts) {
      if (count < 0 || count > drafts) throw std::invalid_argument("a verify accepts 0..drafts drafts");
    }
    drafts_ = drafts;
    accepted_counts_ = std::move(accepted_counts);
    ledger_.SetNumBanks(drafts + 2);  // the bank ring of the fold: one more than the rows of a verify
  }

  // ---- observation ----
  std::vector<std::string> calls;      // "reset", "expect:<n>", "prefill:<n>", "begin:<run_cycle>", "wait", "advance:<more>", "drain"
  std::vector<int32_t> ingested;       // tokens received since the last reset, in order
  std::vector<size_t> prefill_totals;  // for every PrefillChunk, the prompt length the device was told to expect (0 = none)
  size_t cycles_enqueued{};
  size_t handoffs{};                           // dense prefill: hand-offs performed (one per BeginDecode with a cycle)
  size_t handoff_length{};                     // prompt length at the last hand-off
  size_t status_checks{};                      // CheckDecodeStatus calls (not in `calls`, which tests compare by suffix)
  std::vector<std::string> order;              // "drain", "status" and "reset" in the order the device saw them
  bool inflight() const { return inflight_; }  // a cycle was enqueued and its record was not consumed by WaitCycle
  bool running() const { return running_; }    // ... and it has not completed on the device yet (nothing synchronized)
  size_t ledger_length() const { return ledger_.Length(); }
  int ledger_bank() const { return ledger_.Bank(); }
  size_t CallCount(const std::string& name) const { return std::count(calls.begin(), calls.end(), name); }
  const std::vector<int32_t>& slot_ingested(size_t s) const { return slot_ingested_.at(s); }

  // ---- fault injection ----
  bool fail_wait{};
  bool malformed_cycle_count{};
  bool context_healthy{true};
  bool fail_status_check{};  // every CheckDecodeStatus() of a started decode reports error bits
  size_t fail_cycle{};       // 1-based cycle of a decode that accumulates error bits when it completes (0 = none)

  // ---- Qwen38DecodeDevice ----
  void MarkContextFailed() noexcept override { context_healthy = false; }
  bool ContextIsHealthy() const noexcept override { return context_healthy; }
  int Capacity() const override { return capacity_; }
  bool MtpEnabled() const override { return mtp_; }
  size_t MaxBatchSize() const override { return max_batch_size_; }

  void Reset() override {
    calls.push_back("reset");
    order.push_back("reset");
    Complete();  // Reset() waits for every enqueued cycle...
    ledger_.Reset();
    ingested.clear();
    inflight_ = false;
    decoding_ = false;
    pending_ = -1;
    cycle_ = 0;
    expected_prefill_ = 0;
    error_bits_ = 0;  // ...then clears the device state, error bits included
    for (size_t s = 0; s < max_batch_size_; ++s) {
      slot_ledgers_[s].Reset();
      slot_ingested_[s].clear();
      slot_decoding_[s] = false;
      slot_pending_[s] = -1;
      slot_cycles_[s] = 0;
    }
  }

  void ResetSlot(size_t slot_idx) override {
    calls.push_back("reset_slot:" + std::to_string(slot_idx));
    slot_ledgers_.at(slot_idx).Reset();
    slot_ingested_.at(slot_idx).clear();
    slot_decoding_.at(slot_idx) = false;
    slot_pending_.at(slot_idx) = -1;
    slot_cycles_.at(slot_idx) = 0;
    if (slot_idx == 0) {
      ledger_.Reset();
      ingested.clear();
      decoding_ = false;
      pending_ = -1;
      cycle_ = 0;
    }
  }

  void Drain() noexcept override {
    calls.push_back("drain");
    order.push_back("drain");
    Complete();  // the cycle in flight finishes, but its record stays unconsumed: inflight_ is not cleared
  }

  void ExpectPrefillTokens(size_t total_tokens) override {
    calls.push_back("expect:" + std::to_string(total_tokens));
    if (decoding_ || ledger_.Length() != 0) throw std::logic_error("the prompt length was announced after its ingestion started");
    expected_prefill_ = total_tokens;
    if (on_expect_prefill) on_expect_prefill(total_tokens);
  }

  void PrefillChunk(std::span<const int32_t> tokens) override {
    calls.push_back("prefill:" + std::to_string(tokens.size()));
    if (decoding_) throw std::logic_error("prefill after decoding started");
    if (tokens.empty()) throw std::logic_error("empty prefill chunk");
    prefill_totals.push_back(expected_prefill_);
    ledger_.RequireRoom(tokens.size(), "the prompt", 3);
    if (dense_prefill) {
      ledger_.AdvancePrompt(tokens.size());
    } else {
      for (size_t i = 0; i < tokens.size();) {
        const size_t step = i + 1 < tokens.size() ? 2 : 1;
        ledger_.Advance(step);
        i += step;
      }
    }
    ingested.insert(ingested.end(), tokens.begin(), tokens.end());
    pending_ = TokenAt(ledger_.Length());
  }

  void PrefillChunkSlot(size_t slot_idx, std::span<const int32_t> tokens) override {
    calls.push_back("prefill_slot:" + std::to_string(slot_idx) + ":" + std::to_string(tokens.size()));
    if (slot_decoding_.at(slot_idx)) throw std::logic_error("prefill after decoding started");
    if (tokens.empty()) throw std::logic_error("empty prefill chunk");
    auto& sl = slot_ledgers_.at(slot_idx);
    sl.RequireRoom(tokens.size(), "the prompt", 3);
    if (dense_prefill) {
      sl.AdvancePrompt(tokens.size());
    } else {
      for (size_t i = 0; i < tokens.size();) {
        const size_t step = i + 1 < tokens.size() ? 2 : 1;
        sl.Advance(step);
        i += step;
      }
    }
    slot_ingested_.at(slot_idx).insert(slot_ingested_.at(slot_idx).end(), tokens.begin(), tokens.end());
    slot_pending_.at(slot_idx) = TokenAt(sl.Length());
    if (slot_idx == 0) {
      ledger_ = sl;
      ingested = slot_ingested_[0];
      pending_ = slot_pending_[0];
    }
  }

  int32_t PendingToken() const override { return pending_; }
  int32_t PendingTokenSlot(size_t slot_idx) const override { return slot_pending_.at(slot_idx); }
  size_t PrefillChunkTokens() const override { return preferred_chunk; }

  void BeginDecode(bool run_cycle) override {
    calls.push_back(std::string("begin:") + (run_cycle ? "1" : "0"));
    if (decoding_) throw std::logic_error("decoding already started");
    if (ledger_.Length() == 0) throw std::logic_error("BeginDecode before the prompt was ingested");
    decoding_ = true;
    expected_prefill_ = 0;
    if (run_cycle && dense_prefill) {
      if (ingested.size() != ledger_.Length()) throw std::logic_error("hand-off before every chunk was ingested");
      ++handoffs;
      handoff_length = ledger_.Length();
    }
    if (run_cycle) Enqueue();
  }

  void BeginDecodeSlot(size_t slot_idx, bool want_more) override {
    calls.push_back("begin_slot:" + std::to_string(slot_idx) + ":" + (want_more ? "1" : "0"));
    if (slot_decoding_.at(slot_idx)) throw std::logic_error("decoding already started");
    if (slot_ledgers_.at(slot_idx).Length() == 0) throw std::logic_error("BeginDecodeSlot before prompt was ingested");
    slot_decoding_.at(slot_idx) = true;
    if (slot_idx == 0) decoding_ = true;
    if (want_more && dense_prefill) {
      ++handoffs;
      handoff_length = slot_ledgers_.at(slot_idx).Length();
    }
  }

  void EnqueueBatchDecode(std::span<const size_t> active_slots) override {
    if (active_slots.empty()) return;
    calls.push_back("enqueue_batch:" + std::to_string(active_slots.size()));
    for (size_t s : active_slots) {
      slot_ledgers_.at(s).RequireRoom(2, "the next verify step");
      ++slot_cycles_.at(s);
    }
    inflight_ = true;
    ++cycles_enqueued;
  }

  void WaitBatchDecode(std::span<const size_t> active_slots,
                       std::span<Qwen38CycleRecord> records) override {
    calls.push_back("wait_batch:" + std::to_string(active_slots.size()));
    if (fail_wait) throw std::runtime_error("injected device failure");
    if (!inflight_) throw std::logic_error("no batch cycle in flight");
    inflight_ = false;
    for (size_t i = 0; i < active_slots.size(); ++i) {
      const size_t s = active_slots[i];
      const size_t length = slot_ledgers_.at(s).Length();
      Qwen38CycleRecord record;
      record.pred0 = TokenAt(length + 1);
      record.accepted = mtp_ && accept_pattern[(slot_cycles_.at(s) - 1) % accept_pattern.size()];
      record.pred1 = record.accepted ? TokenAt(length + 2) : -7;
      const size_t step = record.TokenCount();
      record.new_pending = record.accepted ? record.pred1 : record.pred0;
      record.new_length = static_cast<int64_t>(length + step);
      records[i] = record;
    }
  }

  void AdvanceBatchDecode(std::span<const size_t> active_slots,
                          std::span<const Qwen38CycleRecord> records,
                          std::span<const uint8_t> want_more,
                          bool defer_next_cycle) override {
    if (inflight_) throw std::logic_error("advance with a cycle still in flight");
    std::vector<size_t> next_slots;
    for (size_t i = 0; i < active_slots.size(); ++i) {
      const size_t s = active_slots[i];
      slot_ledgers_.at(s).Advance(records[i].TokenCount());
      slot_ledgers_.at(s).CheckDeviceLength(records[i].new_length);
      slot_pending_.at(s) = static_cast<int32_t>(records[i].new_pending);
      if (s == 0) {
        ledger_ = slot_ledgers_[0];
        pending_ = slot_pending_[0];
      }
      if (want_more[i] != 0) {
        next_slots.push_back(s);
      }
    }
    calls.push_back("advance_batch:" + std::to_string(next_slots.size()));
    if (!defer_next_cycle && !next_slots.empty()) {
      EnqueueBatchDecode(next_slots);
    }
  }

  Qwen38CycleRecord WaitCycle() override {
    calls.push_back("wait");
    if (fail_wait) throw std::runtime_error("injected device failure");
    if (!inflight_) throw std::logic_error("no cycle in flight");
    Complete();
    inflight_ = false;
    if (fail_cycle != 0 && cycle_ == fail_cycle) throw std::runtime_error("injected device validation failure");
    Qwen38CycleRecord record;
    if (malformed_cycle_count) {
      record.accepted_count = 100;
      record.proposed_count = 100;
      return record;
    }
    const size_t length = ledger_.Length();
    size_t accepted_drafts = 0;
    if (accepted_counts_.empty()) {
      record.accepted = mtp_ && accept_pattern[(cycle_ - 1) % accept_pattern.size()];  // the legacy single-draft flag
      accepted_drafts = record.accepted ? 1 : 0;
    } else {
      accepted_drafts = mtp_ ? static_cast<size_t>(accepted_counts_[(cycle_ - 1) % accepted_counts_.size()]) : 0;
      record.accepted = accepted_drafts > 0;
      record.accepted_count = static_cast<int>(accepted_drafts);
    }
    record.proposed_count = mtp_ ? drafts_ : 1;
    // Rows after the first mismatch are meaningless; the cycle commits the accepted drafts and one token more.
    const size_t step = accepted_drafts + 1;
    const std::array<int64_t*, 4> predictions{&record.pred0, &record.pred1, &record.pred2, &record.pred3};
    for (size_t i = 0; i < predictions.size(); ++i) {
      *predictions[i] = i < step ? TokenAt(length + 1 + i) : -7;
    }
    if (record.TokenCount() != step) throw std::logic_error("the scripted record disagrees with its own token count");
    record.new_pending = *predictions[step - 1];
    record.new_length = static_cast<int64_t>(length + step);
    return record;
  }

  void Advance(const Qwen38CycleRecord& record, bool want_more) override {
    calls.push_back(std::string("advance:") + (want_more ? "1" : "0"));
    if (inflight_) throw std::logic_error("advance with a cycle still in flight");
    ledger_.Advance(record.TokenCount());
    ledger_.CheckDeviceLength(record.new_length);
    pending_ = static_cast<int32_t>(record.new_pending);
    if (want_more) Enqueue();
  }

  void CheckDecodeStatus() override {
    ++status_checks;
    order.push_back("status");
    if (!decoding_) return;  // like the real device: nothing to check before the decode started
    Complete();              // it synchronizes the stream itself, also while a cycle is in flight or its record unconsumed
    if (fail_status_check || error_bits_ != 0) throw std::runtime_error("injected device error bits");
  }

 private:
  void Enqueue() {
    ledger_.RequireRoom(static_cast<size_t>(drafts_) + 1, "the next verify step");
    inflight_ = true;
    running_ = true;
    ++cycle_;
    ++cycles_enqueued;
  }

  // The stream was synchronized: the cycle that was running completes and leaves its validation bits behind.
  void Complete() {
    if (!running_) return;
    running_ = false;
    if (fail_cycle != 0 && cycle_ == fail_cycle) error_bits_ |= 2;
  }

  int capacity_;
  bool mtp_;
  int drafts_{1};
  std::vector<int> accepted_counts_;
  size_t max_batch_size_{1};
  Qwen38DecodeLedger ledger_;
  std::vector<Qwen38DecodeLedger> slot_ledgers_;
  std::vector<std::vector<int32_t>> slot_ingested_;
  std::vector<bool> slot_decoding_;
  std::vector<int32_t> slot_pending_;
  std::vector<size_t> slot_cycles_;
  bool inflight_{};
  bool running_{};
  bool decoding_{};
  int32_t pending_{-1};
  size_t cycle_{};
  int64_t error_bits_{};
  size_t expected_prefill_{};  // ExpectPrefillTokens: device-local, dropped by Reset() and BeginDecode()
};

}  // namespace Generators::test
