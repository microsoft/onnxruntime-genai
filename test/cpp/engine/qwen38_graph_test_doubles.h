// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "models/qwen38_decode_device.h"

namespace Generators::test {

// A scripted stand-in for the CUDA-graph decode loop. It reproduces the loop's observable contract without a model:
//
//  - the "model" is a function of the position: the greedy token at sequence position p is TokenAt(p), so a Request that
//    runs it sees the sequence prompt ++ TokenAt(H), TokenAt(H + 1), ... (H = prompt length) regardless of how the steps
//    were chunked, and re-ingesting a longer history (a continued turn) keeps producing TokenAt of the next position;
//  - the drafter is right on the cycles named by `accept_pattern` (cycled), so cycles commit 1 or 2 tokens;
//  - `dense_prefill` models the fast prefill: chunks of any size advance the ledger at bank 0 (no S=2 pairs), the hand-off
//    to the decode state happens in BeginDecode(run_cycle = true) and only once all chunks arrived, `preferred_chunk` is
//    the chunk size the device asks the executor for;
//  - the pending token / in-flight cycle / ledger bookkeeping follows Qwen38GraphEngine, and misuse (a wait without a
//    cycle in flight, an advance of something that was not waited for, prefill after decoding started, ...) throws, so the
//    tests also prove the executor drives the device in a legal order.
class ScriptedQwen38Device final : public Qwen38DecodeDevice {
 public:
  explicit ScriptedQwen38Device(int capacity = 68, bool mtp = true) : capacity_{capacity}, mtp_{mtp}, ledger_{capacity} {}

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

  // ---- observation ----
  std::vector<std::string> calls;  // "reset", "prefill:<n>", "begin:<run_cycle>", "wait", "advance:<more>", "drain"
  std::vector<int32_t> ingested;   // tokens received since the last reset, in order
  size_t cycles_enqueued{};
  size_t handoffs{};        // dense prefill: hand-offs performed (one per BeginDecode with a cycle)
  size_t handoff_length{};  // prompt length at the last hand-off
  size_t status_checks{};   // CheckDecodeStatus calls (not in `calls`, which tests compare by suffix)
  bool inflight() const { return inflight_; }
  size_t ledger_length() const { return ledger_.Length(); }
  int ledger_bank() const { return ledger_.Bank(); }
  size_t CallCount(const std::string& name) const { return std::count(calls.begin(), calls.end(), name); }

  // ---- fault injection ----
  bool fail_wait{};
  bool fail_status_check{};

  // ---- Qwen38DecodeDevice ----
  int Capacity() const override { return capacity_; }
  bool MtpEnabled() const override { return mtp_; }

  void Reset() override {
    calls.push_back("reset");
    ledger_.Reset();
    ingested.clear();
    inflight_ = false;
    decoding_ = false;
    pending_ = -1;
    cycle_ = 0;
  }

  void Drain() noexcept override {
    calls.push_back("drain");
    inflight_ = false;
  }

  void PrefillChunk(std::span<const int32_t> tokens) override {
    calls.push_back("prefill:" + std::to_string(tokens.size()));
    if (decoding_) throw std::logic_error("prefill after decoding started");
    if (tokens.empty()) throw std::logic_error("empty prefill chunk");
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

  int32_t PendingToken() const override { return pending_; }
  size_t PrefillChunkTokens() const override { return preferred_chunk; }

  void BeginDecode(bool run_cycle) override {
    calls.push_back(std::string("begin:") + (run_cycle ? "1" : "0"));
    if (decoding_) throw std::logic_error("decoding already started");
    if (ledger_.Length() == 0) throw std::logic_error("BeginDecode before the prompt was ingested");
    decoding_ = true;
    if (run_cycle && dense_prefill) {
      if (ingested.size() != ledger_.Length()) throw std::logic_error("hand-off before every chunk was ingested");
      ++handoffs;
      handoff_length = ledger_.Length();
    }
    if (run_cycle) Enqueue();
  }

  Qwen38CycleRecord WaitCycle() override {
    calls.push_back("wait");
    if (fail_wait) throw std::runtime_error("injected device failure");
    if (!inflight_) throw std::logic_error("no cycle in flight");
    inflight_ = false;
    Qwen38CycleRecord record;
    const size_t length = ledger_.Length();
    record.pred0 = TokenAt(length + 1);
    record.accepted = mtp_ && accept_pattern[(cycle_ - 1) % accept_pattern.size()];
    record.pred1 = record.accepted ? TokenAt(length + 2) : -7;  // a rejected draft leaves row 1 meaningless
    const size_t step = record.TokenCount();
    record.new_pending = record.accepted ? record.pred1 : record.pred0;
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
    if (inflight_) throw std::logic_error("status check with a cycle in flight");
    ++status_checks;
    if (fail_status_check) throw std::runtime_error("injected device error bits");
  }

 private:
  void Enqueue() {
    ledger_.RequireRoom(2, "the next verify step");
    inflight_ = true;
    ++cycle_;
    ++cycles_enqueued;
  }

  int capacity_;
  bool mtp_;
  Qwen38DecodeLedger ledger_;
  bool inflight_{};
  bool decoding_{};
  int32_t pending_{-1};
  size_t cycle_{};
};

}  // namespace Generators::test
