// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "request_status.h"
#include "step_plan.h"

namespace Generators {

enum class StepOutputKind { DecoderOutputs,
                            BackendRows };

struct SlotHandle {
  uint32_t index{};
  uint64_t epoch{};
  bool operator==(const SlotHandle& other) const { return index == other.index && epoch == other.epoch; }
};

struct CompletionKey {
  uint64_t context_id{};
  uint64_t lease_epoch{};
  SlotHandle slot;
  uint64_t turn_id{};
  uint64_t submission_id{};
  bool operator==(const CompletionKey& other) const {
    return context_id == other.context_id && lease_epoch == other.lease_epoch &&
           slot == other.slot && turn_id == other.turn_id && submission_id == other.submission_id;
  }
};

// Raw completed backend work, distinct from API-visible speculative accounting. Decode target forwards
// count once per underlying batch, even when every row's delivery is discarded. Prefill work is tokens
// actually reprocessed, including replay; no drafter-forward count is inferred from proposal width.
struct BackendWorkMetrics {
  uint64_t work_records{};
  uint64_t target_forward_passes{};
  uint64_t prefill_tokens{};
  uint64_t discarded_completions{};
};

struct SpeculativeRowMetrics {
  size_t proposed{};
  size_t accepted{};
};

// The reference keeps backend continuation resources alive through settlement. Engine never
// interprets its payload and does not assume that it is a restorable state snapshot.
using BackendContinuationCheckpoint = std::shared_ptr<const void>;

struct SelectedTokenBlock {
  std::array<int32_t, kMaxGeneratedTokensPerStep> candidates{};
  size_t candidate_count{};
  SpeculativeRowMetrics speculation;
  BackendContinuationCheckpoint checkpoint;
};

struct PrefillProgress {
  size_t prompt_tokens_processed_this_step{};
  bool prompt_complete{};
};

struct BackendRowResult {
  CompletionKey key;
  std::optional<PrefillProgress> prefill;
  std::optional<SelectedTokenBlock> selected_tokens;
};

// Only participating rows appear in this plan; an admitted resident may be idle at a phase barrier.
struct BackendPlannedRow {
  size_t request_row{};
  CompletionKey key;
  size_t max_prompt_tokens{};
  bool allow_phase_completion{};
};

enum class ContinuationDisposition { Continue,
                                     Invalidate,
                                     Release };

struct TokenCommitAck {
  CompletionKey key;
  size_t candidates_consumed{};
  size_t tokens_appended{};
  size_t tokens_staged_for_events{};
  uint64_t request_sequence_length_after{};
  bool turn_finished{};
  GenerationFinishReason finish_reason{GenerationFinishReason::None};
  ContinuationDisposition disposition{ContinuationDisposition::Continue};
};

}  // namespace Generators
