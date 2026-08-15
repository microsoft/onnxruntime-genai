// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Generators {

struct Request;

using StepTransactionId = uint64_t;

enum class StepOutcomeKind {
  NoWork,
  CapacityDeferred,
  UnserviceableRequest,
  Committed,
  RetryableBatchAbort,
  ExecutionCapacityExceeded,
  ExecutionContractFailure,
  FatalExecutionFailure,
};

enum class ExecutionFailureKind {
  RetryableAbort,
  CapacityExceeded,
  Unknown,
};

struct StepOutcome {
  StepOutcomeKind kind{StepOutcomeKind::NoWork};
  StepTransactionId transaction_id{};
  const void* request_id{};
};

// One request that free paged-cache blocks alone are holding up, and how many more it needs.
struct BlockCapacityShortfall {
  const void* request_id{};
  size_t blocks{};

  bool Any() const { return request_id != nullptr; }
};

struct StepPlanningResult {
  bool executable{};
  bool capacity_deferred{};
  const void* unserviceable_request_id{};
  // A request the planner would otherwise have run and deferred only because the block pool is
  // short of free blocks. Freeing that many blocks is exactly what would let it run, so it is the
  // demand a preemption decision has to cover. Row limits and residency limits are deliberately
  // excluded, because returning blocks to the pool does not relieve either of them.
  BlockCapacityShortfall blocked_admission;  // First waiting request short of blocks.
  BlockCapacityShortfall blocked_resident;   // First resident whose growth is short of blocks.
  StepOutcome outcome;
};

struct StepPlanningLimits {
  std::optional<size_t> max_scheduled_tokens;  // Temporary cap for this planning attempt.
  std::optional<size_t> max_prefill_requests;  // Decodes do not consume this row cap.
  // True when this is another attempt at the same engine step rather than a fresh one. The
  // scheduler keeps the requests it suspended earlier in the step out of admission, so a retry
  // cannot hand the reclaimed capacity straight back to the request the step just suspended.
  bool retry_of_current_step{};
};

struct RequestStepPlan {
  std::shared_ptr<Request> request;
  const void* request_id{};
  int64_t sequence_length_before{};     // Search length before this transaction appends a token.
  size_t unprocessed_token_count{};     // Prompt chunk or single decode token sent to the model.
  size_t packed_token_offset{};         // First row for this request in the flat varlen input.
  size_t logits_row_index{};            // Last packed row; its logits produce this request's next token.
  size_t target_cache_slots{};          // Committed KV slots required after the model run succeeds.
  size_t whole_sequence_cache_slots{};  // KV slots the whole sequence needs; admitted and reserved on this.
  bool is_prefill{};
  bool newly_admitted{};
};

struct StepPlan {
  StepTransactionId transaction_id{};
  std::vector<RequestStepPlan> requests;
  size_t scheduled_request_limit{};  // Provisional rows cache feasibility may select.
  // Cap on requests admitted to the cache by this plan; zero leaves it to the batch size. Used
  // after a preemption so the reclaimed capacity is not spent on more waiting work than the
  // preemption was measured for.
  size_t new_admission_limit{};
  size_t token_count{};
  size_t proposed_block_table_columns{};
  bool graph_capture_eligible{};

  bool Empty() const { return requests.empty(); }
};

class EngineStepError : public std::runtime_error {
 public:
  EngineStepError(StepOutcome outcome, std::string message)
      : std::runtime_error{std::move(message)},
        outcome_{outcome} {}

  const StepOutcome& Outcome() const { return outcome_; }

 private:
  StepOutcome outcome_;
};

}  // namespace Generators
