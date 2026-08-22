// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Generators {

struct Request;

using StepTransactionId = uint64_t;

// Upper bound on the speculative drafts a single request may attach to one decode step. The
// packed recurrent-state operators checkpoint at most eight tokens, so no cache can roll back
// further than a step of that length.
inline constexpr size_t kMaxDraftTokensPerStep = 7;

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

struct StepPlanningResult {
  bool executable{};
  bool capacity_deferred{};
  const void* unserviceable_request_id{};
  StepOutcome outcome;
};

struct RequestStepPlan {
  std::shared_ptr<Request> request;
  const void* request_id{};
  int64_t sequence_length_before{};     // Search length before this transaction appends a token.
  size_t unprocessed_token_count{};     // Prompt chunk or single decode token sent to the model.
  size_t draft_token_count{};           // Trailing speculative tokens of that count, verified this step.
  size_t packed_token_offset{};         // First row for this request in the flat varlen input.
  size_t logits_row_index{};            // Last packed row; its logits produce this request's next token.
  size_t target_cache_slots{};          // Committed KV slots required after the model run succeeds.
  size_t whole_sequence_cache_slots{};  // KV slots the whole sequence needs; admitted and reserved on this.
  bool is_prefill{};
  bool newly_admitted{};
};

// Fixed decoder-state demand for a step, planned atomically with the paged-block demand so the
// Engine can prove the reservation matches the plan before any state is published. Row order
// mirrors StepPlan::requests exactly. `required` is false (and every count zero) when the model has
// no fixed groups, which keeps the dense paged path unchanged.
struct FixedStateResourcePlan {
  bool required{};          // The plan needs fixed slots this step (the model has fixed groups).
  size_t row_count{};       // Fixed rows the reservation must expose, one per scheduled request.
  size_t new_slot_count{};  // Rows that admit a fresh request and consume a free fixed slot.
  size_t staging_bytes{};   // Gather+output staging bytes the reservation must allocate.
  bool capture_checkpoints{};  // Some request verifies drafts, so the step needs the state series.
};

struct StepPlan {
  StepTransactionId transaction_id{};
  std::vector<RequestStepPlan> requests;
  size_t scheduled_request_limit{};  // Provisional rows cache feasibility may select.
  size_t token_count{};
  size_t proposed_block_table_columns{};
  FixedStateResourcePlan fixed_state;
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
