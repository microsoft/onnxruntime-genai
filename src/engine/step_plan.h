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

enum class StepOutcomeKind {
  NoWork,
  CapacityDeferred,
  UnserviceableRequest,
  Committed,
  RetryableBatchAbort,
  ExecutionContractFailure,
  FatalExecutionFailure,
};

enum class ExecutionFailureKind {
  RetryableAbort,
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
  int64_t sequence_length_before{};
  size_t unprocessed_token_count{};
  size_t packed_token_offset{};
  size_t logits_row_index{};
  size_t target_cache_slots{};
  bool is_prefill{};
  bool newly_admitted{};
};

struct StepPlan {
  StepTransactionId transaction_id{};
  std::vector<RequestStepPlan> requests;
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
