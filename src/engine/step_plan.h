// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "request_status.h"

namespace Generators {

struct Request;

using StepTransactionId = uint64_t;

enum class StepOutcomeKind {
  NoWork,
  CapacityDeferred,
  UnserviceableRequest,
  Committed,
  RetryableBatchAbort,
  InvalidRequest,
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
  StepOutcome terminal_outcome;
};

struct RequestStepPlan {
  std::shared_ptr<Request> request;
  const void* request_id{};
  RequestStatus status_before{RequestStatus::Unassigned};
  int64_t sequence_length_before{};
  int64_t processed_sequence_length_before{};
  int64_t seen_sequence_length_before{};
  int64_t processed_sequence_length_after{};
  size_t unprocessed_token_offset{};
  size_t unprocessed_token_count{};
  size_t packed_token_offset{};
  size_t logits_row_index{};
  size_t committed_cache_slots{};
  size_t committed_block_count{};
  size_t target_cache_slots{};
  size_t tail_slots_to_consume{};
  size_t new_blocks_required{};
  bool is_prefill{};
  bool newly_admitted{};
};

struct StepPlan {
  StepTransactionId transaction_id{};
  std::vector<RequestStepPlan> requests;
  size_t prompt_token_count{};
  size_t decode_token_count{};
  size_t proposed_block_table_columns{};
  bool graph_capture_eligible{};
  bool capacity_deferred{};
  const void* unserviceable_request_id{};

  bool Empty() const { return requests.empty(); }
};

}  // namespace Generators
