// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../generators.h"
#include "step_plan.h"

namespace Generators {

class PagedCacheReservation;

struct ExecutionContext {
  explicit ExecutionContext(StepTransactionId transaction_id = 0,
                            const StepPlan* plan = nullptr)
      : transaction_id{transaction_id},
        plan{plan},
        run_options{OrtRunOptions::Create()} {}

  StepTransactionId transaction_id{};
  const StepPlan* plan{};
  PagedCacheReservation* cache_reservation{};
  std::unique_ptr<OrtRunOptions> run_options;
  size_t block_table_columns{};
  int graph_id{-1};
  bool graph_capture_eligible{};
  bool execution_started{};
  bool execution_completed{};
};

}  // namespace Generators
