// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../generators.h"
#include "step_plan.h"

namespace Generators {

class PagedCacheReservation;

struct ExecutionContext {
  explicit ExecutionContext(const StepPlan* plan = nullptr)
      : plan{plan},
        run_options{OrtRunOptions::Create()} {}

  const StepPlan* plan{};
  PagedCacheReservation* cache_reservation{};
  std::unique_ptr<OrtRunOptions> run_options;
  size_t block_table_columns{};
};

}  // namespace Generators
