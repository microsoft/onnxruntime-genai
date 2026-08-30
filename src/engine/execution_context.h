// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "generator/generators.h"
#include "fixed_state_pool.h"
#include "step_plan.h"

namespace Generators {

class PagedCacheReservation;

struct ExecutionContext {
  explicit ExecutionContext(const StepPlan* plan = nullptr)
      : plan{plan},
        run_options{OrtRunOptions::Create()} {}

  const StepPlan* plan{};
  PagedCacheReservation* cache_reservation{};
  // Fixed decoder-state resources for this step, in scheduled request row order. Empty when the
  // model has no fixed groups. HybridDecoderIO binds them alongside packed inputs. The reservation
  // owns the handles, bindings, names, and OrtValue objects; these views remain valid only for this
  // synchronous transaction.
  std::span<const FixedStateSlotHandle> fixed_state_slots;
  std::span<const FixedStateBinding> fixed_state_bindings;
  size_t fixed_state_staging_bytes{};
  std::unique_ptr<OrtRunOptions> run_options;
  size_t block_table_columns{};
};

}  // namespace Generators
