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
  // Optional packed int32 token ids already resident on the model device. VarlenDecoderIO casts
  // these directly into its int64 input tensor without reading a request's host token mirror.
  DeviceSpan<int32_t> input_ids;
  // Optional packed [token_count, hidden_size] input supplied by an auxiliary decoder driver.
  // Ordinary decoder steps leave this null.
  OrtValue* hidden_states_input{};
  std::unique_ptr<OrtRunOptions> run_options;
  size_t block_table_columns{};
};

}  // namespace Generators
