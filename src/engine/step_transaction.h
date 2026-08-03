// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdexcept>
#include <string>

#include "step_plan.h"

namespace Generators {

enum class StepTransactionPhase {
  Planned,
  Reserved,
  Executing,
  Executed,
  Committed,
  RolledBack,
};

enum class EngineHealth {
  Healthy,
  Unhealthy,
};

class StepTransaction {
 public:
  explicit StepTransaction(const StepPlan& plan);
  StepTransaction(StepPlan&&) = delete;

  const StepPlan& Plan() const { return plan_; }
  StepTransactionPhase Phase() const { return phase_; }
  bool IsResolved() const;

  void MarkReserved();
  void MarkExecuting();
  void MarkExecuted();
  void Commit();
  void RollBack();

 private:
  void Transition(StepTransactionPhase expected, StepTransactionPhase next);

  const StepPlan& plan_;
  StepTransactionPhase phase_{StepTransactionPhase::Planned};
};

class EngineStepError : public std::runtime_error {
 public:
  EngineStepError(StepOutcome outcome, std::string message);

  const StepOutcome& Outcome() const { return outcome_; }

 private:
  StepOutcome outcome_;
};

}  // namespace Generators
