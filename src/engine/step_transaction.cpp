// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "step_transaction.h"

namespace Generators {

StepTransaction::StepTransaction(const StepPlan& plan)
    : plan_{plan} {}

bool StepTransaction::IsResolved() const {
  return phase_ == StepTransactionPhase::Committed ||
         phase_ == StepTransactionPhase::RolledBack;
}

void StepTransaction::MarkReserved() {
  Transition(StepTransactionPhase::Planned, StepTransactionPhase::Reserved);
}

void StepTransaction::MarkExecuting() {
  Transition(StepTransactionPhase::Reserved, StepTransactionPhase::Executing);
}

void StepTransaction::MarkExecuted() {
  Transition(StepTransactionPhase::Executing, StepTransactionPhase::Executed);
}

void StepTransaction::Commit() {
  Transition(StepTransactionPhase::Executed, StepTransactionPhase::Committed);
}

void StepTransaction::RollBack() {
  if (phase_ == StepTransactionPhase::Committed) {
    throw std::logic_error("Cannot roll back a committed step transaction.");
  }
  phase_ = StepTransactionPhase::RolledBack;
}

void StepTransaction::Transition(StepTransactionPhase expected, StepTransactionPhase next) {
  if (phase_ != expected) {
    throw std::logic_error("Invalid step transaction phase transition.");
  }
  phase_ = next;
}

EngineStepError::EngineStepError(StepOutcome outcome, std::string message)
    : std::runtime_error{std::move(message)},
      outcome_{outcome} {}

}  // namespace Generators
