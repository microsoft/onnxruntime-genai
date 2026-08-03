// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "engine/model_executor.h"
#include "engine/step_transaction.h"

namespace Generators {
namespace test {
namespace {

TEST(StepTransactionTest, SuccessfulTransactionFollowsExpectedPhases) {
  StepPlan plan;
  plan.transaction_id = 7;
  StepTransaction transaction{plan};

  EXPECT_EQ(transaction.Phase(), StepTransactionPhase::Planned);
  transaction.MarkReserved();
  transaction.MarkExecuting();
  transaction.MarkExecuted();
  transaction.Commit();

  EXPECT_EQ(transaction.Phase(), StepTransactionPhase::Committed);
  EXPECT_TRUE(transaction.IsResolved());
  EXPECT_EQ(transaction.Plan().transaction_id, 7u);
}

TEST(StepTransactionTest, RollbackResolvesAnyUncommittedPhase) {
  for (StepTransactionPhase target : {
           StepTransactionPhase::Planned,
           StepTransactionPhase::Reserved,
           StepTransactionPhase::Executing,
           StepTransactionPhase::Executed,
       }) {
    StepPlan plan;
    StepTransaction transaction{plan};
    if (target >= StepTransactionPhase::Reserved) transaction.MarkReserved();
    if (target >= StepTransactionPhase::Executing) transaction.MarkExecuting();
    if (target >= StepTransactionPhase::Executed) transaction.MarkExecuted();

    transaction.RollBack();
    transaction.RollBack();

    EXPECT_EQ(transaction.Phase(), StepTransactionPhase::RolledBack);
    EXPECT_TRUE(transaction.IsResolved());
  }
}

TEST(StepTransactionTest, InvalidPhaseTransitionIsRejected) {
  StepPlan plan;
  StepTransaction transaction{plan};

  EXPECT_THROW(transaction.MarkExecuting(), std::logic_error);
  EXPECT_EQ(transaction.Phase(), StepTransactionPhase::Planned);
}

TEST(StepTransactionTest, CommittedTransactionCannotRollBack) {
  StepPlan plan;
  StepTransaction transaction{plan};
  transaction.MarkReserved();
  transaction.MarkExecuting();
  transaction.MarkExecuted();
  transaction.Commit();

  EXPECT_THROW(transaction.RollBack(), std::logic_error);
  EXPECT_EQ(transaction.Phase(), StepTransactionPhase::Committed);
}

TEST(EngineStepErrorTest, PreservesStructuredOutcome) {
  const char request{};
  EngineStepError error{
      {
          StepOutcomeKind::RetryableBatchAbort,
          42,
          &request,
      },
      "execution aborted",
  };

  EXPECT_EQ(error.Outcome().kind, StepOutcomeKind::RetryableBatchAbort);
  EXPECT_EQ(error.Outcome().transaction_id, 42u);
  EXPECT_EQ(error.Outcome().request_id, &request);
  EXPECT_STREQ(error.what(), "execution aborted");
}

TEST(ModelExecutionErrorTest, PreservesFailureClassification) {
  ModelExecutionError error{ExecutionFailureKind::RetryableAbort, "retry"};

  EXPECT_EQ(error.FailureKind(), ExecutionFailureKind::RetryableAbort);
  EXPECT_STREQ(error.what(), "retry");
}

}  // namespace
}  // namespace test
}  // namespace Generators
