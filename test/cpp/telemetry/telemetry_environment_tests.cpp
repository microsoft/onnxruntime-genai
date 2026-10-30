// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scoped-environment-variable.h"
#include "telemetry/telemetry_environment.h"

#include <deque>
#include <atomic>
#include <chrono>
#include <future>
#include <shared_mutex>
#include <utility>

namespace {

using Generators::test::ScopedEnvironmentVariable;
using namespace Generators::TelemetryInternal;

class TelemetryEnvironmentTests : public ::testing::Test {
 protected:
  std::shared_mutex lifecycle_mutex_;

  void SetUp() override {
    for (const char* name : kTelemetrySuppressionVariables) {
      variables_.emplace_back(name, std::nullopt);
    }
    variables_.emplace_back("ORT_DISABLE_TELEMETRY", std::nullopt);
    variables_.emplace_back("ORT_TELEMETRY_DISABLED", std::nullopt);
  }

 private:
  std::deque<ScopedEnvironmentVariable> variables_;
};

TEST_F(TelemetryEnvironmentTests, ExplicitOptOutRequiresAffirmativeValue) {
  ScopedEnvironmentVariable opt_out{"ORT_DISABLE_TELEMETRY"};
  const std::pair<const char*, bool> cases[] = {
      {"1", true},
      {"true", true},
      {"TRUE", true},
      {" yes ", true},
      {"on", true},
      {"Y", true},
      {"", false},
      {" ", false},
      {"0", false},
      {"false", false},
      {"no", false},
      {"off", false},
      {"random", false},
  };
  for (const auto& [value, expected] : cases) {
    opt_out.Set(value);
    EXPECT_EQ(ShouldSuppressTelemetryFromEnvironment(), expected) << value;
  }
  opt_out.Set(std::string(Generators::kMaxTelemetryInputBytes + 1, ' '));
  EXPECT_TRUE(ShouldSuppressTelemetryFromEnvironment());
  opt_out.Set(std::nullopt);
  EXPECT_FALSE(ShouldSuppressTelemetryFromEnvironment());
}

TEST_F(TelemetryEnvironmentTests, LegacyAliasUsesPermissiveTruthyValues) {
  ScopedEnvironmentVariable opt_out{"ORT_TELEMETRY_DISABLED"};
  const std::pair<const char*, bool> cases[] = {
      {"1", true},
      {"true", true},
      {" yes ", true},
      {"on", true},
      {"Y", true},
      {"random", true},
      {"", false},
      {"0", false},
      {"false", false},
      {"no", false},
      {"off", false},
  };
  for (const auto& [value, expected] : cases) {
    opt_out.Set(value);
    EXPECT_EQ(ShouldSuppressTelemetryFromEnvironment(), expected) << value;
  }
  opt_out.Set(std::string(Generators::kMaxTelemetryInputBytes + 1, ' '));
  EXPECT_TRUE(ShouldSuppressTelemetryFromEnvironment());
  opt_out.Set(std::nullopt);
  EXPECT_FALSE(ShouldSuppressTelemetryFromEnvironment());
}

TEST_F(TelemetryEnvironmentTests, EnvironmentOptOutObservedByControlIsPermanent) {
  std::atomic<bool> permanently_disabled{false};
  std::atomic<bool> enabled{true};
  ScopedEnvironmentVariable opt_out{"ORT_DISABLE_TELEMETRY", "true"};

  ApplyTelemetryEnabledControl(false, permanently_disabled, enabled, lifecycle_mutex_);
  EXPECT_TRUE(permanently_disabled.load());
  EXPECT_FALSE(enabled.load());

  opt_out.Set(std::nullopt);
  ApplyTelemetryEnabledControl(true, permanently_disabled, enabled, lifecycle_mutex_);
  EXPECT_TRUE(permanently_disabled.load());
  EXPECT_FALSE(enabled.load());
}

TEST_F(TelemetryEnvironmentTests, OrdinaryApiDisableRemainsReversible) {
  std::atomic<bool> permanently_disabled{false};
  std::atomic<bool> enabled{true};

  ApplyTelemetryEnabledControl(false, permanently_disabled, enabled, lifecycle_mutex_);
  EXPECT_FALSE(permanently_disabled.load());
  EXPECT_FALSE(enabled.load());

  ApplyTelemetryEnabledControl(true, permanently_disabled, enabled, lifecycle_mutex_);
  EXPECT_FALSE(permanently_disabled.load());
  EXPECT_TRUE(enabled.load());
}

TEST_F(TelemetryEnvironmentTests, ControlReadsOptOutOnlyAfterAcquiringLifecycleLock) {
  std::atomic<bool> permanently_disabled{false};
  std::atomic<bool> enabled{true};
  ScopedEnvironmentVariable opt_out{"ORT_DISABLE_TELEMETRY"};
  std::unique_lock<std::shared_mutex> lock(lifecycle_mutex_);
  std::promise<void> started;
  auto control = std::async(std::launch::async, [&] {
    started.set_value();
    ApplyTelemetryEnabledControl(true, permanently_disabled, enabled, lifecycle_mutex_);
  });
  started.get_future().wait();
  EXPECT_EQ(control.wait_for(std::chrono::milliseconds{50}), std::future_status::timeout);
  opt_out.Set("true");
  lock.unlock();
  control.get();

  EXPECT_TRUE(permanently_disabled.load());
  EXPECT_FALSE(enabled.load());
  opt_out.Set(std::nullopt);
  ApplyTelemetryEnabledControl(true, permanently_disabled, enabled, lifecycle_mutex_);
  EXPECT_FALSE(enabled.load());
}

TEST_F(TelemetryEnvironmentTests, DelayedProcessLoggingHonorsPermanentOptOut) {
  std::atomic<bool> permanently_disabled{false};
  std::atomic<bool> enabled{true};
  ApplyTelemetryEnabledControl(false, permanently_disabled, enabled, lifecycle_mutex_);
  EXPECT_TRUE(LockTelemetryForLogging(false, permanently_disabled, enabled, lifecycle_mutex_));
  EXPECT_FALSE(LockTelemetryForLogging(true, permanently_disabled, enabled, lifecycle_mutex_));

  std::promise<void> resume;
  auto process_logging = std::async(std::launch::async, [&] {
    resume.get_future().wait();
    return LockTelemetryForLogging(false, permanently_disabled, enabled, lifecycle_mutex_).owns_lock();
  });
  ScopedEnvironmentVariable opt_out{"ORT_DISABLE_TELEMETRY", "true"};
  ApplyTelemetryEnabledControl(false, permanently_disabled, enabled, lifecycle_mutex_);
  opt_out.Set(std::nullopt);
  resume.set_value();
  EXPECT_FALSE(process_logging.get());
}

TEST_F(TelemetryEnvironmentTests, RecognizesEveryCiAndTestFlagAndFailsClosed) {
  EXPECT_FALSE(ShouldSuppressTelemetryFromEnvironment());
  const std::string oversized(Generators::kMaxTelemetryInputBytes + 1, ' ');
  for (const char* name : kTelemetrySuppressionVariables) {
    ScopedEnvironmentVariable variable{name, "random"};
    EXPECT_TRUE(ShouldSuppressTelemetryFromEnvironment()) << name;
    variable.Set("0");
    EXPECT_FALSE(ShouldSuppressTelemetryFromEnvironment()) << name;
    variable.Set(oversized);
    EXPECT_TRUE(ShouldSuppressTelemetryFromEnvironment()) << name;
    variable.Set(std::nullopt);
    EXPECT_FALSE(ShouldSuppressTelemetryFromEnvironment()) << name;
  }
}

}  // namespace
