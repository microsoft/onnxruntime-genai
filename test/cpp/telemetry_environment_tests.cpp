// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scoped-environment-variable.h"
#include "telemetry/telemetry_environment.h"

#include <array>
#include <memory>
#include <vector>

namespace {

using Generators::test::ScopedEnvironmentVariable;
using namespace Generators::TelemetryInternal;

class TelemetryEnvironmentTests : public ::testing::Test {
 protected:
  void SetUp() override {
    for (const char* name : kTelemetrySuppressionVariables) {
      variables_.push_back(std::make_unique<ScopedEnvironmentVariable>(name, std::nullopt));
    }
    variables_.push_back(
        std::make_unique<ScopedEnvironmentVariable>("ORT_DISABLE_TELEMETRY", std::nullopt));
  }

 private:
  std::vector<std::unique_ptr<ScopedEnvironmentVariable>> variables_;
};

TEST_F(TelemetryEnvironmentTests, NonFalseValueTruthTable) {
  for (const char* value : {"1", "true", "TRUE", " yes ", "anything"}) {
    EXPECT_TRUE(IsNonFalseValue(value)) << value;
  }
  for (const char* value : {"", " ", "0", "false", "FALSE", "no", "off"}) {
    EXPECT_FALSE(IsNonFalseValue(value)) << value;
  }
}

TEST_F(TelemetryEnvironmentTests, ExplicitOptOutTruthTable) {
  ScopedEnvironmentVariable opt_out{"ORT_DISABLE_TELEMETRY"};
  for (const char* value : {"1", "true", "TRUE", " yes ", "on", "Y"}) {
    opt_out.Set(value);
    EXPECT_TRUE(ShouldSuppressTelemetryFromEnvironment()) << value;
  }
  for (const char* value : {"", " ", "0", "false", "no", "off", "random"}) {
    opt_out.Set(value);
    EXPECT_FALSE(ShouldSuppressTelemetryFromEnvironment()) << value;
  }
  opt_out.Set(std::nullopt);
  EXPECT_FALSE(ShouldSuppressTelemetryFromEnvironment());
}

TEST_F(TelemetryEnvironmentTests, EveryCiAndTestFlagUsesNonFalseSemantics) {
  EXPECT_FALSE(ShouldSuppressTelemetryFromEnvironment());
  for (const char* name : kTelemetrySuppressionVariables) {
    ScopedEnvironmentVariable variable{name};
    for (const char* value : {"1", "TRUE", " yes ", "random"}) {
      variable.Set(value);
      EXPECT_TRUE(ShouldSuppressTelemetryFromEnvironment()) << name << "=" << value;
    }
    for (const char* value : {"", " ", "0", "FALSE", "no", "off"}) {
      variable.Set(value);
      EXPECT_FALSE(ShouldSuppressTelemetryFromEnvironment()) << name << "=" << value;
    }
    variable.Set(std::nullopt);
    EXPECT_FALSE(ShouldSuppressTelemetryFromEnvironment()) << name;
  }
}

TEST_F(TelemetryEnvironmentTests, RejectsOversizedEnvironmentInsteadOfTruncating) {
  ScopedEnvironmentVariable variable{"ORTGENAI_TEST_BOUNDED_ENV", std::nullopt};
  using namespace Generators;
  EXPECT_EQ(ReadEnvironmentVariable("ORTGENAI_TEST_BOUNDED_ENV", 0), std::string{});
  for (const size_t size : {0u, 1024u, 16384u}) {
    const std::string value(size, 'x');
    variable.Set(value);
    const auto actual = ReadEnvironmentVariable("ORTGENAI_TEST_BOUNDED_ENV", kMaxTelemetryInputBytes);
    ASSERT_TRUE(actual);
    EXPECT_EQ(*actual, value);
  }
  variable.Set(std::string(kMaxTelemetryInputBytes + 1, 'x'));
  EXPECT_FALSE(ReadEnvironmentVariable("ORTGENAI_TEST_BOUNDED_ENV", kMaxTelemetryInputBytes));
  variable.Set("x");
  EXPECT_FALSE(ReadEnvironmentVariable("ORTGENAI_TEST_BOUNDED_ENV", 0));
}

TEST_F(TelemetryEnvironmentTests, EveryOversizedSuppressionFlagFailsClosed) {
  const std::string oversized(Generators::kMaxTelemetryInputBytes + 1, ' ');
  for (const char* name : kTelemetrySuppressionVariables) {
    ScopedEnvironmentVariable variable{name, oversized};
    EXPECT_TRUE(ShouldSuppressTelemetryFromEnvironment()) << name;
  }
  ScopedEnvironmentVariable opt_out{"ORT_DISABLE_TELEMETRY", oversized};
  EXPECT_TRUE(ShouldSuppressTelemetryFromEnvironment());
}

TEST(EnvironmentTests, ScopedVariableRestoresUnsetEmptyAndPopulatedValues) {
  ScopedEnvironmentVariable original{"ORTGENAI_TEST_ENV_RESTORE", std::nullopt};
  for (const std::optional<std::string>& value :
       std::array<std::optional<std::string>, 3>{std::nullopt, std::string{}, "original"}) {
    original.Set(value);
    {
      ScopedEnvironmentVariable temporary{"ORTGENAI_TEST_ENV_RESTORE", "temporary"};
      EXPECT_EQ(Generators::ReadEnvironmentVariable("ORTGENAI_TEST_ENV_RESTORE", 1024), "temporary");
    }
    EXPECT_EQ(Generators::ReadEnvironmentVariable("ORTGENAI_TEST_ENV_RESTORE", 1024),
              value.value_or(std::string{}));
#ifdef _WIN32
    ::SetLastError(ERROR_SUCCESS);
    const DWORD size = ::GetEnvironmentVariableA("ORTGENAI_TEST_ENV_RESTORE", nullptr, 0);
    if (!value) {
      EXPECT_EQ(size, 0u);
      EXPECT_EQ(::GetLastError(), static_cast<DWORD>(ERROR_ENVVAR_NOT_FOUND));
    } else {
      EXPECT_NE(::GetLastError(), static_cast<DWORD>(ERROR_ENVVAR_NOT_FOUND));
    }
#else
    EXPECT_EQ(std::getenv("ORTGENAI_TEST_ENV_RESTORE") != nullptr, value.has_value());
#endif
  }
}

}  // namespace
