// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "models/env_utils.h"
#include "scoped-environment-variable.h"
#include "telemetry/telemetry_environment.h"

#include <array>
#include <utility>

namespace Generators::test {
namespace {

TEST(EnvironmentTests, TruthyValueTruthTable) {
  const std::pair<const char*, bool> cases[] = {
      {"1", true},
      {"true", true},
      {"TRUE", true},
      {" yes ", true},
      {"anything", true},
      {"", false},
      {" ", false},
      {"0", false},
      {"false", false},
      {"FALSE", false},
      {"no", false},
      {"off", false},
  };
  for (const auto& [value, expected] : cases) {
    EXPECT_EQ(TelemetryInternal::IsTruthyValue(value), expected) << value;
  }
  EXPECT_TRUE(TelemetryInternal::IsTruthyValue(std::string(kMaxTelemetryInputBytes + 1, ' ')));
}

TEST(EnvironmentTests, ReaderRejectsOversizedValuesWithoutTruncating) {
  ScopedEnvironmentVariable variable{"ORTGENAI_TEST_BOUNDED_ENV", std::nullopt};
  EXPECT_EQ(ReadEnvironmentVariable("ORTGENAI_TEST_BOUNDED_ENV", 0), std::string{});
  for (const size_t size : {0u, 1024u, 16383u, 16384u}) {
    const std::string value(size, 'x');
    variable.Set(value);
    EXPECT_EQ(ReadEnvironmentVariable("ORTGENAI_TEST_BOUNDED_ENV", kMaxTelemetryInputBytes), value);
  }
  variable.Set(std::string(kMaxTelemetryInputBytes + 1, 'x'));
  EXPECT_FALSE(ReadEnvironmentVariable("ORTGENAI_TEST_BOUNDED_ENV", kMaxTelemetryInputBytes));
  variable.Set("x");
  EXPECT_FALSE(ReadEnvironmentVariable("ORTGENAI_TEST_BOUNDED_ENV", 0));
}

TEST(EnvironmentTests, GetEnvPreservesStringAndBooleanSemantics) {
  ScopedEnvironmentVariable variable{"ORTGENAI_TEST_ENV", std::nullopt};
  for (const std::optional<std::string>& input :
       std::array<std::optional<std::string>, 2>{std::nullopt, std::string{}}) {
    variable.Set(input);
    EXPECT_TRUE(GetEnv("ORTGENAI_TEST_ENV").empty());
    for (const bool initial : {false, true}) {
      bool value = initial;
      GetEnv("ORTGENAI_TEST_ENV", value);
      EXPECT_EQ(value, initial);
    }
  }
  const std::pair<const char*, bool> cases[] = {
      {"1", true}, {"true", true}, {"0", false}, {"false", false}};
  for (const auto& [input, expected] : cases) {
    variable.Set(input);
    EXPECT_EQ(GetEnv("ORTGENAI_TEST_ENV"), input);
    bool value = !expected;
    GetEnv("ORTGENAI_TEST_ENV", value);
    EXPECT_EQ(value, expected);
  }
  for (const char* input : {"TRUE", " yes ", "random"}) {
    variable.Set(input);
    bool value = false;
    EXPECT_THROW(GetEnv("ORTGENAI_TEST_ENV", value), std::invalid_argument);
    EXPECT_FALSE(value);
  }
  const std::string large(kMaxTelemetryInputBytes + 1, 'x');
  variable.Set(large);
  EXPECT_EQ(GetEnv("ORTGENAI_TEST_ENV"), large);
}

TEST(EnvironmentTests, ScopedVariableRestoresUnsetEmptyAndPopulatedValues) {
  ScopedEnvironmentVariable original{"ORTGENAI_TEST_ENV_RESTORE", std::nullopt};
  for (const std::optional<std::string>& value :
       std::array<std::optional<std::string>, 3>{std::nullopt, std::string{}, "original"}) {
    original.Set(value);
    {
      ScopedEnvironmentVariable temporary{"ORTGENAI_TEST_ENV_RESTORE", "temporary"};
      EXPECT_EQ(ReadEnvironmentVariable("ORTGENAI_TEST_ENV_RESTORE", 1024), "temporary");
    }
    EXPECT_EQ(ReadEnvironmentVariable("ORTGENAI_TEST_ENV_RESTORE", 1024), value.value_or(std::string{}));
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
}  // namespace Generators::test
