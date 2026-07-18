// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telemetry/telemetry_environment.h"

#include <cstdlib>
#include <string>

#include <gtest/gtest.h>

namespace {

void SetEnv(const char* name, const char* value) {
#ifdef _WIN32
  EXPECT_NE(::SetEnvironmentVariableA(name, value), 0);
#else
  setenv(name, value, 1);
#endif
}

void UnsetEnv(const char* name) {
#ifdef _WIN32
  EXPECT_NE(::SetEnvironmentVariableA(name, nullptr), 0);
#else
  unsetenv(name);
#endif
}

class ScopedEnvVar {
 public:
  explicit ScopedEnvVar(const char* name) : name_{name} {
#ifdef _WIN32
    ::SetLastError(ERROR_SUCCESS);
    const DWORD required_size = ::GetEnvironmentVariableA(name, nullptr, 0);
    if (required_size == 0) {
      had_value_ = ::GetLastError() != ERROR_ENVVAR_NOT_FOUND;
    } else {
      saved_.resize(required_size);
      const DWORD written = ::GetEnvironmentVariableA(name, saved_.data(), required_size);
      had_value_ = written < required_size;
      saved_.resize(had_value_ ? written : 0);
    }
#else
    const char* value = std::getenv(name);
    had_value_ = value != nullptr;
    if (had_value_) saved_ = value;
#endif
  }

  ~ScopedEnvVar() {
    if (had_value_) {
      SetEnv(name_, saved_.c_str());
    } else {
      UnsetEnv(name_);
    }
  }

 private:
  const char* name_;
  bool had_value_{false};
  std::string saved_;
};

void ExpectOptOutEnv(const char* ort_value, const char* genai_value, bool expected) {
  ScopedEnvVar ort_guard{"ORT_TELEMETRY_DISABLED"};
  ScopedEnvVar genai_guard{"ORT_GENAI_TELEMETRY_DISABLED"};

  if (ort_value != nullptr) {
    SetEnv("ORT_TELEMETRY_DISABLED", ort_value);
  } else {
    UnsetEnv("ORT_TELEMETRY_DISABLED");
  }
  if (genai_value != nullptr) {
    SetEnv("ORT_GENAI_TELEMETRY_DISABLED", genai_value);
  } else {
    UnsetEnv("ORT_GENAI_TELEMETRY_DISABLED");
  }

  EXPECT_EQ(Generators::TelemetryInternal::IsTelemetryDisabledByEnvVar(), expected);
}

}  // namespace

TEST(TelemetryEnvironmentTests, ExplicitOptOutTruthTable) {
  using Generators::TelemetryInternal::IsEnvTruthy;

  EXPECT_TRUE(IsEnvTruthy("1"));
  EXPECT_TRUE(IsEnvTruthy("true"));
  EXPECT_TRUE(IsEnvTruthy("TRUE"));
  EXPECT_TRUE(IsEnvTruthy(" yes "));
  EXPECT_TRUE(IsEnvTruthy("on"));
  EXPECT_TRUE(IsEnvTruthy("Y"));

  EXPECT_FALSE(IsEnvTruthy(""));
  EXPECT_FALSE(IsEnvTruthy(" "));
  EXPECT_FALSE(IsEnvTruthy("0"));
  EXPECT_FALSE(IsEnvTruthy("false"));
  EXPECT_FALSE(IsEnvTruthy("no"));
  EXPECT_FALSE(IsEnvTruthy("off"));
  EXPECT_FALSE(IsEnvTruthy("random"));
}

TEST(TelemetryEnvironmentTests, CiValueTruthTable) {
  using Generators::TelemetryInternal::IsCiValueTruthy;

  EXPECT_TRUE(IsCiValueTruthy("1"));
  EXPECT_TRUE(IsCiValueTruthy("true"));
  EXPECT_TRUE(IsCiValueTruthy("TRUE"));
  EXPECT_TRUE(IsCiValueTruthy(" yes "));
  EXPECT_TRUE(IsCiValueTruthy("anything"));

  EXPECT_FALSE(IsCiValueTruthy(""));
  EXPECT_FALSE(IsCiValueTruthy(" "));
  EXPECT_FALSE(IsCiValueTruthy("0"));
  EXPECT_FALSE(IsCiValueTruthy("false"));
  EXPECT_FALSE(IsCiValueTruthy("FALSE"));
  EXPECT_FALSE(IsCiValueTruthy("no"));
  EXPECT_FALSE(IsCiValueTruthy("off"));
}

TEST(TelemetryEnvironmentTests, OptOutEnvVarParsing) {
  ExpectOptOutEnv("1", nullptr, true);
  ExpectOptOutEnv("TRUE", nullptr, true);
  ExpectOptOutEnv("0", nullptr, false);
  ExpectOptOutEnv("random", nullptr, false);
  ExpectOptOutEnv(nullptr, "1", true);
  ExpectOptOutEnv(nullptr, "TRUE", true);
  ExpectOptOutEnv(nullptr, "0", false);
  ExpectOptOutEnv(nullptr, "random", false);
  ExpectOptOutEnv("1", "0", true);
  ExpectOptOutEnv(nullptr, "0", false);
  ExpectOptOutEnv(nullptr, nullptr, false);
}

TEST(TelemetryEnvironmentTests, CiDetectionSuppresses) {
  ScopedEnvVar guard{"APPVEYOR"};

  // Only assert the positive direction so the test remains deterministic when it itself runs in CI.
  SetEnv("APPVEYOR", "true");
  EXPECT_TRUE(Generators::TelemetryInternal::IsRunningInCI());
}

TEST(TelemetryEnvironmentTests, RunningUnitTestsSuppresses) {
  ScopedEnvVar guard{"ORT_RUNNING_UNIT_TESTS"};

  SetEnv("ORT_RUNNING_UNIT_TESTS", "1");
  EXPECT_TRUE(Generators::TelemetryInternal::IsRunningUnitTests());

  SetEnv("ORT_RUNNING_UNIT_TESTS", "0");
  EXPECT_FALSE(Generators::TelemetryInternal::IsRunningUnitTests());

  UnsetEnv("ORT_RUNNING_UNIT_TESTS");
  EXPECT_FALSE(Generators::TelemetryInternal::IsRunningUnitTests());
}
