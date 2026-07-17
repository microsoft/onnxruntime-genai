// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <cctype>
#include <cstdlib>
#include <string>
#include <string_view>

namespace Generators::TelemetryInternal {

// Well-known CI / build-pipeline environment variables. Mirrors ONNX Runtime, Olive, and Foundry
// Local so telemetry suppression behaves consistently across stacks; keep the lists in sync if any
// changes.
inline constexpr std::array<const char*, 13> kCiEnvironmentVariableNames = {
    "CI",                                  // Generic CI flag used by many providers
    "TF_BUILD",                            // Azure Pipelines
    "GITHUB_ACTIONS",                      // GitHub Actions
    "GITLAB_CI",                           // GitLab CI
    "CIRCLECI",                            // CircleCI
    "TRAVIS",                              // Travis CI
    "JENKINS_URL",                         // Jenkins
    "CODEBUILD_BUILD_ID",                  // AWS CodeBuild
    "BUILDKITE",                           // Buildkite
    "TEAMCITY_VERSION",                    // TeamCity
    "APPVEYOR",                            // AppVeyor
    "BITBUCKET_BUILD_NUMBER",              // Bitbucket Pipelines
    "SYSTEM_TEAMFOUNDATIONCOLLECTIONURI",  // Azure DevOps
};

inline std::string GetTelemetryEnv(const char* name) {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
  const char* value = std::getenv(name);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
  return value != nullptr ? std::string(value) : std::string();
}

inline std::string_view TrimAscii(std::string_view s) {
  size_t begin = 0;
  size_t end = s.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(s[begin]))) ++begin;
  while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
  return s.substr(begin, end - begin);
}

inline std::string ToLowerAscii(std::string_view s) {
  std::string out{s};
  for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

// Case-insensitive check for explicit telemetry opt-out values.
inline bool IsEnvTruthy(std::string_view value) {
  const std::string v = ToLowerAscii(TrimAscii(value));
  return v == "1" || v == "true" || v == "yes" || v == "on" || v == "y";
}

// A CI variable counts as present unless its (trimmed) value is empty or an explicit falsey token, so
// a runner exporting e.g. CI=false does not trip detection.
inline bool IsCiValueTruthy(std::string_view value) {
  const std::string v = ToLowerAscii(TrimAscii(value));
  return !v.empty() && v != "0" && v != "false" && v != "no" && v != "off";
}

// True if a well-known CI / build-pipeline environment variable is set to a truthy value. Telemetry
// is suppressed in CI to avoid polluting the tenant from automated builds and tests.
inline bool IsRunningInCI() {
  for (const char* name : kCiEnvironmentVariableNames) {
    if (IsCiValueTruthy(GetTelemetryEnv(name))) return true;
  }
  return false;
}

// True if ORT_RUNNING_UNIT_TESTS is set to a truthy value. onnxruntime-genai's and ONNX Runtime's own
// unit-test entry points set this before creating anything, so local (non-CI) test runs never
// initialize the telemetry uploader or emit events. This is an internal harness signal, not a
// user-facing opt-out. The variable name is shared with ONNX Runtime.
inline bool IsRunningUnitTests() {
  return IsCiValueTruthy(GetTelemetryEnv("ORT_RUNNING_UNIT_TESTS"));
}

// ORT_TELEMETRY_DISABLED is the shared ONNX Runtime/GenAI opt-out. ORT_GENAI_TELEMETRY_DISABLED is
// accepted as a GenAI-specific alias. In GenAI these disable model/generate lifecycle telemetry while
// still allowing ProcessInfo outside CI/unit tests.
inline bool IsTelemetryDisabledByEnvVar() {
  return IsEnvTruthy(GetTelemetryEnv("ORT_TELEMETRY_DISABLED")) ||
         IsEnvTruthy(GetTelemetryEnv("ORT_GENAI_TELEMETRY_DISABLED"));
}

}  // namespace Generators::TelemetryInternal
