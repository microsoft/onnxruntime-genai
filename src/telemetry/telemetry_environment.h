// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <cctype>
#include <cstdlib>
#include <string>
#include <string_view>

#ifdef _WIN32
#include <Windows.h>
#endif

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
#ifdef _WIN32
  DWORD required_size = ::GetEnvironmentVariableA(name, nullptr, 0);
  while (required_size != 0) {
    std::string value(required_size, '\0');
    const DWORD written = ::GetEnvironmentVariableA(name, value.data(), required_size);
    if (written == 0) {
      return {};
    }
    if (written < required_size) {
      value.resize(written);
      return value;
    }

    // The value grew between calls. Windows returns its new required size, including the null.
    required_size = written;
  }
  return {};
#else
  const char* value = std::getenv(name);
  return value != nullptr ? std::string(value) : std::string();
#endif
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

// A CI variable counts as present unless its (trimmed) value is empty or an explicit falsey token, so
// a runner exporting e.g. CI=false does not trip detection.
inline bool IsCiValueTruthy(std::string_view value) {
  const std::string v = ToLowerAscii(TrimAscii(value));
  return !v.empty() && v != "0" && v != "false" && v != "no" && v != "off";
}

struct HostEnvironmentEvidence {
  bool docker_marker{};
  bool podman_marker{};
  bool kubernetes{};
  bool aws_ecs{};
  bool generic_container{};
  bool android_emulator{};
  bool apple_virtual_machine{};
  std::string systemd_container;
  std::string cgroup;
  std::string dmi;
  std::string cpu_info;
  std::string kernel_release;
};

struct HostEnvironmentInfo {
  bool is_container;
  bool is_virtual_machine;
  bool is_emulator;
  const char* container_type;
  const char* virtualization_type;
  const char* environment_class;
  const char* detection_confidence;
  const char* device_id_scope;
};

inline bool ContainsAscii(std::string_view haystack, std::string_view needle) {
  return ToLowerAscii(haystack).find(ToLowerAscii(needle)) != std::string::npos;
}

// Classifies only positive evidence. "undetected" deliberately does not claim bare metal.
inline HostEnvironmentInfo ClassifyHostEnvironment(const HostEnvironmentEvidence& evidence) {
  const std::string container_name = ToLowerAscii(TrimAscii(evidence.systemd_container));
  const std::string combined_container_evidence =
      ToLowerAscii(evidence.cgroup + " " + container_name);

  const char* container_type = "none";
  int container_confidence = 0;
  if (evidence.kubernetes || ContainsAscii(combined_container_evidence, "kubepods")) {
    container_type = "kubernetes";
    container_confidence = 2;
  } else if (evidence.aws_ecs) {
    container_type = "amazonECS";
    container_confidence = 2;
  } else if (evidence.podman_marker || ContainsAscii(combined_container_evidence, "libpod") ||
             ContainsAscii(combined_container_evidence, "podman")) {
    container_type = "podman";
    container_confidence = 2;
  } else if (evidence.docker_marker || ContainsAscii(combined_container_evidence, "docker")) {
    container_type = "docker";
    container_confidence = 2;
  } else if (ContainsAscii(combined_container_evidence, "containerd")) {
    container_type = "containerd";
    container_confidence = 1;
  } else if (ContainsAscii(combined_container_evidence, "lxc")) {
    container_type = "lxc";
    container_confidence = 1;
  } else if (!container_name.empty() && container_name != "none") {
    container_type = "other";
    container_confidence = 2;
  } else if (evidence.generic_container) {
    container_type = "other";
    container_confidence = 1;
  }

  const std::string dmi = ToLowerAscii(evidence.dmi);
  const std::string cpu_info = ToLowerAscii(evidence.cpu_info);
  const std::string kernel_release = ToLowerAscii(evidence.kernel_release);
  const char* virtualization_type = "none";
  int virtualization_confidence = 0;
  if (evidence.android_emulator) {
    virtualization_type = "androidEmulator";
    virtualization_confidence = 2;
  } else if (evidence.apple_virtual_machine) {
    virtualization_type = "appleVirtualMachine";
    virtualization_confidence = 2;
  } else if (kernel_release.find("microsoft") != std::string::npos ||
             kernel_release.find("wsl") != std::string::npos) {
    virtualization_type = "wsl";
    virtualization_confidence = 2;
  } else if (dmi.find("vmware") != std::string::npos) {
    virtualization_type = "vmware";
    virtualization_confidence = 2;
  } else if (dmi.find("virtualbox") != std::string::npos ||
             dmi.find("innotek") != std::string::npos) {
    virtualization_type = "virtualBox";
    virtualization_confidence = 2;
  } else if (dmi.find("microsoft corporation") != std::string::npos &&
             dmi.find("virtual machine") != std::string::npos) {
    virtualization_type = "hyperV";
    virtualization_confidence = 2;
  } else if (dmi.find("amazon ec2") != std::string::npos) {
    virtualization_type = "amazonEC2";
    virtualization_confidence = 2;
  } else if (dmi.find("google compute engine") != std::string::npos) {
    virtualization_type = "googleComputeEngine";
    virtualization_confidence = 2;
  } else if (dmi.find("openstack") != std::string::npos) {
    virtualization_type = "openStack";
    virtualization_confidence = 2;
  } else if (dmi.find("kvm") != std::string::npos) {
    virtualization_type = "kvm";
    virtualization_confidence = 2;
  } else if (dmi.find("qemu") != std::string::npos) {
    virtualization_type = "qemu";
    virtualization_confidence = 2;
  } else if (dmi.find("xen") != std::string::npos) {
    virtualization_type = "xen";
    virtualization_confidence = 2;
  } else if (dmi.find("parallels") != std::string::npos) {
    virtualization_type = "parallels";
    virtualization_confidence = 2;
  } else if (dmi.find("bhyve") != std::string::npos) {
    virtualization_type = "bhyve";
    virtualization_confidence = 2;
  } else if (cpu_info.find("hypervisor") != std::string::npos) {
    virtualization_type = "other";
    virtualization_confidence = 1;
  }

  const bool is_container = container_confidence != 0;
  const bool is_virtual_machine = virtualization_confidence != 0;
  const bool is_emulator = evidence.android_emulator;
  const char* environment_class = "undetected";
  if (is_container && is_virtual_machine) {
    environment_class = "containerOnVirtualMachine";
  } else if (is_container) {
    environment_class = "container";
  } else if (is_emulator) {
    environment_class = "emulator";
  } else if (is_virtual_machine) {
    environment_class = "virtualMachine";
  }

  const int confidence =
      container_confidence > virtualization_confidence ? container_confidence
                                                       : virtualization_confidence;
  return {is_container,
          is_virtual_machine,
          is_emulator,
          container_type,
          virtualization_type,
          environment_class,
          confidence == 2 ? "high" : confidence == 1 ? "medium"
                                                     : "none",
          is_container ? "container" : is_virtual_machine ? "virtualMachine"
                                                          : "installation"};
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

// True if ORT_DISABLE_TELEMETRY is set to a truthy value (1/true/yes/on/y, case-insensitive).
// The 1DS provider latches this full opt-out during initialization on every supported platform.
inline bool IsTelemetryDisabledByEnvironment() {
  const std::string value = ToLowerAscii(TrimAscii(GetTelemetryEnv("ORT_DISABLE_TELEMETRY")));
  return value == "1" || value == "true" || value == "yes" || value == "on" || value == "y";
}

}  // namespace Generators::TelemetryInternal
