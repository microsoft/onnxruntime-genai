// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <cctype>
#include <string>
#include <string_view>
#include "../models/env_utils.h"
#include "telemetry_string.h"

namespace Generators::TelemetryInternal {

// CI flags mirror ONNX Runtime, Olive, and Foundry Local. The test-harness flag is shared with ORT.
inline constexpr std::array<const char*, 14> kTelemetrySuppressionVariables = {
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
    "ORT_RUNNING_UNIT_TESTS",              // GenAI / ORT native test harness
};

inline constexpr size_t kMaxProcessingBytes = 64 * 1024;

inline std::string_view TrimAscii(std::string_view s) {
  s = s.substr(0, kMaxProcessingBytes);
  size_t begin = 0;
  size_t end = s.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(s[begin]))) ++begin;
  while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
  return s.substr(begin, end - begin);
}

inline std::string ToLowerAscii(std::string_view s) {
  std::string out{s.substr(0, kMaxProcessingBytes)};
  for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

// CI/test flags accept any nonempty value except 0/false/no/off, ignoring case and whitespace.
inline bool IsTruthyValue(std::string_view value) {
  if (value.size() > kMaxTelemetryInputBytes) return true;
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

// Classifies only positive evidence. "undetected" deliberately does not claim bare metal.
inline HostEnvironmentInfo ClassifyHostEnvironment(const HostEnvironmentEvidence& evidence) {
  const std::string container_name = ToLowerAscii(TrimAscii(evidence.systemd_container));
  const std::string combined_container_evidence =
      ToLowerAscii(std::string{std::string_view{evidence.cgroup}.substr(
                       0, kMaxProcessingBytes / 2)} +
                   " " + container_name);

  const char* container_type = "none";
  int container_confidence = 0;
  if (evidence.kubernetes || combined_container_evidence.find("kubepods") != std::string::npos) {
    container_type = "kubernetes";
    container_confidence = 2;
  } else if (evidence.aws_ecs) {
    container_type = "amazonECS";
    container_confidence = 2;
  } else if (evidence.podman_marker || combined_container_evidence.find("libpod") != std::string::npos ||
             combined_container_evidence.find("podman") != std::string::npos) {
    container_type = "podman";
    container_confidence = 2;
  } else if (evidence.docker_marker || combined_container_evidence.find("docker") != std::string::npos) {
    container_type = "docker";
    container_confidence = 2;
  } else if (combined_container_evidence.find("containerd") != std::string::npos) {
    container_type = "containerd";
    container_confidence = 1;
  } else if (combined_container_evidence.find("lxc") != std::string::npos) {
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

// CI/test flags accept any non-false value; the explicit opt-out accepts only affirmative tokens.
// Rejected reads fail closed so an unreadable suppression flag never enables collection.
inline bool ShouldSuppressTelemetryFromEnvironment() {
  for (const char* name : kTelemetrySuppressionVariables) {
    const auto value = GetEnv(name, kMaxTelemetryInputBytes);
    if (!value || IsTruthyValue(*value)) return true;
  }
  const auto input = GetEnv("ORT_DISABLE_TELEMETRY", kMaxTelemetryInputBytes);
  if (!input) return true;
  const std::string value = ToLowerAscii(TrimAscii(*input));
  return value == "1" || value == "true" || value == "yes" || value == "on" || value == "y";
}

}  // namespace Generators::TelemetryInternal
