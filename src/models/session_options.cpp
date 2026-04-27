// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "session_options.h"

#include <functional>
#include <unordered_map>

#include "../cuda/session_options.h"
#include "../dml/session_options.h"
#include "../nvtensorrtrtx/session_options.h"
#include "../openvino/session_options.h"
#include "../qnn/session_options.h"
#include "../ryzenai/session_options.h"
#include "../amdgpu/session_options.h"
#include "../vitisai/session_options.h"
#include "../webgpu/session_options.h"

namespace Generators {

// Each execution provider has a dedicated AppendExecutionProvider function in its
// own src/<provider>/session_options.cpp file. The dispatch table in
// SetProviderSessionOptions (below) maps provider names to these functions.
// Providers that are not in the dispatch table are treated as generic and
// attempted via V2 (plugin) then V1 (legacy) API paths.

std::vector<const OrtEpDevice*> ApplyDeviceFiltering(const Config::ProviderOptions& provider_options,
                                                     const std::vector<const OrtEpDevice*>& devices) {
  const auto& filtering_options = provider_options.device_filtering_options;
  if (!filtering_options ||
      (!filtering_options->hardware_device_id &&
       !filtering_options->hardware_vendor_id &&
       !filtering_options->hardware_device_type)) {
    return devices;
  }

  std::vector<const OrtEpDevice*> filtered_devices;

  for (const auto* device : devices) {
    bool match = true;
    auto* ort_device = device->Device();

    if (filtering_options->hardware_device_id &&
        ort_device->DeviceId() != *filtering_options->hardware_device_id) {
      match = false;
    }

    if (filtering_options->hardware_vendor_id &&
        ort_device->VendorId() != *filtering_options->hardware_vendor_id) {
      match = false;
    }

    if (filtering_options->hardware_device_type &&
        ort_device->Type() != *filtering_options->hardware_device_type) {
      match = false;
    }

    if (match) {
      filtered_devices.push_back(device);
    }
  }

  if (filtered_devices.empty()) {
    std::string error_msg = "No devices matched the filtering criteria specified in provider options. Filter criteria:";
    if (filtering_options->hardware_device_id) {
      error_msg += " hardware_device_id=" + std::to_string(*filtering_options->hardware_device_id);
    }
    if (filtering_options->hardware_vendor_id) {
      error_msg += " hardware_vendor_id=" + std::to_string(*filtering_options->hardware_vendor_id);
    }
    if (filtering_options->hardware_device_type) {
      error_msg += " hardware_device_type=" + std::to_string(static_cast<int>(*filtering_options->hardware_device_type));
    }
    error_msg += ". Available devices:";
    for (const auto* device : devices) {
      auto* ort_device = device->Device();
      error_msg += " [device_id=" + std::to_string(ort_device->DeviceId()) +
                   ", vendor_id=" + std::to_string(ort_device->VendorId()) +
                   ", type=" + std::to_string(static_cast<int>(ort_device->Type())) + "]";
    }
    error_msg += ". Verify that the device filtering options in genai_config.json match an available device.";
    throw std::runtime_error(error_msg);
  }
  return filtered_devices;
}

// Helper to check if a provider is registered and get all matching EP devices
std::vector<const OrtEpDevice*> FindRegisteredEpDevices(const std::string& ep_name) {
  auto device_ptrs = GetOrtEnv().GetEpDevices();
  std::vector<const OrtEpDevice*> ep_devices_ptrs;
  for (auto* device : device_ptrs) {
    if (device->Name() == ep_name) {
      ep_devices_ptrs.push_back(device);
    }
  }
  return ep_devices_ptrs;
}

// If an execution provider is plugged-in via the registered plugin mechanism,
// this function appends the provider using the V2 API.
// Returns true if the provider was appended via the plugin path, false if the
// provider is not registered.
bool AppendExecutionProviderV2(
    OrtSessionOptions& session_options,
    const Config::ProviderOptions& provider_options,
    DeviceType device_type,
    const std::string& ep_name) {
  auto ep_devices_ptrs = FindRegisteredEpDevices(ep_name);
  if (ep_devices_ptrs.empty()) return false;  // Not registered

  std::unordered_map<std::string, std::string> options;
  for (auto& option : provider_options.options) {
    options.insert(option);
  }

  auto filtered_ep_device_ptrs = ApplyDeviceFiltering(provider_options, ep_devices_ptrs);
  if (device_type == DeviceType::WEBGPU ||
      device_type == DeviceType::CUDA) {
    // WebGPU EP factory and CUDA EP factory currently only support one device at a time.
    filtered_ep_device_ptrs = {filtered_ep_device_ptrs.front()};
  }

  session_options.AppendExecutionProvider_V2(GetOrtEnv(), filtered_ep_device_ptrs, options);
  return true;
}

void AppendExecutionProviderV1(OrtSessionOptions& session_options,
                               const Config::ProviderOptions& provider_options) {
  std::vector<const char*> keys, values;
  for (auto& option : provider_options.options) {
    keys.emplace_back(option.first.c_str());
    values.emplace_back(option.second.c_str());
  }
  session_options.AppendExecutionProvider(provider_options.name.c_str(), keys.data(),
                                          values.data(), keys.size());
}

DeviceInterface* SetProviderSessionOptions(OrtSessionOptions& session_options,
                                           const std::vector<std::string>& providers,
                                           const std::vector<Config::ProviderOptions>& provider_options_list,
                                           bool is_primary_session_options,
                                           const Config& config,
                                           bool disable_graph_capture) {
  using AppendExecutionProviderFn = DeviceInterface* (*)(OrtSessionOptions&,
                                                         const Config::ProviderOptions&,
                                                         const Config&,
                                                         bool);

  // Dispatch table: maps provider name (as it appears in genai_config.json) to
  // the corresponding provider-specific AppendExecutionProvider function.
  static const std::unordered_map<std::string, AppendExecutionProviderFn> append_execution_provider{
      {"cuda", CUDAExecutionProvider::AppendExecutionProvider},
      {"DML", DMLExecutionProvider::AppendExecutionProvider},
      {"NvTensorRtRtx", NvTensorRtRtxExecutionProvider::AppendExecutionProvider},
      {"OpenVINO", OpenVINOExecutionProvider::AppendExecutionProvider},
      {"RyzenAI", RyzenAIExecutionProvider::AppendExecutionProvider},
      {"QNN", QNNExecutionProvider::AppendExecutionProvider},
      {"AMDGPU", AMDGPUExecutionProvider::AppendExecutionProvider},
      {"VitisAI", VitisAIExecutionProvider::AppendExecutionProvider},
      {"WebGPU", WebGPUExecutionProvider::AppendExecutionProvider},
  };

  DeviceInterface* device{};

  auto providers_list = providers;
  if (!is_primary_session_options) {
    // Providers specified in a non-primary provider options list are added
    // to the primary providers. They are considered immutable and implicitly
    // added as providers.
    for (const auto& provider_options : provider_options_list) {
      if (std::find(providers_list.begin(), providers_list.end(), provider_options.name) == providers_list.end()) {
        providers_list.push_back(provider_options.name);
      }
    }
  }

  for (const auto& provider : providers_list) {
    auto provider_options_it = std::find_if(provider_options_list.begin(), provider_options_list.end(),
                                            [&provider](const Config::ProviderOptions& po) { return po.name == provider; });

    if (provider_options_it == provider_options_list.end()) {
      throw std::runtime_error("Provider options not found for provider: " + provider);
    }
    const auto& provider_options = *provider_options_it;

    const auto append_provider_it = append_execution_provider.find(provider_options.name);
    if (append_provider_it != append_execution_provider.end()) {
      auto session_device = append_provider_it->second(session_options, provider_options, config, disable_graph_capture);
      if (is_primary_session_options && session_device && !device) {
        device = session_device;  // Set the device if not already set by a previous provider
      }
    } else {
      if (!AppendExecutionProviderV2(session_options, provider_options,
                                     DeviceType::CPU, provider_options.name)) {
        AppendExecutionProviderV1(session_options, provider_options);
      }
    }
  }

  return device;
}

}  // namespace Generators
