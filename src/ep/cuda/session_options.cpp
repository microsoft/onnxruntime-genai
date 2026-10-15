// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "session_options.h"
#include "models/session_options.h"

namespace Generators::CUDAExecutionProvider {

namespace {

void AppendProviderBridgeExecutionProvider(
    OrtSessionOptions& session_options,
    const Config::ProviderOptions& provider_options,
    DeviceInterface*& device) {
  auto ort_provider_options = OrtCUDAProviderOptionsV2::Create();
  const auto normalized_options =
      NormalizeProviderBridgeOptions(provider_options);
  std::vector<const char*> keys, values;
  for (const auto& option : normalized_options.options) {
    keys.emplace_back(option.first.c_str());
    values.emplace_back(option.second.c_str());
  }
  ort_provider_options->Update(keys.data(), values.data(), keys.size());

  // Device type determines the scoring device.
  // Create and set our cudaStream_t
  ort_provider_options->UpdateValue("user_compute_stream", device->GetCudaStream());

  session_options.AppendExecutionProvider_CUDA_V2(*ort_provider_options);
}

}  // namespace

void AddCudaStreamConfig(OrtSessionOptions& session_options, DeviceInterface* device,
                         const std::string& config_key) {
  if (device) {
    void* stream_ptr = device->GetCudaStream();
    std::stringstream stream_value;
    stream_value << reinterpret_cast<uintptr_t>(stream_ptr);
    session_options.AddConfigEntry(config_key.c_str(), stream_value.str().c_str());
  }
}

DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& /*config*/,
                                         bool /*disable_graph_capture*/) {
  auto device = GetDeviceInterface(DeviceType::CUDA);
  AddCudaStreamConfig(session_options, device);

  // CUDA EP honors enable_cuda_graph from the config as-is. Non-decoder
  // sessions (vision, embedding, speech) should set enable_cuda_graph=0
  // in their own session_options in genai_config.json.

  // Try pre-registered plugin path first
  if (!AppendExecutionProviderV2(session_options, provider_options,
                                 DeviceType::CUDA, "CUDAExecutionProvider")) {
    // Register the CUDA execution provider as a provider-bridge provider.
    CUDAExecutionProvider::AppendProviderBridgeExecutionProvider(
        session_options, provider_options, device);
  }

  return device;
}

}  // namespace Generators::CUDAExecutionProvider
