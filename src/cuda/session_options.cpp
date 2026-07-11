// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "session_options.h"
#include "../models/session_options.h"

#include <algorithm>

#if defined(_WIN32)
std::string CurrentModulePath();
#endif

namespace Generators::CUDAExecutionProvider {

namespace {

// The plugin and legacy CUDA EP intentionally use the same canonical provider
// name. The build of onnxruntime_providers_cuda determines which implementation
// is present.
constexpr const char* kCudaEpName = "CUDAExecutionProvider";

#if defined(ORTGENAI_REGISTER_BUNDLED_CUDA_PLUGIN_EP)
// The CUDA plugin uses the same native filename as the legacy CUDA provider.
#if defined(_WIN32)
constexpr const char* kDefaultCudaPluginLibrary = "onnxruntime_providers_cuda.dll";
#elif defined(__APPLE__)
constexpr const char* kDefaultCudaPluginLibrary = "libonnxruntime_providers_cuda.dylib";
#else
constexpr const char* kDefaultCudaPluginLibrary = "libonnxruntime_providers_cuda.so";
#endif

// Best-effort, idempotent registration of the bundled CUDA plugin library.
// If the library is absent or fails to load, logs a warning and continues so the
// caller transparently falls back to the built-in CUDA EP.
void TryRegisterBundledCudaPluginEp() {
  if (!FindRegisteredEpDevices(kCudaEpName).empty()) {
    return;  // ORT (for example, its Python package) already registered it.
  }

  fs::path library_path{kDefaultCudaPluginLibrary};
#if defined(_WIN32)
  // CurrentModulePath is implemented in generators.cpp and includes the trailing separator.
  library_path = fs::path(::CurrentModulePath()) / kDefaultCudaPluginLibrary;
#elif defined(__linux__) && !defined(__ANDROID__)
  // Resolve next to libonnxruntime-genai rather than relying on the process search path.
  library_path = fs::path(Ort::GetCurrentModuleDir()) / kDefaultCudaPluginLibrary;
#endif

  try {
    EnsureExecutionProviderLibraryRegistered(kCudaEpName, library_path);
  } catch (const std::exception& e) {
    Log("warning", std::string("Failed to register the bundled CUDA plugin EP library '") +
                       library_path.string() + "': " + e.what() +
                       ". Falling back to the provider-bridge CUDA EP.");
  }
}
#endif  // ORTGENAI_REGISTER_BUNDLED_CUDA_PLUGIN_EP

void AppendProviderBridgeExecutionProvider(
    OrtSessionOptions& session_options,
    const Config::ProviderOptions& provider_options,
    DeviceInterface*& device) {
  auto ort_provider_options = OrtCUDAProviderOptionsV2::Create();
  std::vector<const char*> keys, values;

  // Memory management settings
  const char* arena_keys[] = {
      "max_mem",
      "arena_extend_strategy",
      "initial_chunk_size_bytes",
      "max_dead_bytes_per_chunk",
      "initial_growth_chunk_size_bytes"};
  size_t arena_values[] = {
      static_cast<size_t>(0),
      static_cast<size_t>(-1),
      static_cast<size_t>(-1),
      static_cast<size_t>(-1),
      static_cast<size_t>(-1)};
  bool use_arena_management = false;

  for (auto& option : provider_options.options) {
    auto it = std::find(std::begin(arena_keys), std::end(arena_keys), option.first);

    if (it == std::end(arena_keys)) {
      keys.emplace_back(option.first.c_str());
      values.emplace_back(option.second.c_str());
    } else {
      const size_t idx = std::distance(std::begin(arena_keys), it);
      long long parsed_value = std::stoll(option.second);
      if (parsed_value < -1) {
        throw std::out_of_range("Arena configuration option value is out of range");
      }
      arena_values[idx] = (parsed_value == -1)
                              ? static_cast<size_t>(-1)
                              : static_cast<size_t>(parsed_value);
      use_arena_management = true;
    }
  }
  ort_provider_options->Update(keys.data(), values.data(), keys.size());

  // Device type determines the scoring device.
  // Create and set our cudaStream_t
  ort_provider_options->UpdateValue("user_compute_stream", device->GetCudaStream());

  // Use fine-grained memory management of BFC Arena.
  // The arena_cfg must outlive the AppendExecutionProvider_CUDA_V2 call below,
  // so it is declared outside the if block.
  std::unique_ptr<OrtArenaCfg> arena_cfg;
  if (use_arena_management) {
    arena_cfg = OrtArenaCfg::Create(arena_keys, arena_values, std::size(arena_keys));
    ort_provider_options->UpdateValue("default_memory_arena_cfg", arena_cfg.get());
  }

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

#if defined(ORTGENAI_REGISTER_BUNDLED_CUDA_PLUGIN_EP)
  // Bundled deployment (e.g. the onnxruntime-genai-cuda package): the plugin
  // library ships next to libonnxruntime-genai, so genai registers that
  // colocated file. No caller-side registration is required. Best-effort: if the
  // library is missing, this falls back to the built-in CUDA EP below.
  TryRegisterBundledCudaPluginEp();
#endif

  // Use the V2 path when the canonical CUDA EP has been registered as a plugin.
  // Otherwise use the legacy provider bridge. Both implementations are named
  // CUDAExecutionProvider and consume the same provider options.
  if (!AppendExecutionProviderV2(session_options, provider_options,
                                 DeviceType::CUDA, kCudaEpName)) {
    CUDAExecutionProvider::AppendProviderBridgeExecutionProvider(
        session_options, provider_options, device);
  }

  return device;
}

}  // namespace Generators::CUDAExecutionProvider
