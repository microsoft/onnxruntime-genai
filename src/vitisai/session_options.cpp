// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "session_options.h"

#include "../models/session_options.h"

#include <memory>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace Generators::VitisAIExecutionProvider {

namespace {

// Minimal cross-platform RAII handle for a dynamically loaded shared library.
// Wraps LoadLibrary/FreeLibrary on Windows and dlopen/dlclose elsewhere.
#if defined(_WIN32)
struct DllDeleter {
  using pointer = HMODULE;
  void operator()(HMODULE h) const noexcept {
    if (h) FreeLibrary(h);
  }
};
using DllHandle = std::unique_ptr<HMODULE, DllDeleter>;

DllHandle LoadSharedLibrary(const std::string& path) {
  DllHandle handle{LoadLibrary(path.c_str())};
  if (!handle) {
    throw std::runtime_error("Failed to load external EP library '" + path +
                             "' (Error " + std::to_string(GetLastError()) + ")");
  }
  return handle;
}

void* GetSymbol(const DllHandle& handle, const char* name) {
  return reinterpret_cast<void*>(GetProcAddress(handle.get(), name));
}
#else
struct DllDeleter {
  void operator()(void* h) const noexcept {
    if (h) dlclose(h);
  }
};
using DllHandle = std::unique_ptr<void, DllDeleter>;

DllHandle LoadSharedLibrary(const std::string& path) {
  DllHandle handle{dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL)};
  if (!handle) {
    throw std::runtime_error("Failed to load external EP library '" + path +
                             "': " + dlerror());
  }
  return handle;
}

void* GetSymbol(const DllHandle& handle, const char* name) {
  return dlsym(handle.get(), name);
}
#endif

}  // namespace

DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& config,
                                         bool /*disable_graph_capture*/) {
  // These session config entries need to be in place before the EP is appended
  // so both the plugin and legacy append paths see the same configuration.
  session_options.AddConfigEntry("session.inter_op.allow_spinning", "0");
  session_options.AddConfigEntry("session.intra_op.allow_spinning", "0");
  session_options.AddConfigEntry("model_root", config.config_path.string().c_str());

  // VitisAI does not have a device type specific allocator, so we use CPU
  // as the device type.
  if (!AppendExecutionProviderV2(session_options, provider_options,
                                 DeviceType::CPU, "VitisAIExecutionProvider")) {
    AppendExecutionProviderV1(session_options, provider_options);
  }

  if (const auto opt_it = std::find_if(provider_options.options.begin(), provider_options.options.end(),
                                       [](const auto& pair) { return pair.first == "external_ep_library"; });
      opt_it != provider_options.options.end()) {
    const auto& lib_name = opt_it->second;

    // The library must remain loaded for the lifetime of the process since it
    // provides the EP factory and custom ops used by the ORT session. Park
    // each owning handle in a process-lifetime container so it's released at
    // exit (via static destruction) instead of leaked.
    static std::vector<DllHandle> g_external_ep_libraries;
    auto& lib = g_external_ep_libraries.emplace_back(LoadSharedLibrary(lib_name));

    using CreateEpFactoriesFunc = void (*)(void*, const OrtApiBase*, void*, OrtEpFactory**, size_t, size_t*);
    if (const auto func = reinterpret_cast<CreateEpFactoriesFunc>(GetSymbol(lib, "CreateEpFactories"))) {
      OrtEpFactory* factory = nullptr;
      size_t num = 1;
      func(nullptr, OrtGetApiBase(), nullptr, &factory, num, &num);
    }
    fs::path custom_ops_lib_path(lib_name);
    session_options.RegisterCustomOpsLibrary(custom_ops_lib_path.c_str());
  }

  return nullptr;
}

}  // namespace Generators::VitisAIExecutionProvider
