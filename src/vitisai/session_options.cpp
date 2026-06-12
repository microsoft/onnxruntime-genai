// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "session_options.h"

#include "../models/session_options.h"

#include <memory>
#include <string>
#include <vector>

namespace Generators::VitisAIExecutionProvider {

#if defined(_WIN32)
namespace {

// RAII wrapper for HMODULE so a loaded external EP library is released
// (FreeLibrary) at process exit instead of leaked.
struct HModuleDeleter {
  using pointer = HMODULE;
  void operator()(HMODULE h) const noexcept {
    if (h) FreeLibrary(h);
  }
};
using HModulePtr = std::unique_ptr<HMODULE, HModuleDeleter>;

HModulePtr LoadSharedLibrary(const std::string& path) {
  HModulePtr handle{LoadLibrary(path.c_str())};
  if (!handle) {
    throw std::runtime_error("Failed to load external EP library '" + path +
                             "' (Error " + std::to_string(GetLastError()) + ")");
  }
  return handle;
}

}  // namespace
#endif  // _WIN32

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

#if defined(_WIN32)
  // Windows-only because the CreateEpFactories handshake below needs the
  // OrtApiBase pointer that OrtGetApiBase returns, and only the Windows build
  // of libonnxruntime-genai links against onnxruntime to resolve that symbol;
  // on Linux ORT is loaded dynamically and OrtGetApiBase is not available at
  // link time.
  if (const auto opt_it = std::find_if(provider_options.options.begin(), provider_options.options.end(),
                                       [](const auto& pair) { return pair.first == "external_ep_library"; });
      opt_it != provider_options.options.end()) {
    const auto& lib_name = opt_it->second;

    // The library must remain loaded for the lifetime of the process since it
    // provides the EP factory and custom ops used by the ORT session. Park
    // each owning handle in a process-lifetime container so it's released at
    // exit (via static destruction) instead of leaked.
    static std::vector<HModulePtr> g_external_ep_libraries;
    g_external_ep_libraries.push_back(LoadSharedLibrary(lib_name));
    HMODULE lib = g_external_ep_libraries.back().get();

    using CreateEpFactoriesFunc = void (*)(void*, const OrtApiBase*, void*, OrtEpFactory**, size_t, size_t*);
    if (const auto func = reinterpret_cast<CreateEpFactoriesFunc>(GetProcAddress(lib, "CreateEpFactories"))) {
      OrtEpFactory* factory = nullptr;
      size_t num = 1;
      func(nullptr, OrtGetApiBase(), nullptr, &factory, num, &num);
    }
    fs::path custom_ops_lib_path(lib_name);
    session_options.RegisterCustomOpsLibrary(custom_ops_lib_path.c_str());
  }
#endif  // _WIN32

  return nullptr;
}

}  // namespace Generators::VitisAIExecutionProvider
