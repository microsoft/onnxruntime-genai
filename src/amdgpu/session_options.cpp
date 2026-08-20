// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "session_options.h"

#include <filesystem>
#include <string>

#include "../models/env_utils.h"
#include "../models/session_options.h"
#include "interface.h"

#if defined(_WIN32)
#include <windows.h>
#endif

namespace Generators::AMDGPUExecutionProvider {

namespace {

constexpr const char* kEpPathEnvKey = "AMDGPU_EP_PATH";
#if defined(_WIN32)
constexpr const char* kEpFilename = "amdgpu-ep.dll";
#else
constexpr const char* kEpFilename = "libamdgpu-ep.so";
#endif

// A plugin EP is only discoverable once its library is registered on the OrtEnv. Hosts that can
// supply a path do that themselves; resolve it here for the ones that cannot. No-op if the EP is
// already registered or the library is not found, so an explicit registration always wins.
void EnsureUmbrellaEpRegistered() {
  if (!FindRegisteredEpDevices(kAMDGPUExecutionProviderName).empty())
    return;

  std::error_code ec;
  std::filesystem::path ep_path = GetEnv(kEpPathEnvKey);

#if defined(_WIN32)
  const auto module_of = [](const void* address) -> HMODULE {
    MEMORY_BASIC_INFORMATION mbi;
    if (VirtualQuery(address, &mbi, sizeof(mbi)) && mbi.AllocationBase)
      return reinterpret_cast<HMODULE>(mbi.AllocationBase);
    return nullptr;
  };

  const auto find_next_to_module = [&](HMODULE module) -> std::filesystem::path {
    wchar_t buffer[MAX_PATH + 1] = {0};
    if (GetModuleFileNameW(module, buffer, MAX_PATH + 1))
      if (const auto dir = std::filesystem::path{buffer}.remove_filename(); !dir.empty())
        if (auto path = dir / kEpFilename; std::filesystem::exists(path, ec))
          return path;
    return {};
  };

  if (ep_path.empty())
    // next to onnxruntime-genai, using a symbol in that module as the address marker
    if (const auto module = module_of(reinterpret_cast<const void*>(&GetAMDGPUInterface)))
      ep_path = find_next_to_module(module);

  if (ep_path.empty())
    // next to onnxruntime
    if (const auto module = module_of(reinterpret_cast<const void*>(Ort::api->RegisterExecutionProviderLibrary)))
      ep_path = find_next_to_module(module);

  if (ep_path.empty())
    // next to the current executable
    if (const auto module = GetModuleHandleA(nullptr))
      ep_path = find_next_to_module(module);
#endif

  if (ep_path.empty())
    ep_path = std::filesystem::current_path(ec) / kEpFilename;

  if (!std::filesystem::exists(ep_path, ec))
    return;

  try {
    Ort::RegisterExecutionProviderLibrary(&GetOrtEnv(), kAMDGPUExecutionProviderName, ep_path.native().c_str());
  } catch (const Ort::Exception& e) {
    // Registered but advertising no device: the check above cannot see that, ORT reports it here.
    if (std::string(e.what()).find("already registered") == std::string::npos)
      throw;
  }
}

// Emit static-padding hints so the EP pads the prefill token axis to max_length and
// compiles it once, instead of recompiling per prompt length.
void SetStaticPaddingConfig(OrtSessionOptions& session_options, const Config& config) {
  const auto& decoder = config.model.decoder;
  const std::string seq_len = std::to_string(config.search.max_length);
  const std::string pad_inputs =
      decoder.inputs.input_ids + ":1," + decoder.inputs.position_ids + ":1";
  const std::string pad_outputs = decoder.outputs.logits + ":1";

  session_options.AddConfigEntry("ep.migraphx.static_pad_seq", "1");
  session_options.AddConfigEntry("ep.migraphx.static_pad_seq_len", seq_len.c_str());
  session_options.AddConfigEntry("ep.migraphx.static_pad_inputs", pad_inputs.c_str());
  session_options.AddConfigEntry("ep.migraphx.static_pad_outputs", pad_outputs.c_str());

  session_options.AddConfigEntry("ep.migraphx.hip_graph_enable", "1");
}

}  // namespace

DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& config,
                                         bool /*disable_graph_capture*/) {
  EnsureUmbrellaEpRegistered();

  SetStaticPaddingConfig(session_options, config);

  // Umbrella-level hint: the model architecture drives the EP's backend routing.
  session_options.AddConfigEntry("ep.amdgpuexecutionprovider.model_arch", config.model.type.c_str());

  // DirectML backend: host-accessible decode inputs.
  session_options.AddConfigEntry("ep.directml.enable_host_accessible", "1");

  AppendExecutionProviderV2(session_options, provider_options,
                            DeviceType::AMDGPU, kAMDGPUExecutionProviderName);

  return GetAMDGPUInterface();
}

}  // namespace Generators::AMDGPUExecutionProvider
