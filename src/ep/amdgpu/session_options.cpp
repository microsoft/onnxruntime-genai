// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "session_options.h"

#include <filesystem>
#include <string>

#include "models/env_utils.h"
#include "models/session_options.h"
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
  if (!FindRegisteredEpDevices(kAMDGPUExecutionProviderName).empty()) {
    // Already registered (by a host, or by a prior genai model). genai introduces nothing here, so it
    // does not claim ownership — a host-owned registration must keep the ownership flag false.
    return;
  }

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
    // genai introduced this registration -> genai owns it and may unregister it at teardown. Scoped to
    // the current OrtGlobals so it is cleared when the env is torn down (Shutdown / re-init).
    GetOrtGlobals()->amdgpu_owns_ep_registration_ = true;
  } catch (const Ort::Exception& e) {
    // Registered but advertising no device: the check above cannot see that, ORT reports it here.
    // A concurrent/pre-existing registration raced us: genai did not introduce it, so do not take
    // ownership (leave amdgpu_owns_ep_registration_ as-is, default false).
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
}

}  // namespace

void ReleaseOwnedUmbrellaEp(OrtEnv& env, bool& owns_registration) {
  // genai releases the library + its OrtEnv-shared allocators only when it owns the registration; a host
  // that registered them keeps both. env and owns_registration are passed in (not fetched via
  // GetOrtEnv() / GetOrtGlobals()) because ~OrtGlobals calls this while g_ort_globals_mutex is held, so
  // re-entering those accessors — including via FindRegisteredEpDevices — would re-lock it and deadlock.
  if (!owns_registration)
    return;

  // Release the OrtEnv-shared allocators first: the shared GPU allocator holds the plugin's
  // ExecutionContext -> command queue -> device, so UnregisterExecutionProviderLibrary alone cannot
  // drop the device until these are gone. Must run while the EP device is still registered.
  // Best-effort: ReleaseSharedAllocator is a no-op when no matching shared allocator exists.
  const auto release_shared = [&env](const OrtEpDevice* ep_device, OrtDeviceMemoryType mem_type) {
    try {
      env.ReleaseSharedAllocator(ep_device, mem_type);
    } catch (...) {
      // Best-effort: a no-op when no matching shared allocator exists; a genuine failure here is
      // non-fatal during teardown, so swallow it and continue releasing the remaining allocators.
    }
  };
  try {
    for (const OrtEpDevice* ep_device : env.GetEpDevices()) {
      if (ep_device->Name() != kAMDGPUExecutionProviderName)
        continue;
      release_shared(ep_device, OrtDeviceMemoryType_DEFAULT);
      release_shared(ep_device, OrtDeviceMemoryType_HOST_ACCESSIBLE);
    }
  } catch (...) {
    // Called from noexcept destruction paths (~Model, ~OrtGlobals): never let anything escape (incl.
    // std::bad_alloc from the GetEpDevices vector). Best-effort — a failed release is non-fatal.
  }

  // With the shared allocators released, unregistering drops the factory's last reference, destroying
  // the plugin ProviderFactory and releasing the device + allocators + outstanding allocation handles.
  // The next genai model's EnsureUmbrellaEpRegistered re-creates a fresh library + device.
  try {
    Ort::UnregisterExecutionProviderLibrary(&env, kAMDGPUExecutionProviderName);
    owns_registration = false;
  } catch (...) {
    // Called from noexcept destruction paths (~Model, ~OrtGlobals): best-effort. Leave owns_registration
    // true on failure so a later teardown retries rather than orphaning the registration.
  }
}

DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& config,
                                         bool disable_graph_capture) {
  EnsureUmbrellaEpRegistered();

  SetStaticPaddingConfig(session_options, config);

  // Umbrella-level hint: the model architecture drives the EP's backend routing.
  session_options.AddConfigEntry("ep.amdgpuexecutionprovider.model_arch", config.model.type.c_str());

  // DirectML backend: host-accessible decode inputs.
  // NOTE: the DirectML backend currently force-disables host-accessible regardless of this value
  // (see dml_factory.cc CreateEpImpl), routing decode inputs through the CPU path. This request is
  // therefore advisory; the plugin owns the policy. Left as "1" as the intended request.
  session_options.AddConfigEntry("ep.directml.enable_host_accessible", "1");

  // Runtime (deferred, dynamic-shape) graph capture, applied to whichever backend the umbrella EP
  // routes to: ep.directml.enable_graph_capture for DirectX, ep.migraphx.hip_graph_enable for MIGraphX.
  // On by default for AMDGPU, opt-out per model via provider option enable_graph_capture="0"
  // (IsGraphCaptureEnabled); disable_graph_capture lets a caller force it off for non-decoder
  // sub-sessions whose control-flow nodes are incompatible with captured-graph replay. Both keys are
  // written explicitly so the effective decision overrides any backend default.
  const bool enable_graph_capture =
      IsGraphCaptureEnabled(config.model.decoder.session_options) && !disable_graph_capture;
  session_options.AddConfigEntry("ep.directml.enable_graph_capture", enable_graph_capture ? "1" : "0");
  session_options.AddConfigEntry("ep.migraphx.hip_graph_enable", enable_graph_capture ? "1" : "0");

  // Drop any cached device allocator + init session so this model rebuilds its own; otherwise the
  // dummy-allocator session reuses a prior model's allocator over an already-freed allocation.
  auto& amdgpu_allocator = GetOrtGlobals()->device_allocators_[static_cast<int>(DeviceType::AMDGPU)];
  if (!HasLiveAMDGPUModel()) {
    amdgpu_allocator.allocator_.reset();
    amdgpu_allocator.session_.reset();
    amdgpu_allocator.host_accessible_allocator_ = nullptr;
    amdgpu_allocator.device_id_ = 0;
    // Mirror the reset on the interface singleton so the rebuilt allocator's InitOrt rebinds cleanly.
    ResetAMDGPUInterfaceAllocatorState();
  } else {
    const auto requested_devices = ApplyDeviceFiltering(
        provider_options, FindRegisteredEpDevices(kAMDGPUExecutionProviderName));
    int requested_device_id = amdgpu_allocator.device_id_;
    if (!requested_devices.empty()) {
      if (const OrtMemoryInfo* memory_info =
              requested_devices.front()->GetMemoryInfo(OrtDeviceMemoryType_DEFAULT)) {
        requested_device_id = memory_info->GetDeviceId();
      }
    }
    if (requested_device_id != amdgpu_allocator.device_id_) {
      throw std::runtime_error(
          "AMDGPU: a model is already live on device " + std::to_string(amdgpu_allocator.device_id_) +
          ", but this model requested device " + std::to_string(requested_device_id) +
          ". Concurrent AMDGPU models must target the same device.");
    }
  }

  AppendExecutionProviderV2(session_options, provider_options,
                            DeviceType::AMDGPU, kAMDGPUExecutionProviderName);

  return GetAMDGPUInterface();
}

}  // namespace Generators::AMDGPUExecutionProvider
