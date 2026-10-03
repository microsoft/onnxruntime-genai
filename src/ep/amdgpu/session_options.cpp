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

// Whether genai itself registered the umbrella EP library (vs. a host pre-registering it). Guards
// the teardown-time unregister so genai never tears down a registration it does not own. See the
// header for the full rationale. Process-global; AppendExecutionProvider runs single-threaded per
// model so no synchronization is needed beyond the register path below.
bool g_genai_owns_ep_registration = false;

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
    // Already registered — by a prior genai model (we re-registered after our own teardown) or by a
    // host at startup. Either way genai is not introducing a NEW registration here, so it must not
    // claim ownership: if a host owns it, g_genai_owns_ep_registration must stay false so teardown
    // leaves the host's registration intact. If genai already owned it, the flag is already true.
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
    // genai introduced this registration -> genai owns it and may unregister it at teardown.
    g_genai_owns_ep_registration = true;
  } catch (const Ort::Exception& e) {
    // Registered but advertising no device: the check above cannot see that, ORT reports it here.
    // A concurrent/pre-existing registration raced us: genai did not introduce it, so do not take
    // ownership (leave g_genai_owns_ep_registration as-is, default false).
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

void ReleaseOwnedUmbrellaEp() {
  // The EP library and its OrtEnv-shared allocators live on the process-global OrtEnv, which is shared
  // with the host. When a host (e.g. a benchmark harness) registered the library itself, its own
  // non-OGA sessions depend on that single registration — and on the shared allocators the plugin
  // served through them — for the life of the process. Releasing either here would pull them out from
  // under the host. So genai touches the shared allocators AND unregisters only when it owns the
  // registration (i.e. EnsureUmbrellaEpRegistered introduced it); otherwise it leaves everything in
  // place and relies on the host's own teardown. Both steps share this single ownership gate so they
  // can never disagree: releasing the shared allocators only matters as the precondition that lets the
  // unregister below drop the plugin factory's last reference and destroy the device per model.
  if (!g_genai_owns_ep_registration)
    return;

  // Release the OrtEnv-shared allocators the plugin served via CreateAllocatorImpl. They are owned by
  // the process-global OrtEnv (not the EP library), so UnregisterExecutionProviderLibrary alone does
  // not drop them: the shared GPU allocator holds the plugin's ExecutionContext -> command queue ->
  // ID3D12Device, keeping the factory refcount above zero and pinning the device for the process. If a
  // model removed the GPU, that dead device would be reused by every later model. Must run while the EP
  // device is still registered (before the unregister below). Best-effort: ReleaseSharedAllocator is a
  // documented no-op when no matching shared allocator exists.
  const auto release_shared = [](const OrtEpDevice* ep_device, OrtDeviceMemoryType mem_type) {
    if (OrtStatus* status = Ort::api->ReleaseSharedAllocator(&GetOrtEnv(), ep_device, mem_type))
      Ort::api->ReleaseStatus(status);
  };
  try {
    for (const OrtEpDevice* ep_device : FindRegisteredEpDevices(kAMDGPUExecutionProviderName)) {
      release_shared(ep_device, OrtDeviceMemoryType_DEFAULT);
      release_shared(ep_device, OrtDeviceMemoryType_HOST_ACCESSIBLE);
    }
  } catch (...) {
    // Called from ~Model (noexcept): never let anything escape (incl. std::bad_alloc from the
    // FindRegisteredEpDevices vector). Best-effort — a failed shared-allocator release is non-fatal.
  }

  // With the shared allocators released, unregistering drops the factory's last reference, destroying
  // the plugin ProviderFactory and releasing the device + allocators + outstanding allocation handles.
  // The next genai model's EnsureUmbrellaEpRegistered re-creates a fresh library + device.
  try {
    Ort::UnregisterExecutionProviderLibrary(&GetOrtEnv(), kAMDGPUExecutionProviderName);
  } catch (...) {
    // Called from ~Model (noexcept): best-effort. Never let an exception escape the destructor.
  }
  // The registration is gone; the next genai model's EnsureUmbrellaEpRegistered will re-create it and
  // re-assert ownership. Clear the flag either way so we never try to double-release.
  g_genai_owns_ep_registration = false;
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

  // DirectML backend: runtime (deferred, dynamic-shape) graph fusion. Mirrors the stock
  // DML path (ep/dml/session_options.cpp) but under the ep.directml.* namespace this
  // umbrella forwards to the DirectML backend (the same namespace as enable_host_accessible
  // above). Decoder sessions default ON; the caller passes disable_graph_capture=true for
  // non-decoder sub-sessions (vision/speech) whose control-flow nodes are incompatible with
  // captured-graph replay. Per-model opt-out via provider option enable_graph_capture="0"
  // is honored through IsGraphCaptureEnabled.
  if (IsGraphCaptureEnabled(config.model.decoder.session_options) && !disable_graph_capture) {
    session_options.AddConfigEntry("ep.directml.enable_graph_capture", "1");
  }

  // Drop any cached device allocator + init session so this session rebuilds its own. OGA creates two
  // sessions per model (a trivial dummy-allocator session + the real decoder) and this runs once per
  // session; without the reset the second session reuses the first session's cached allocator, handing
  // it an OrtValue backed by an allocation the first session already freed. Safe: the prior session's
  // device buffers are freed before this runs.
  auto& amdgpu_allocator = GetOrtGlobals()->device_allocators_[static_cast<int>(DeviceType::AMDGPU)];
  amdgpu_allocator.allocator_.reset();
  amdgpu_allocator.session_.reset();
  amdgpu_allocator.host_accessible_allocator_ = nullptr;
  amdgpu_allocator.device_id_ = 0;
  // Mirror the device_allocators_ reset on the interface singleton: clear its cached allocator /
  // memory-info / pinned-allocator / device-id so the rebuilt allocator's InitOrt (which asserts
  // !ort_allocator_) and InitDeviceAllocators run cleanly for this session instead of reusing the
  // prior session's pointers. The singleton itself survives (it is p_device_ for the model lifetime).
  ResetAMDGPUInterfaceAllocatorState();

  AppendExecutionProviderV2(session_options, provider_options,
                            DeviceType::AMDGPU, kAMDGPUExecutionProviderName);

  return GetAMDGPUInterface();
}

}  // namespace Generators::AMDGPUExecutionProvider
