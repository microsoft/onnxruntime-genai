// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#include "../generators.h"
#include "../ort_genai_c.h"
#include "../search.h"
#include "../models/model.h"
#include "interface.h"
#include <filesystem>
#include <mutex>
#include <span>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace Generators {
namespace AMDGPU {

// Mirrors ryzenai/interface.cpp. The behavioural differences are:
//   - ep_name_ matches the registration_name passed to amdgpu-ep's
//     CreateEpFactories (== "amdgpu" -- what callers like model_benchmark
//     use with --ep_library, what genai_config.json uses as a provider
//     name, and what session_options.cpp's dispatch table keys on).
//   - SetupProvider filters OrtHardwareDeviceType_GPU; vendor id matching
//     is delegated to the EP factory's GetSupportedDevicesImpl, which
//     already restricts itself to AMD GPU vendor id 0x1002.
//   - DeviceType::AMDGPU is returned from GetType().
//   - Allocator-backed memory is treated as both host- and device-
//     accessible (true on AMD APU iGPU; same simplification RyzenAI uses).
static constexpr auto ep_path_env_key_ = "AMDGPU_EP_PATH";
static constexpr auto ep_name_ = "amdgpu";
#if defined(_WIN32)
static constexpr auto ep_filename_ = "amdgpu-ep.dll";
#else
static constexpr auto ep_filename_ = "libamdgpu-ep.so";
#endif

static Ort::Allocator* ort_allocator_{};

struct Memory : DeviceBuffer {
  Memory(size_t size) : owned_{true} {
    size_in_bytes_ = size;
    p_cpu_ = p_device_ = static_cast<uint8_t*>(ort_allocator_->Alloc(size_in_bytes_));
  }

  Memory(void* p, size_t size) : owned_{false} {
    size_in_bytes_ = size;
    p_cpu_ = p_device_ = static_cast<uint8_t*>(p);
  }

  ~Memory() override {
    if (owned_)
      ort_allocator_->Free(p_device_);
  }

  const char* GetType() const override { return "AMDGPU"; }

  void AllocateCpu() override {}
  void CopyDeviceToCpu() override {}
  void CopyCpuToDevice() override {}

  void CopyFrom(size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) override {
    CopyThroughCpu(*this, begin_dest, source, begin_source, size_in_bytes);
  }

  void Zero() override {
    memset(p_device_, 0, size_in_bytes_);
  }

  bool owned_;
};

struct Interface : AMDGPUInterface {
  Interface() {
    ep_path_ = ep_filename_;
    // If the EP DLL is already loaded by the host, there is nothing to do.
#if defined(_WIN32)
    if (GetModuleHandleA(ep_filename_))
      return;
#else
    if (auto handle = dlopen(ep_filename_, RTLD_NOLOAD | RTLD_NOW)) {
      dlclose(handle);
      return;
    }
#endif

    std::error_code ec;

    ep_path_ = GetEnv(ep_path_env_key_);

#if defined(_WIN32)
    const auto get_hmod_for_method = [](LPCVOID func) -> HMODULE {
      MEMORY_BASIC_INFORMATION mbi;

      if (VirtualQuery(func, &mbi, sizeof(mbi)) && mbi.AllocationBase)
        return (HMODULE)mbi.AllocationBase;

      return nullptr;
    };

    const auto find_next_to_module = [&](HMODULE hmod) -> std::filesystem::path {
      wchar_t buffer[MAX_PATH + 1] = {0};
      const auto len = sizeof(buffer) / sizeof(buffer[0]);

      if (GetModuleFileNameW(hmod, buffer, len))
        if (const auto dir = std::filesystem::path{buffer}.remove_filename(); !dir.empty())
          if (auto path = dir / ep_filename_; std::filesystem::exists(path, ec))
            return path;

      return {};
    };

    if (ep_path_.empty())
      // Check next to onnxruntime-genai.dll.
      if (const auto hmod = get_hmod_for_method(GetAMDGPUInterface))
        ep_path_ = find_next_to_module(hmod);

    if (ep_path_.empty())
      // Check next to onnxruntime.dll.
      if (const auto hmod = get_hmod_for_method(Ort::api->RegisterExecutionProviderLibrary))
        ep_path_ = find_next_to_module(hmod);

    if (ep_path_.empty())
      // Check next to the current executable.
      if (const auto hmod = GetModuleHandleA(NULL))
        ep_path_ = find_next_to_module(hmod);
#endif  // _WIN32

    if (ep_path_.empty())
      // Fall back to the current working directory.
      ep_path_ = std::filesystem::current_path(ec) / ep_filename_;

    OgaRegisterExecutionProviderLibrary(ep_name_, ep_path_.string().c_str());
  }

  ~Interface() {
  }

  void SetupProvider(OrtSessionOptions& session_options, const ProviderOptions& provider_options) override {
    std::vector<const OrtEpDevice*> supported_devices;

    {
      const OrtEpDevice* const* devices = nullptr;
      size_t ndevices = 0;

      Ort::ThrowOnError(Ort::api->GetEpDevices(&GetOrtEnv(), &devices, &ndevices));

      for (const auto& device : std::span{devices, ndevices}) {
        if (std::string_view{ep_name_} != Ort::api->EpDevice_EpName(device))
          continue;
        const auto* hw = Ort::api->EpDevice_Device(device);
        if (Ort::api->HardwareDevice_Type(hw) != OrtHardwareDeviceType_GPU)
          continue;
        supported_devices.push_back(device);
      }
    }

    if (supported_devices.empty())
      throw std::runtime_error{"No AMDGPU-supported AMD GPU devices detected"};

    std::vector<const char*> ep_keys, ep_values;
    std::vector<std::string> config_keys;

    // The umbrella EP reads provider options from session config entries prefixed
    // with "ep.<name>.", so mirror them there in addition to passing them through
    // the plugin EP V2 metadata below.
    for (auto& option : provider_options) {
      ep_keys.emplace_back(option.first.c_str());
      ep_values.emplace_back(option.second.c_str());
      config_keys.emplace_back(std::string{"ep."} + ep_name_ + "." + option.first);
    }

    for (size_t i = 0; i < config_keys.size(); ++i) {
      Ort::ThrowOnError(Ort::api->AddSessionConfigEntry(
          &session_options, config_keys[i].c_str(), ep_values[i]));
    }

    Ort::ThrowOnError(Ort::api->SessionOptionsAppendExecutionProvider_V2(
        &session_options, &GetOrtEnv(), supported_devices.data(), supported_devices.size(),
        ep_keys.data(), ep_values.data(), ep_keys.size()));
  }

  DeviceType GetType() const override { return DeviceType::AMDGPU; }

  void InitOrt(const OrtApi& /*api*/, Ort::Allocator& allocator) override {
    assert(!ort_allocator_);
    ort_allocator_ = &allocator;
  }

  Ort::Allocator& GetAllocator() override {
    return *ort_allocator_;
  }

  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return std::make_shared<Memory>(size);
  }

  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* p, size_t size) override {
    return std::make_shared<Memory>(p, size);
  }

  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override { return std::make_unique<GreedySearch_Cpu>(params); }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override { return std::make_unique<BeamSearch_Cpu>(params); }

  void Synchronize() override {}

 private:
  std::filesystem::path ep_path_;
};

static std::unique_ptr<Interface> interface_;

}  // namespace AMDGPU

void AMDGPUInterface::Shutdown() {
  AMDGPU::interface_.reset();
}

AMDGPUInterface* GetAMDGPUInterface() {
  static std::once_flag once;

  std::call_once(once, []() {
    AMDGPU::interface_ = std::make_unique<AMDGPU::Interface>();
  });

  return AMDGPU::interface_.get();
}

}  // namespace Generators
