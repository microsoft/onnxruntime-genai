// Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "../generators.h"
#include "../search.h"
#include "../models/model.h"
#include "interface.h"
#include <filesystem>
#include <memory>
#include <span>
#include <unordered_map>

namespace Generators {
namespace AMDGPU {

// ep_name_ is the registration name amdgpu-ep's CreateEpFactories reports, the provider name in
// genai_config.json and the dispatch key in models/session_options.cpp -- all three must match.
static constexpr auto ep_path_env_key_ = "AMDGPU_EP_PATH";
static constexpr auto ep_name_ = "amdgpu";
static constexpr auto ep_memory_name_ = "AMDGPU";
static constexpr uint32_t ep_vendor_id_ = 0x1002;  // OrtDevice::VendorIds::AMD
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

  const char* GetType() const override { return ep_memory_name_; }

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
  Interface(OrtEnv& env) : env_{env} {
    // Resolve the EP library path and register it on env_ below. The EP module may already be
    // resident -- from an earlier env cycle or loaded by the host outside genai -- but ORT keys
    // EP-library registration per-OrtEnv, so a fresh env still needs the library registered on it.
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
      // check next to onnxruntime-genai.dll (use CreateAMDGPUInterface as module-address marker)
      if (const auto hmod = get_hmod_for_method(CreateAMDGPUInterface))
        ep_path_ = find_next_to_module(hmod);

    if (ep_path_.empty())
      // check next to onnxruntime.dll
      if (const auto hmod = get_hmod_for_method(Ort::api->RegisterExecutionProviderLibrary))
        ep_path_ = find_next_to_module(hmod);

    if (ep_path_.empty())
      // check next to current executable
      if (const auto hmod = GetModuleHandleA(NULL))
        ep_path_ = find_next_to_module(hmod);
#endif  // _WIN32

    if (ep_path_.empty())
      // fallback to current working directory
      ep_path_ = std::filesystem::current_path(ec) / ep_filename_;

    // Register on this env, tolerating the case where it is already registered. An OrtEnv can
    // outlive a genai OgaShutdown (the host may hold a reference), so on re-init the same env may
    // already have the library; ORT reports that as "library is already registered under ...".
    try {
      Ort::RegisterExecutionProviderLibrary(&env_, ep_name_, ep_path_.native().c_str());
    } catch (const Ort::Exception& e) {
      const std::string message = e.what();
      if (message.find("already registered") == std::string::npos)
        throw std::runtime_error("Failed to register AMD GPU execution provider library: " + message);
    }
  }

  ~Interface() {
    // Clear the process-global allocator pointer so a subsequent re-init (new interface on a fresh
    // env) starts from a clean state: InitOrt()'s assert holds again and no Memory allocation can
    // dangle against this env's now-destroyed session allocator.
    ort_allocator_ = nullptr;
  }

  void SetupProvider(OrtSessionOptions& session_options, const ProviderOptions& provider_options) override {
    std::vector<const OrtEpDevice*> supported_devices;

    {
      const OrtEpDevice* const* devices = nullptr;
      size_t ndevices = 0;

      Ort::GetEpDevices(&env_, &devices, &ndevices);

      // Vendor id matching is left to the EP factory's device enumeration, which already restricts
      // itself to AMD GPUs.
      for (const auto& device : std::span{devices, ndevices})
        if (std::string_view{ep_name_} == device->Name() &&
            OrtHardwareDeviceType_GPU == device->Device()->Type())
          supported_devices.push_back(device);
    }

    if (supported_devices.empty())
      throw std::runtime_error{"No AMD GPU devices detected"};

    std::unordered_map<std::string, std::string> ep_options;
    for (const auto& option : provider_options) {
      ep_options.emplace(option.first, option.second);
      // The umbrella EP selects its backend from session config entries prefixed with
      // "ep.<registration name>.", so mirror the provider options there as well.
      session_options.AddConfigEntry((std::string{"ep."} + ep_name_ + "." + option.first).c_str(),
                                     option.second.c_str());
    }

    session_options.AppendExecutionProvider_V2(env_, supported_devices, ep_options);
  }

  void ShapeInitSessionProviderOptions(Config::ProviderOptions& init_options,
                                       const Config::ProviderOptions* user_options) const override {
    // The umbrella EP selects a backend from its provider options, so the trivial init session needs
    // the same options the model sessions get; with none of them it has no backend to allocate from.
    if (user_options)
      init_options.options = user_options->options;
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
  OrtEnv& env_;  // The env this interface belongs to; valid until OrtGlobals tears this interface down.
  std::filesystem::path ep_path_;
};

}  // namespace AMDGPU

std::unique_ptr<OrtMemoryInfo> AMDGPUInterface::GetMemoryInfo() const {
  // The umbrella EP publishes its allocator through the plugin EP API (CreateMemoryInfo_V2 plus
  // EpDevice_AddAllocatorInfo). OrtMemoryInfo::Create only knows ORT-internal device names, so the
  // allocator lookup has to use the same V2 parameters the EP factory registered with.
  OrtMemoryInfo* memory_info = nullptr;
  Ort::ThrowOnError(Ort::api->CreateMemoryInfo_V2(AMDGPU::ep_memory_name_, OrtMemoryInfoDeviceType_GPU,
                                                  AMDGPU::ep_vendor_id_,
                                                  /*device_id*/ 0,
                                                  OrtDeviceMemoryType_DEFAULT,
                                                  /*alignment*/ 0,
                                                  OrtAllocatorType::OrtDeviceAllocator,
                                                  &memory_info));
  return std::unique_ptr<OrtMemoryInfo>{memory_info};
}

std::unique_ptr<DeviceInterface> CreateAMDGPUInterface(OrtEnv& env) {
  return std::make_unique<AMDGPU::Interface>(env);
}

}  // namespace Generators
