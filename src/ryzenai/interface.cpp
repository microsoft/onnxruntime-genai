// Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "../generators.h"
#include "../search.h"
#include "../models/model.h"
#include "interface.h"
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>

namespace Generators {
namespace RyzenAI {

static constexpr auto ep_path_env_key_ = "RYZENAI_EP_PATH";
static constexpr auto ep_name_ = "RyzenAILightExecutionProvider";
#if defined(_WIN32)
static constexpr auto ep_filename_ = "onnxruntime_providers_ryzenai.dll";
#else
static constexpr auto ep_filename_ = "libonnxruntime_providers_ryzenai.so";
#endif
static constexpr auto func_shutdown_ = "RyzenAI_Shutdown";

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

  const char* GetType() const override { return "RyzenAI"; }

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
struct Interface : RyzenAIInterface {
  Interface(OrtEnv& env) : env_{env} {
#if defined(_WIN32)
    // Record whether the EP module is already resident, before our registration below loads it. If it
    // was not resident, our registration loads it and genai owns shutting it down in the destructor;
    // if it was already resident (loaded by the host outside genai), another owner is responsible.
    // ORT unloads the EP library when this interface's env is torn down, so each fresh interface
    // re-evaluates this against a genuinely unloaded module on the next re-init cycle.
    owns_ep_module_ = (GetModuleHandleA(ep_filename_) == nullptr);
#endif

    // Resolve the EP library path (SetupProvider also needs it for RegisterCustomOpsLibrary_V2) and
    // register it on env_ below. The EP module may already be resident -- from an earlier env cycle or
    // loaded by the host outside genai -- but ORT keys EP-library registration per-OrtEnv, so a fresh
    // env still needs the library registered on it.
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
      // check next to onnxruntime-genai.dll (use CreateRyzenAIInterface as module-address marker)
      if (const auto hmod = get_hmod_for_method(CreateRyzenAIInterface))
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

    // Register on this env, tolerating the case where it is already registered. An OrtEnv can outlive
    // a genai OgaShutdown (the host may hold a reference), so on re-init the same env may already have
    // the library; ORT reports that as "library is already registered under ...", which is benign here.
    if (auto status = std::unique_ptr<OrtStatus>{
            Ort::api->RegisterExecutionProviderLibrary(&env_, ep_name_, ep_path_.native().c_str())}) {
      const std::string message = status->GetErrorMessage();
      if (message.find("already registered") == std::string::npos)
        throw std::runtime_error("Failed to register RyzenAI execution provider library: " + message);
    }
  }

  ~Interface() {
    // TODO: make it linux compatible
#if defined(_WIN32)
    // Only shut the EP down if genai loaded the module. If it was already resident when this interface
    // was constructed, another owner is responsible for it and shutting it down here could pull state
    // out from under them.
    if (owns_ep_module_)
      if (const auto mod = GetModuleHandleA(ep_filename_))
        if (const auto func = reinterpret_cast<void (*)()>(GetProcAddress(mod, func_shutdown_)))
          func();
#endif  // _WIN32
  }

  void SetupProvider(OrtSessionOptions& session_options, const ProviderOptions& provider_options) override {
    std::vector<const OrtEpDevice*> supported_devices;

    {
      const OrtEpDevice* const* devices = nullptr;
      size_t ndevices = 0;

      Ort::ThrowOnError(Ort::api->GetEpDevices(&env_, &devices, &ndevices));

      for (const auto& device : std::span{devices, ndevices})
        if (std::string_view{ep_name_} == Ort::api->EpDevice_EpName(device) &&
            OrtHardwareDeviceType_NPU == Ort::api->HardwareDevice_Type(Ort::api->EpDevice_Device(device)))
          supported_devices.push_back(device);
    }

    if (supported_devices.empty())
      throw std::runtime_error{"No RyzenAI devices detected"};

    {
      std::vector<const char*> ep_keys, ep_values;

      for (auto& option : provider_options) {
        ep_keys.emplace_back(option.first.c_str());
        ep_values.emplace_back(option.second.c_str());
      }

      // this call merges provider_options into session_options
      Ort::ThrowOnError(Ort::api->SessionOptionsAppendExecutionProvider_V2(&session_options,
                                                                           &env_, supported_devices.data(),
                                                                           supported_devices.size(),
                                                                           ep_keys.data(), ep_values.data(), ep_keys.size()));
    }

    Ort::ThrowOnError(Ort::api->RegisterCustomOpsLibrary_V2(&session_options, ep_path_.native().c_str()));
  }

  DeviceType GetType() const override { return DeviceType::RyzenAI; }

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
#if defined(_WIN32)
  bool owns_ep_module_{false};  // True if genai's registration loaded the EP module, so genai shuts it down.
#endif
};

}  // namespace RyzenAI

std::unique_ptr<DeviceInterface> CreateRyzenAIInterface(OrtEnv& env) {
  return std::make_unique<RyzenAI::Interface>(env);
}

}  // namespace Generators
