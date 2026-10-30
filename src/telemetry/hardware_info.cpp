// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "hardware_info.h"

#if defined(ORTGENAI_ENABLE_TELEMETRY)

#include "telemetry_io.h"
#include "telemetry_redaction.h"
#include "telemetry_string.h"
#include "device_info.h"
#include "../generator/generators.h"
#include "../logging.h"
#include "../models/onnxruntime_api.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <thread>

#if defined(__linux__)
#include <sys/stat.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <initguid.h>
#include <dxcore.h>
#include <psapi.h>
#include <wrl/client.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <unistd.h>
#else
#include <dlfcn.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace Generators {
namespace {

constexpr uint64_t kBytesPerMB = 1024 * 1024;
constexpr size_t kMaxProbeBytes = 64 * 1024;

void Diagnostic(std::string_view message) noexcept {
  if (!g_log.enabled) {
    return;
  }

  try {
    Log("debug", message);
  } catch (...) {
  }
}

int64_t ToMB(uint64_t bytes) {
  return static_cast<int64_t>(bytes / kBytesPerMB);
}

std::string VendorName(int64_t vendor_id) {
  switch (vendor_id) {
    case 0x10de:
      return "NVIDIA";
    case 0x1002:
      return "AMD";
    case 0x8086:
      return "Intel";
    case 0x106b:
      return "Apple";
    case 0x17cb:
      return "Qualcomm";
    default:
      return "unknown";
  }
}

void AddDevice(HardwareInfo& info, HardwareDeviceInfo device) {
  if (info.devices.size() >= kMaxHardwareInventoryEntries) {
    info.inventory_truncated = true;
    return;
  }

  device.name = ScrubStringForTelemetry(device.name);
  device.vendor = ScrubStringForTelemetry(device.vendor);
  device.driver_version = ScrubStringForTelemetry(device.driver_version);
  device.source = BoundTelemetryString(device.source);
  info.devices.push_back(std::move(device));
}

#if defined(__linux__)
std::string ReadText(const std::filesystem::path& path, size_t limit = kMaxProbeBytes) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return {};
  }

  std::string text(limit + 1, '\0');
  file.read(text.data(), static_cast<std::streamsize>(text.size()));
  text.resize(static_cast<size_t>(file.gcount()));
  if (text.size() > limit) {
    text.resize(limit);
    const auto last_line = text.rfind('\n');
    text.resize(last_line == std::string::npos ? 0 : last_line + 1);
  }

  return text;
}
#endif

#if defined(_WIN32)

std::string CpuRegistryString(const char* name) {
  std::array<char, kMaxTelemetryStringLength + 1> value{};
  DWORD size = static_cast<DWORD>(value.size());
  if (::RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                     name, RRF_RT_REG_SZ, nullptr, value.data(), &size) != ERROR_SUCCESS) {
    return "unknown";
  }

  return ScrubStringForTelemetry(TelemetryInternal::TrimAscii(
      BoundedTelemetryCString(value.data(), value.size())));
}

HardwareDeviceInfo ReadCpuInfo() {
  HardwareDeviceInfo cpu;
  cpu.type = "CPU";
  cpu.source = "windows";
  cpu.name = CpuRegistryString("ProcessorNameString");
  cpu.vendor = CpuRegistryString("VendorIdentifier");

  DWORD mhz = 0;
  DWORD size = sizeof(mhz);
  if (::RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                     "~MHz", RRF_RT_REG_DWORD, nullptr, &mhz, &size) == ERROR_SUCCESS) {
    cpu.nominal_clock_mhz = mhz;
  }

  DWORD bytes = 0;
  ::GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
  if (bytes > 0 && bytes <= 1024 * 1024) {
    std::vector<unsigned char> buffer(bytes);
    if (::GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &bytes)) {
      int64_t cores = 0;
      size_t offset = 0;
      constexpr auto header_size =
          offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor);
      while (offset + header_size <= bytes) {
        const auto* entry =
            reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
                buffer.data() + offset);
        if (entry->Size < header_size + sizeof(PROCESSOR_RELATIONSHIP) ||
            entry->Size > bytes - offset) {
          cores = -1;
          break;
        }

        ++cores;
        offset += entry->Size;
      }

      if (offset == bytes && cores > 0) {
        cpu.physical_core_count = cores;
      }
    }
  }

  return cpu;
}

class SystemModule {
 public:
  explicit SystemModule(HMODULE handle) : handle_{handle} {}
  ~SystemModule() {
    if (handle_) {
      ::FreeLibrary(handle_);
    }
  }

  SystemModule(const SystemModule&) = delete;
  SystemModule& operator=(const SystemModule&) = delete;

  HMODULE Get() const {
    return handle_;
  }

 private:
  HMODULE handle_{};
};

struct AdapterMemory {
  uint64_t local_budget{};
  uint64_t local_used{};
  uint64_t nonlocal_used{};
  bool available{};
};

AdapterMemory ReadAdapterMemory(IDXCoreAdapter& adapter) {
  AdapterMemory memory;
  if (!adapter.IsQueryStateSupported(DXCoreAdapterState::AdapterMemoryBudget)) {
    return memory;
  }

  uint32_t node_count = 1;
  constexpr auto physical_adapter_count = static_cast<DXCoreAdapterProperty>(15);
  if (adapter.IsPropertySupported(physical_adapter_count) &&
      FAILED(adapter.GetProperty(physical_adapter_count, &node_count))) {
    return memory;
  }
  if (node_count == 0 || node_count > kMaxHardwareInventoryEntries) {
    return memory;
  }

  for (uint32_t node = 0; node < node_count; ++node) {
    DXCoreAdapterMemoryBudgetNodeSegmentGroup group{node, DXCoreSegmentGroup::Local};
    DXCoreAdapterMemoryBudget local{};
    DXCoreAdapterMemoryBudget nonlocal{};
    if (FAILED(adapter.QueryState(DXCoreAdapterState::AdapterMemoryBudget, &group, &local))) {
      return {};
    }

    group.segmentGroup = DXCoreSegmentGroup::NonLocal;
    if (FAILED(adapter.QueryState(DXCoreAdapterState::AdapterMemoryBudget, &group, &nonlocal))) {
      return {};
    }

    memory.local_budget += local.budget;
    memory.local_used += local.currentUsage;
    memory.nonlocal_used += nonlocal.currentUsage;
  }

  memory.available = true;
  return memory;
}

struct DxCoreInventory {
  // Declared first so it is destroyed last, after every DXCore COM object.
  std::unique_ptr<SystemModule> library;
  std::vector<std::pair<Microsoft::WRL::ComPtr<IDXCoreAdapter>, std::string>> adapters;
  bool available{};
  bool complete{true};
  bool truncated{};
};

DxCoreInventory ReadDxCoreInventory() {
  DxCoreInventory result;
  result.library = std::make_unique<SystemModule>(
      ::LoadLibraryExW(L"dxcore.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32));
  if (!result.library->Get()) {
    Diagnostic("Telemetry DXCore APIs are unavailable; accelerator fields are unknown");
    return result;
  }

  using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
  const auto create_factory =
      reinterpret_cast<CreateFactory>(
          ::GetProcAddress(result.library->Get(), "DXCoreCreateAdapterFactory"));
  Microsoft::WRL::ComPtr<IDXCoreAdapterFactory> factory;
  if (!create_factory || FAILED(create_factory(IID_PPV_ARGS(&factory)))) {
    Diagnostic("Telemetry could not create the DXCore adapter factory");
    return result;
  }

  constexpr GUID generic_ml{0xb71b0d41, 0x1088, 0x422f, {0xa2, 0x7c, 0x02, 0x50, 0xb7, 0xd3, 0xa9, 0x88}};
  constexpr GUID npu{0xd46140c4, 0xadd7, 0x451b, {0x9e, 0x56, 0x06, 0xfe, 0x8c, 0x3b, 0x58, 0xed}};
  constexpr GUID gpu{0xb69eb219, 0x3ded, 0x4464, {0x97, 0x9f, 0xa0, 0x0b, 0xd4, 0x68, 0x70, 0x06}};
  const std::array attributes{DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE, generic_ml,
                              npu, gpu, DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS};
  std::set<uint64_t> seen;
  for (const auto& attribute : attributes) {
    Microsoft::WRL::ComPtr<IDXCoreAdapterList> list;
    if (FAILED(factory->CreateAdapterList(1, &attribute, IID_PPV_ARGS(&list)))) {
      result.complete = false;
      continue;
    }

    result.available = true;
    for (uint32_t index = 0; index < list->GetAdapterCount(); ++index) {
      if (index >= kMaxHardwareInventoryEntries) {
        result.truncated = true;
        break;
      }

      Microsoft::WRL::ComPtr<IDXCoreAdapter> adapter;
      bool hardware = false;
      LUID luid{};
      if (FAILED(list->GetAdapter(index, IID_PPV_ARGS(&adapter))) ||
          FAILED(adapter->GetProperty(DXCoreAdapterProperty::IsHardware, &hardware)) ||
          FAILED(adapter->GetProperty(DXCoreAdapterProperty::InstanceLuid, &luid))) {
        result.complete = false;
        continue;
      }
      if (!hardware) {
        continue;
      }

      const auto key =
          (static_cast<uint64_t>(static_cast<uint32_t>(luid.HighPart)) << 32) |
          luid.LowPart;
      if (!seen.insert(key).second) {
        continue;
      }
      if (result.adapters.size() >= kMaxHardwareInventoryEntries) {
        result.truncated = true;
        break;
      }

      std::string type = "Unknown";
      if (adapter->IsAttributeSupported(npu)) {
        type = "NPU";
      } else if (adapter->IsAttributeSupported(gpu) ||
                 adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS)) {
        type = "GPU";
      }
      result.adapters.emplace_back(std::move(adapter), std::move(type));
    }
  }

  return result;
}

void ReadWindowsAccelerators(HardwareInfo& info) {
  auto inventory = ReadDxCoreInventory();
  info.inventory_truncated |= inventory.truncated || !inventory.complete;
  for (const auto& [adapter, type] : inventory.adapters) {
    HardwareDeviceInfo device;
    device.type = type;
    device.source = "dxcore";

    DXCoreHardwareID id{};
    if (SUCCEEDED(adapter->GetProperty(DXCoreAdapterProperty::HardwareID, &id))) {
      device.vendor_id = id.vendorID;
      device.device_id = id.deviceID;
      device.vendor = VendorName(device.vendor_id);
    }

    size_t size = 0;
    if (SUCCEEDED(adapter->GetPropertySize(DXCoreAdapterProperty::DriverDescription, &size)) &&
        size > 0 && size <= kMaxTelemetryStringLength + 1) {
      std::string name(size, '\0');
      if (SUCCEEDED(adapter->GetProperty(DXCoreAdapterProperty::DriverDescription,
                                         name.size(), name.data()))) {
        device.name = ScrubStringForTelemetry(BoundedTelemetryCString(name.c_str(), name.size()));
      }
    }

    uint64_t version = 0;
    if (SUCCEEDED(adapter->GetProperty(DXCoreAdapterProperty::DriverVersion, &version))) {
      device.driver_version =
          std::to_string((version >> 48) & 0xffff) + "." +
          std::to_string((version >> 32) & 0xffff) + "." +
          std::to_string((version >> 16) & 0xffff) + "." +
          std::to_string(version & 0xffff);
    }

    uint64_t bytes = 0;
    if (SUCCEEDED(adapter->GetProperty(DXCoreAdapterProperty::DedicatedAdapterMemory, &bytes))) {
      device.dedicated_memory_mb = ToMB(bytes);
    }
    if (SUCCEEDED(adapter->GetProperty(DXCoreAdapterProperty::SharedSystemMemory, &bytes))) {
      device.shared_memory_mb = ToMB(bytes);
    }

    const auto memory = ReadAdapterMemory(*adapter.Get());
    if (memory.available) {
      device.memory_budget_mb = ToMB(memory.local_budget);
      device.memory_budget_available_mb =
          ToMB(memory.local_budget > memory.local_used
                   ? memory.local_budget - memory.local_used
                   : 0);
    }

    AddDevice(info, std::move(device));
  }
}

#elif defined(__APPLE__)

template <typename T>
bool ReadSysctl(const char* key, T& value) {
  size_t size = sizeof(value);
  return ::sysctlbyname(key, &value, &size, nullptr, 0) == 0 && size == sizeof(value);
}

HardwareDeviceInfo ReadCpuInfo() {
  HardwareDeviceInfo cpu;
  cpu.type = "CPU";
  cpu.source = "sysctl";
  std::array<char, kMaxTelemetryStringLength + 1> name{};
  size_t size = name.size();
  if (::sysctlbyname("machdep.cpu.brand_string", name.data(), &size, nullptr, 0) == 0) {
    cpu.name = ScrubStringForTelemetry(BoundedTelemetryCString(name.data(), name.size()));
  }
#if defined(__aarch64__)
  cpu.vendor = "Apple";
#endif
  uint32_t cores = 0;
  if (ReadSysctl("hw.physicalcpu", cores)) {
    cpu.physical_core_count = cores;
  }
  uint64_t frequency = 0;
  if (ReadSysctl("hw.cpufrequency_max", frequency)) {
    cpu.max_clock_mhz = static_cast<int64_t>(frequency / 1000000);
  }
  return cpu;
}

#elif defined(__linux__)

int64_t ReadPhysicalCoreCount() {
  std::error_code error;
  std::filesystem::directory_iterator it{"/sys/devices/system/cpu", error};
  std::set<std::pair<int64_t, int64_t>> cores;
  size_t inspected = 0;
  for (; !error && it != std::filesystem::directory_iterator{}; it.increment(error)) {
    if (++inspected > 4096) {
      return -1;
    }
    const auto name = it->path().filename().string();
    if (name.compare(0, 3, "cpu") != 0 ||
        TelemetryInternal::ParseHardwareInteger(std::string_view{name}.substr(3)) < 0) {
      continue;
    }
    const auto topology = it->path() / "topology";
    const auto package =
        TelemetryInternal::ParseHardwareInteger(ReadText(topology / "physical_package_id"));
    const auto core =
        TelemetryInternal::ParseHardwareInteger(ReadText(topology / "core_id"));
    if (package < 0 || core < 0) {
      return -1;
    }
    cores.emplace(package, core);
  }
  return !error && !cores.empty() ? static_cast<int64_t>(cores.size()) : -1;
}

HardwareDeviceInfo ReadCpuInfo() {
  auto cpu = TelemetryInternal::ParseCpuInfo(ReadText("/proc/cpuinfo"));
  cpu.source = "proc_sysfs";
  cpu.physical_core_count = ReadPhysicalCoreCount();
  const auto khz = TelemetryInternal::ParseHardwareInteger(
      ReadText("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"));
  if (khz >= 0) {
    cpu.max_clock_mhz = khz / 1000;
  }
  return cpu;
}

using NvmlDevice = void*;
struct NvmlMemory {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
};
struct NvmlProcessInfo {
  unsigned int pid;
  unsigned long long used_gpu_memory;
  unsigned int gpu_instance_id;
  unsigned int compute_instance_id;
};

class NvmlLibrary {
 public:
  NvmlLibrary() {
    library_ = ::dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!library_) {
      Diagnostic("Telemetry NVML library is unavailable; NVIDIA fields are unknown");
      return;
    }
    init_ = Symbol<int (*)()>("nvmlInit_v2");
    shutdown_ = Symbol<int (*)()>("nvmlShutdown");
    get_count_ = Symbol<int (*)(unsigned int*)>("nvmlDeviceGetCount_v2");
    get_handle_ =
        Symbol<int (*)(unsigned int, NvmlDevice*)>("nvmlDeviceGetHandleByIndex_v2");
    ready_ = init_ && shutdown_ && get_count_ && get_handle_ && init_() == 0;
    if (!ready_) {
      Diagnostic("Telemetry NVML initialization failed; NVIDIA fields are unknown");
    }
  }

  ~NvmlLibrary() {
    if (ready_) {
      shutdown_();
    }
    if (library_) {
      ::dlclose(library_);
    }
  }

  NvmlLibrary(const NvmlLibrary&) = delete;
  NvmlLibrary& operator=(const NvmlLibrary&) = delete;

  void AddHardware(HardwareInfo& info) const {
    unsigned int count = 0;
    if (!ready_ || get_count_(&count) != 0) {
      return;
    }
    info.inventory_truncated |= count > kMaxHardwareInventoryEntries;
    const auto get_name = Symbol<int (*)(NvmlDevice, char*, unsigned int)>(
        "nvmlDeviceGetName");
    const auto get_memory =
        Symbol<int (*)(NvmlDevice, NvmlMemory*)>("nvmlDeviceGetMemoryInfo");
    const auto get_driver =
        Symbol<int (*)(char*, unsigned int)>("nvmlSystemGetDriverVersion");
    const auto get_max_clock =
        Symbol<int (*)(NvmlDevice, int, unsigned int*)>("nvmlDeviceGetMaxClockInfo");
    const auto get_clock =
        Symbol<int (*)(NvmlDevice, int, unsigned int*)>("nvmlDeviceGetClockInfo");
    const auto get_bus =
        Symbol<int (*)(NvmlDevice, unsigned int*)>("nvmlDeviceGetMemoryBusWidth");
    std::array<char, kMaxTelemetryStringLength + 1> driver{};
    const bool has_driver = get_driver && get_driver(driver.data(), driver.size()) == 0;
    for (unsigned int index = 0;
         index < (std::min)(count, static_cast<unsigned int>(kMaxHardwareInventoryEntries));
         ++index) {
      NvmlDevice handle = nullptr;
      if (get_handle_(index, &handle) != 0) {
        continue;
      }
      HardwareDeviceInfo device;
      device.type = "GPU";
      device.vendor = "NVIDIA";
      device.vendor_id = 0x10de;
      device.source = "nvml";
      std::array<char, kMaxTelemetryStringLength + 1> name{};
      if (get_name && get_name(handle, name.data(), name.size()) == 0) {
        device.name = ScrubStringForTelemetry(
            BoundedTelemetryCString(name.data(), name.size()));
      }
      if (has_driver) {
        device.driver_version = ScrubStringForTelemetry(
            BoundedTelemetryCString(driver.data(), driver.size()));
      }
      NvmlMemory memory{};
      if (get_memory && get_memory(handle, &memory) == 0) {
        device.dedicated_memory_mb = ToMB(memory.total);
        device.free_memory_mb = ToMB(memory.free);
      }
      unsigned int value = 0;
      if (get_max_clock && get_max_clock(handle, 0, &value) == 0) {
        device.max_clock_mhz = value;
      }
      if (get_clock && get_clock(handle, 2, &value) == 0) {
        device.memory_clock_mhz = value;
      }
      if (get_bus && get_bus(handle, &value) == 0) {
        device.memory_bus_width_bits = value;
      }
      AddDevice(info, std::move(device));
    }
  }

  int64_t ProcessMemoryMB() const {
    using Query = int (*)(NvmlDevice, unsigned int*, NvmlProcessInfo*);
    const auto compute =
        Symbol<Query>("nvmlDeviceGetComputeRunningProcesses_v3");
    const auto graphics =
        Symbol<Query>("nvmlDeviceGetGraphicsRunningProcesses_v3");
    unsigned int count = 0;
    if (!ready_ || !compute || !graphics || get_count_(&count) != 0 ||
        count > kMaxHardwareInventoryEntries) {
      return -1;
    }

    uint64_t bytes = 0;
    for (unsigned int index = 0; index < count; ++index) {
      NvmlDevice device = nullptr;
      if (get_handle_(index, &device) != 0) {
        return -1;
      }
      std::map<std::pair<unsigned int, unsigned int>, uint64_t> instances;
      for (const auto query : {compute, graphics}) {
        std::array<NvmlProcessInfo, 1024> processes{};
        auto process_count = static_cast<unsigned int>(processes.size());
        if (query(device, &process_count, processes.data()) != 0 ||
            process_count > processes.size()) {
          return -1;
        }
        for (unsigned int process_index = 0; process_index < process_count;
             ++process_index) {
          const auto& process = processes[process_index];
          if (process.pid != static_cast<unsigned int>(::getpid())) {
            continue;
          }
          if (process.used_gpu_memory ==
              (std::numeric_limits<unsigned long long>::max)()) {
            return -1;
          }

          auto& used =
              instances[{process.gpu_instance_id, process.compute_instance_id}];
          used = (std::max)(used, static_cast<uint64_t>(process.used_gpu_memory));
        }
      }
      for (const auto& [instance, used] : instances) {
        if (!TelemetryInternal::AccumulateProcessMemoryBytes(bytes, used)) {
          return -1;
        }
      }
    }
    return ToMB(bytes);
  }

 private:
  template <typename T>
  T Symbol(const char* name) const {
    return reinterpret_cast<T>(library_ ? ::dlsym(library_, name) : nullptr);
  }

  void* library_{};
  bool ready_{};
  int (*init_)(){};
  int (*shutdown_)(){};
  int (*get_count_)(unsigned int*){};
  int (*get_handle_)(unsigned int, NvmlDevice*){};
};

#else

HardwareDeviceInfo ReadCpuInfo() {
  HardwareDeviceInfo cpu;
  cpu.type = "CPU";
  cpu.source = "unavailable";
  return cpu;
}

#endif

void ReadOrtDevices(HardwareInfo& info) {
  constexpr uint32_t kOrtHardwareDeviceApiVersion = 22;
  if (!Ort::api || Ort::api_version < kOrtHardwareDeviceApiVersion) {
    Diagnostic("Telemetry ORT hardware metadata API is unavailable");
    return;
  }

  const OrtEpDevice* const* devices = nullptr;
  size_t count = 0;
  try {
    Ort::GetEpDevices(&GetOrtEnv(), &devices, &count);
  } catch (...) {
    Diagnostic("Telemetry could not read ORT hardware inventory");
    return;
  }

  std::set<const OrtHardwareDevice*> seen;
  std::vector<HardwareDeviceInfo> ort_devices;
  for (size_t index = 0; index < (std::min)(count, kMaxHardwareInventoryEntries);
       ++index) {
    const auto* hardware = Ort::api->EpDevice_Device(devices[index]);
    if (!hardware || !seen.insert(hardware).second) {
      continue;
    }

    HardwareDeviceInfo device;
    switch (Ort::api->HardwareDevice_Type(hardware)) {
      case OrtHardwareDeviceType_CPU:
        device.type = "CPU";
        break;
      case OrtHardwareDeviceType_GPU:
        device.type = "GPU";
        break;
      case OrtHardwareDeviceType_NPU:
        device.type = "NPU";
        break;
      default:
        continue;
    }

    device.vendor_id = Ort::api->HardwareDevice_VendorId(hardware);
    device.device_id = Ort::api->HardwareDevice_DeviceId(hardware);
    device.source = "onnxruntime";
    const auto vendor =
        BoundedTelemetryCString(Ort::api->HardwareDevice_Vendor(hardware));
    device.vendor =
        vendor.empty() ? VendorName(device.vendor_id) : ScrubStringForTelemetry(vendor);
    if (const auto* metadata = Ort::api->HardwareDevice_Metadata(hardware)) {
      const auto name =
          BoundedTelemetryCString(Ort::api->GetKeyValue(metadata, "Description"));
      if (!name.empty()) {
        device.name = ScrubStringForTelemetry(name);
      }
#if defined(__linux__)
      const auto bus_id =
          BoundedTelemetryCString(Ort::api->GetKeyValue(metadata, "pci_bus_id"), 13);
      bool valid = bus_id.size() == 12;
      for (size_t i = 0; valid && i < bus_id.size(); ++i) {
        if (i == 4 || i == 7 || i == 10) {
          valid = bus_id[i] == (i == 10 ? '.' : ':');
        } else {
          valid = std::isxdigit(static_cast<unsigned char>(bus_id[i])) != 0;
        }
      }
      if (valid) {
        const auto path = std::filesystem::path{"/sys/bus/pci/devices"} / bus_id;
        const auto name_storage = ReadText(path / "product_name");
        const auto name_text = TelemetryInternal::TrimAscii(name_storage);
        if (!name_text.empty()) {
          device.name = ScrubStringForTelemetry(name_text);
        }
        const auto driver_storage = ReadText(path / "driver/module/version");
        const auto driver = TelemetryInternal::TrimAscii(driver_storage);
        if (!driver.empty()) {
          device.driver_version = ScrubStringForTelemetry(driver);
        }
        const auto total = TelemetryInternal::ParseHardwareInteger(
            ReadText(path / "mem_info_vram_total"));
        const auto used = TelemetryInternal::ParseHardwareInteger(
            ReadText(path / "mem_info_vram_used"));
        if (total >= 0) {
          device.dedicated_memory_mb = ToMB(static_cast<uint64_t>(total));
          if (used >= 0 && used <= total) {
            device.free_memory_mb =
                ToMB(static_cast<uint64_t>(total - used));
          }
        }
      }
#endif
    }
    ort_devices.push_back(std::move(device));
  }
  const std::span<const HardwareDeviceInfo> matchable_ort_devices =
      info.inventory_truncated || count > kMaxHardwareInventoryEntries
          ? std::span<const HardwareDeviceInfo>{}
          : std::span<const HardwareDeviceInfo>{ort_devices};
  for (const auto& device : ort_devices) {
    if (auto* existing = TelemetryInternal::FindOrtHardwareDeviceMatch(
            info.devices, device, matchable_ort_devices)) {
      existing->type = device.type;
      if (existing->vendor_id < 0) {
        existing->vendor_id = device.vendor_id;
      }
      if (existing->device_id < 0) {
        existing->device_id = device.device_id;
      }
    } else {
      AddDevice(info, device);
    }
  }
  info.inventory_truncated |= count > kMaxHardwareInventoryEntries;
}

}  // namespace

namespace TelemetryInternal {

HardwareDeviceInfo* FindOrtHardwareDeviceMatch(
    std::span<HardwareDeviceInfo> candidates,
    const HardwareDeviceInfo& ort_device,
    std::span<const HardwareDeviceInfo> ort_devices) {
  const auto compatible = [&](const auto& candidate) {
    return (candidate.type == ort_device.type || candidate.type == "Unknown") &&
           (ort_device.type == "CPU" ||
            (candidate.source != "onnxruntime" &&
             candidate.vendor_id == ort_device.vendor_id));
  };
  const auto exact = std::find_if(candidates.begin(), candidates.end(), [&](const auto& candidate) {
    return compatible(candidate) &&
           (ort_device.type == "CPU" ||
            (candidate.device_id >= 0 && candidate.device_id == ort_device.device_id));
  });
  if (exact != candidates.end()) {
    return &*exact;
  }

  // Without a product ID or a physical-device key, only a one-to-one match is safe.
  if (std::count_if(candidates.begin(), candidates.end(), compatible) != 1 ||
      std::count_if(ort_devices.begin(), ort_devices.end(), [&](const auto& device) {
        return device.type == ort_device.type && device.vendor_id == ort_device.vendor_id;
      }) != 1) {
    return nullptr;
  }
  const auto unknown = std::find_if(candidates.begin(), candidates.end(), [&](const auto& candidate) {
    return compatible(candidate) && candidate.device_id < 0;
  });
  return unknown == candidates.end() ? nullptr : &*unknown;
}

bool AccumulateProcessMemoryBytes(uint64_t& total, uint64_t used) {
  if (used == (std::numeric_limits<uint64_t>::max)() ||
      used > (std::numeric_limits<uint64_t>::max)() - total) {
    return false;
  }
  total += used;
  return true;
}

int64_t ParseHardwareInteger(std::string_view value, int base) {
  value = TrimAscii(value);
  if (value.empty() || base < 2 || base > 36) {
    return -1;
  }

  int64_t result = -1;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), result, base);
  return error == std::errc{} && end == value.data() + value.size() && result >= 0
             ? result
             : -1;
}

HardwareDeviceInfo ParseCpuInfo(std::string_view cpu_info) {
  HardwareDeviceInfo cpu;
  cpu.type = "CPU";
  cpu_info = cpu_info.substr(0, kMaxProbeBytes);
  while (!cpu_info.empty()) {
    const auto end = cpu_info.find('\n');
    const auto line = cpu_info.substr(0, end);
    const auto colon = line.find(':');
    if (colon != std::string_view::npos) {
      const auto key = TrimAscii(line.substr(0, colon));
      const auto value = TrimAscii(line.substr(colon + 1));
      if (key == "model name" || (key == "Processor" && cpu.name == "unknown")) {
        cpu.name = ScrubStringForTelemetry(value);
      } else if (key == "vendor_id") {
        cpu.vendor = ScrubStringForTelemetry(value);
      }
    }
    if (end == std::string_view::npos || (line.empty() && cpu.name != "unknown")) {
      break;
    }
    cpu_info.remove_prefix(end + 1);
  }
  return cpu;
}

}  // namespace TelemetryInternal

HardwareInfo CollectHardwareInfo() noexcept {
  HardwareInfo info;
  try {
    auto cpu = ReadCpuInfo();
    const auto logical_cores = std::thread::hardware_concurrency();
    if (logical_cores != 0) {
      cpu.logical_core_count = logical_cores;
    }
    AddDevice(info, std::move(cpu));

#if defined(_WIN32)
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    if (::GlobalMemoryStatusEx(&memory)) {
      info.available_system_memory_mb = ToMB(memory.ullAvailPhys);
    }
    ReadWindowsAccelerators(info);
#elif defined(__linux__)
    const auto meminfo = ReadText("/proc/meminfo");
    const auto start = meminfo.find("MemAvailable:");
    if (start != std::string::npos) {
      const auto line = std::string_view{meminfo}.substr(start + 13);
      const auto kb = TelemetryInternal::ParseHardwareInteger(
          line.substr(0, line.find("kB")));
      if (kb >= 0) {
        info.available_system_memory_mb = kb / 1024;
      }
    }
    NvmlLibrary{}.AddHardware(info);
#endif

    ReadOrtDevices(info);
  } catch (...) {
    info.inventory_truncated = true;
    Diagnostic("Telemetry hardware inventory failed; partial data will be reported");
  }
  return info;
}

int64_t ReadProcessCpuTimeMs() noexcept {
#if defined(_WIN32)
  FILETIME creation{}, exit{}, kernel{}, user{};
  if (!::GetProcessTimes(::GetCurrentProcess(), &creation, &exit, &kernel, &user)) {
    return -1;
  }
  const uint64_t kernel_ticks =
      (static_cast<uint64_t>(kernel.dwHighDateTime) << 32) | kernel.dwLowDateTime;
  const uint64_t user_ticks =
      (static_cast<uint64_t>(user.dwHighDateTime) << 32) | user.dwLowDateTime;
  return static_cast<int64_t>((kernel_ticks + user_ticks) / 10000);
#else
  rusage usage{};
  if (::getrusage(RUSAGE_SELF, &usage) != 0) {
    return -1;
  }
  return (static_cast<int64_t>(usage.ru_utime.tv_sec) + usage.ru_stime.tv_sec) *
             1000 +
         (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1000;
#endif
}

ProcessResourceUsage CollectProcessResourceUsage() noexcept {
  ProcessResourceUsage usage;
  usage.cpu_time_ms = ReadProcessCpuTimeMs();
  try {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS memory{};
    if (::K32GetProcessMemoryInfo(::GetCurrentProcess(), &memory, sizeof(memory))) {
      usage.resident_memory_mb = ToMB(memory.WorkingSetSize);
    }
    auto inventory = ReadDxCoreInventory();
    if (inventory.available && inventory.complete && !inventory.truncated) {
      uint64_t gpu_bytes = 0;
      uint64_t npu_bytes = 0;
      bool gpu_complete = true;
      bool npu_complete = true;
      for (const auto& [adapter, type] : inventory.adapters) {
        if (type == "Unknown") {
          gpu_complete = false;
          npu_complete = false;
          continue;
        }
        const bool is_npu = type == "NPU";
        const auto adapter_memory = ReadAdapterMemory(*adapter.Get());
        (is_npu ? npu_complete : gpu_complete) &= adapter_memory.available;
        (is_npu ? npu_bytes : gpu_bytes) +=
            adapter_memory.local_used + adapter_memory.nonlocal_used;
      }
      usage.gpu_memory_mb = gpu_complete ? ToMB(gpu_bytes) : -1;
      usage.npu_memory_mb = npu_complete ? ToMB(npu_bytes) : -1;
      usage.accelerator_memory_source = "dxcore_process_local_and_nonlocal";
    }
#elif defined(__linux__)
    const auto statm = ReadText("/proc/self/statm");
    const auto first_space = statm.find(' ');
    if (first_space != std::string::npos) {
      const auto resident =
          TelemetryInternal::TrimAscii(std::string_view{statm}.substr(first_space + 1));
      const auto pages = TelemetryInternal::ParseHardwareInteger(
          resident.substr(0, resident.find(' ')));
      const auto page_size = ::sysconf(_SC_PAGESIZE);
      if (pages >= 0 && page_size > 0 &&
          pages <= (std::numeric_limits<int64_t>::max)() / page_size) {
        usage.resident_memory_mb =
            ToMB(static_cast<uint64_t>(pages * page_size));
      }
    }
    struct stat pid_namespace{};
    const auto& device_info = GetDeviceInfo();
    if (::stat("/proc/self/ns/pid", &pid_namespace) != 0 ||
        !TelemetryInternal::ShouldQueryNvmlProcessMemory(
            device_info.is_container, device_info.virtualization_type == "wsl",
            static_cast<uint64_t>(pid_namespace.st_ino))) {
      Diagnostic("Telemetry NVML process memory is unknown without verified host PIDs");
    } else {
      usage.gpu_memory_mb = NvmlLibrary{}.ProcessMemoryMB();
    }
    if (usage.gpu_memory_mb >= 0) {
      usage.accelerator_memory_source = "nvml_process_nvidia_only";
    }
#elif defined(__APPLE__)
    mach_task_basic_info_data_t memory{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (::task_info(::mach_task_self(), MACH_TASK_BASIC_INFO,
                    reinterpret_cast<task_info_t>(&memory), &count) == KERN_SUCCESS) {
      usage.resident_memory_mb = ToMB(memory.resident_size);
    }
#endif
  } catch (...) {
    Diagnostic("Telemetry process resource query failed; reporting unknown");
  }
  if (usage.resident_memory_mb < 0 || usage.cpu_time_ms < 0) {
    Diagnostic("Telemetry process RAM or CPU query unavailable; reporting unknown");
  }
  return usage;
}

}  // namespace Generators

#else

namespace Generators {

HardwareInfo CollectHardwareInfo() noexcept {
  return {};
}

ProcessResourceUsage CollectProcessResourceUsage() noexcept {
  return {};
}

int64_t ReadProcessCpuTimeMs() noexcept {
  return -1;
}

namespace TelemetryInternal {

int64_t ParseHardwareInteger(std::string_view, int) {
  return -1;
}

HardwareDeviceInfo ParseCpuInfo(std::string_view) {
  return {};
}

}  // namespace TelemetryInternal
}  // namespace Generators

#endif
