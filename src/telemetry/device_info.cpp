// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "device_info.h"

// The platform device-info implementation is only needed when telemetry is
// compiled in. Guarding the whole translation unit keeps default (OFF) builds
// from pulling in platform headers such as <sys/sysinfo.h>, which are not
// portable to every target (e.g. some non-Linux Unix platforms).
#if defined(ORTGENAI_ENABLE_TELEMETRY)

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <random>
#include <sstream>
#include <string>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <sddl.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#include <mach/mach.h>
#include <unistd.h>
#else  // Linux
#include <unistd.h>
#include <fstream>
#include <sys/sysinfo.h>
#endif

namespace Generators {

namespace {

// FNV-1a 64-bit hash -> fixed-width hex. Stable across platforms and runs
// (unlike std::hash), so the derived device id stays consistent over time.
std::string Fnv1aHex(const std::string& input) {
  uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : input) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
  return std::string(buf);
}

// Generate a random v4 UUID string.
std::string GenerateUuidV4() {
  // Draw the 128 bits directly from std::random_device, a non-deterministic
  // CSPRNG-backed source. Seeding a PRNG (e.g. mt19937) from one or two
  // random_device draws would cap the entropy and risk device-id collisions
  // across a large fleet, so we source each field straight from the device.
  std::random_device rd;
  uint64_t hi = (static_cast<uint64_t>(rd()) << 32) | rd();
  uint64_t lo = (static_cast<uint64_t>(rd()) << 32) | rd();
  hi = (hi & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL;  // version 4
  lo = (lo & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;  // variant 10xx
  char buf[37];
  std::snprintf(buf, sizeof(buf), "%08x-%04x-%04x-%04x-%012llx",
                static_cast<uint32_t>(hi >> 32),
                static_cast<uint32_t>((hi >> 16) & 0xFFFF),
                static_cast<uint32_t>(hi & 0xFFFF),
                static_cast<uint32_t>(lo >> 48),
                static_cast<unsigned long long>(lo & 0xFFFFFFFFFFFFULL));
  return std::string(buf);
}

std::string GetOsName() {
#if defined(_WIN32)
  return "Windows";
#elif defined(__APPLE__)
  return "macOS";
#else
  return "Linux";
#endif
}

std::string GetOsVersion() {
#if defined(_WIN32)
  OSVERSIONINFOW osvi{};
  osvi.dwOSVersionInfoSize = sizeof(osvi);
  // RtlGetVersion always succeeds and provides accurate version info
  using RtlGetVersionFunc = LONG(WINAPI*)(OSVERSIONINFOW*);
  auto ntdll = GetModuleHandleW(L"ntdll.dll");
  if (ntdll) {
    auto rtl_get_version = reinterpret_cast<RtlGetVersionFunc>(GetProcAddress(ntdll, "RtlGetVersion"));
    if (rtl_get_version) {
      rtl_get_version(&osvi);
      return std::to_string(osvi.dwMajorVersion) + "." +
             std::to_string(osvi.dwMinorVersion) + "." +
             std::to_string(osvi.dwBuildNumber);
    }
  }
  return "unknown";
#elif defined(__APPLE__)
  char buf[256]{};
  size_t len = sizeof(buf);
  if (sysctlbyname("kern.osproductversion", buf, &len, nullptr, 0) == 0) {
    return std::string(buf, len > 0 ? len - 1 : 0);
  }
  return "unknown";
#else
  std::ifstream ifs("/etc/os-release");
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.find("VERSION_ID=") == 0) {
      auto val = line.substr(11);
      // Strip quotes
      if (!val.empty() && val.front() == '"') val = val.substr(1);
      if (!val.empty() && val.back() == '"') val.pop_back();
      return val;
    }
  }
  return "unknown";
#endif
}

std::string GetOsArchitecture() {
#if defined(__aarch64__) || defined(_M_ARM64)
  return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
  return "x64";
#elif defined(__i386__) || defined(_M_IX86)
  return "x86";
#else
  return "unknown";
#endif
}

int GetProcessorCount() {
#if defined(_WIN32)
  SYSTEM_INFO si{};
  GetSystemInfo(&si);
  return static_cast<int>(si.dwNumberOfProcessors);
#else
  auto n = sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? static_cast<int>(n) : 1;
#endif
}

int GetTotalMemoryMB() {
#if defined(_WIN32)
  MEMORYSTATUSEX ms{};
  ms.dwLength = sizeof(ms);
  if (GlobalMemoryStatusEx(&ms)) {
    return static_cast<int>(ms.ullTotalPhys / (1024 * 1024));
  }
  return 0;
#elif defined(__APPLE__)
  int64_t mem = 0;
  size_t len = sizeof(mem);
  if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) {
    return static_cast<int>(mem / (1024 * 1024));
  }
  return 0;
#else
  struct sysinfo si{};
  if (sysinfo(&si) == 0) {
    return static_cast<int>((static_cast<uint64_t>(si.totalram) * si.mem_unit) / (1024 * 1024));
  }
  return 0;
#endif
}

std::string GetCpuModel() {
#if defined(_WIN32)
  HKEY key{};
  if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                    "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                    0, KEY_READ, &key) == ERROR_SUCCESS) {
    char buf[256]{};
    DWORD size = sizeof(buf);
    if (RegQueryValueExA(key, "ProcessorNameString", nullptr, nullptr,
                         reinterpret_cast<LPBYTE>(buf), &size) == ERROR_SUCCESS) {
      RegCloseKey(key);
      return std::string(buf);
    }
    RegCloseKey(key);
  }
  return "unknown";
#elif defined(__APPLE__)
  char buf[256]{};
  size_t len = sizeof(buf);
  if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0) {
    return std::string(buf, len > 0 ? len - 1 : 0);
  }
  return "unknown";
#else
  std::ifstream ifs("/proc/cpuinfo");
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.find("model name") == 0) {
      auto pos = line.find(':');
      if (pos != std::string::npos) {
        auto result = line.substr(pos + 1);
        // Trim leading whitespace
        auto start = result.find_first_not_of(" \t");
        return start != std::string::npos ? result.substr(start) : result;
      }
    }
  }
  return "unknown";
#endif
}

// Directory where the locally-generated device id is persisted (mirrors the
// onnxruntime developer-tools location). Returns an empty path when no per-user
// location is available, so the caller skips persistence rather than writing to a
// predictable world-writable temp path. Never throws.
std::filesystem::path GetDeviceIdStorageDir() {
#if defined(_WIN32)
  const char* base = std::getenv("LOCALAPPDATA");
  if (!base) return {};
  return std::filesystem::path(base) / "Microsoft" / "DeveloperTools";
#elif defined(__APPLE__)
  const char* home = std::getenv("HOME");
  if (!home) return {};
  return std::filesystem::path(home) / "Library" / "Application Support" / "Microsoft" / "DeveloperTools";
#else
  const char* home = std::getenv("HOME");
  if (!home) return {};
  return std::filesystem::path(home) / "Microsoft" / "DeveloperTools";
#endif
}

// Return a persistent device id: a locally-generated random UUID (NOT a hardware
// identifier such as the machine GUID), created once and stored in the user's
// profile. Because it is a generated UUID it is not a device fingerprint and can
// be reset by deleting the file. The raw UUID is never sent — callers hash it.
std::string GetOrCreatePersistentDeviceId() {
  std::error_code ec;
  const std::filesystem::path dir = GetDeviceIdStorageDir();
  if (dir.empty()) {
    // No per-user storage location (e.g. HOME/LOCALAPPDATA unset). Use an
    // ephemeral UUID instead of persisting to a predictable world-writable temp
    // path, which would be a target for pre-seeding or symlink attacks.
    return GenerateUuidV4();
  }
  const std::filesystem::path file = dir / ".onnxruntime-genai";

  {
    std::ifstream in(file);
    std::string uuid;
    if (in && std::getline(in, uuid) && !uuid.empty()) {
      return uuid;
    }
  }

  std::string uuid = GenerateUuidV4();
  std::filesystem::create_directories(dir, ec);
  std::ofstream out(file, std::ios::trunc);
  if (out) {
    out << uuid;
  }
  // If persisting failed, the UUID is still used for this process; it just won't
  // be stable across runs.
  return uuid;
}

}  // namespace

const DeviceInfo& GetDeviceInfo() {
  static DeviceInfo info = [] {
    DeviceInfo di;
    // "c:" prefix marks a custom device id; the value is a stable hash of a
    // locally-generated UUID (no hardware identifier is ever sent).
    di.device_id = "c:" + Fnv1aHex(GetOrCreatePersistentDeviceId());
    di.os = GetOsName();
    di.os_version = GetOsVersion();
    di.os_architecture = GetOsArchitecture();
    di.processor_count = GetProcessorCount();
    di.total_memory_mb = GetTotalMemoryMB();
    di.cpu_model = GetCpuModel();
    return di;
  }();
  return info;
}

}  // namespace Generators

#endif  // ORTGENAI_ENABLE_TELEMETRY
