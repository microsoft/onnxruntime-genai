// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "device_info.h"

#include <array>
#include <cstdint>
#include <functional>
#include <mutex>
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
#include <clocale>
#include <ctime>
#else  // Linux
#include <unistd.h>
#include <fstream>
#include <clocale>
#include <ctime>
#include <sys/sysinfo.h>
#endif

namespace Generators {

namespace {

// Simple hash to avoid sending raw machine identifiers
std::string HashString(const std::string& input) {
  auto h = std::hash<std::string>{}(input);
  std::ostringstream oss;
  oss << std::hex << h;
  return oss.str();
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
  struct sysinfo si {};
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

std::string GetMachineId() {
#if defined(_WIN32)
  HKEY key{};
  if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                    "SOFTWARE\\Microsoft\\Cryptography",
                    0, KEY_READ, &key) == ERROR_SUCCESS) {
    char buf[256]{};
    DWORD size = sizeof(buf);
    if (RegQueryValueExA(key, "MachineGuid", nullptr, nullptr,
                         reinterpret_cast<LPBYTE>(buf), &size) == ERROR_SUCCESS) {
      RegCloseKey(key);
      return std::string(buf);
    }
    RegCloseKey(key);
  }
  return "unknown";
#elif defined(__APPLE__)
  // Use IOPlatformSerialNumber via sysctl
  char buf[256]{};
  size_t len = sizeof(buf);
  if (sysctlbyname("kern.uuid", buf, &len, nullptr, 0) == 0) {
    return std::string(buf, len > 0 ? len - 1 : 0);
  }
  return "unknown";
#else
  std::ifstream ifs("/etc/machine-id");
  std::string id;
  if (std::getline(ifs, id) && !id.empty()) {
    return id;
  }
  return "unknown";
#endif
}

std::string GetUserLocale() {
#if defined(_WIN32)
  char buf[LOCALE_NAME_MAX_LENGTH]{};
  if (GetLocaleInfoA(LOCALE_USER_DEFAULT, LOCALE_SNAME, buf, sizeof(buf))) {
    return std::string(buf);
  }
  return "unknown";
#else
  const char* loc = std::setlocale(LC_ALL, nullptr);
  return loc ? std::string(loc) : "unknown";
#endif
}

std::string GetUserTimezone() {
#if defined(_WIN32)
  TIME_ZONE_INFORMATION tzi{};
  if (GetTimeZoneInformation(&tzi) != TIME_ZONE_ID_INVALID) {
    // Convert wide timezone name to narrow string
    char narrow[64]{};
    WideCharToMultiByte(CP_UTF8, 0, tzi.StandardName, -1, narrow, sizeof(narrow), nullptr, nullptr);
    return std::string(narrow);
  }
  return "unknown";
#else
  time_t t = time(nullptr);
  struct tm local {};
  localtime_r(&t, &local);
  return local.tm_zone ? std::string(local.tm_zone) : "unknown";
#endif
}

}  // namespace

const DeviceInfo& GetDeviceInfo() {
  static DeviceInfo info = [] {
    DeviceInfo di;
    di.device_id = HashString(GetMachineId());
    di.os = GetOsName();
    di.os_version = GetOsVersion();
    di.os_architecture = GetOsArchitecture();
    di.processor_count = GetProcessorCount();
    di.total_memory_mb = GetTotalMemoryMB();
    di.cpu_model = GetCpuModel();
    di.user_locale = GetUserLocale();
    di.user_timezone = GetUserTimezone();
    return di;
  }();
  return info;
}

}  // namespace Generators
