// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "device_info.h"

// The platform device-info implementation is only needed when telemetry is
// compiled in. Guarding the whole translation unit keeps default (OFF) builds
// from pulling in platform headers such as <sys/sysinfo.h>, which are not
// portable to every target (e.g. some non-Linux Unix platforms).
#if defined(ORTGENAI_ENABLE_TELEMETRY)

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__ANDROID__)
#include <sys/sysinfo.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <TargetConditionals.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <mach/mach.h>
#include <unistd.h>
#else  // Linux
#include <unistd.h>
#include <sys/sysinfo.h>
#endif

#if !defined(_WIN32)
#include <fcntl.h>
#include <pwd.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace Generators {

namespace {

constexpr char kDeviceIdHashSalt[] = "onnxruntime-genai:";

// FNV-1a 64-bit hash -> fixed-width hex. Stable across platforms and runs
// (unlike std::hash), so the derived device id stays consistent over time.
std::string Fnv1aHex(const std::string& input) {
  uint64_t h = 14695981039346656037ULL;
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
  uint64_t hi{};
  uint64_t lo{};
  try {
    std::random_device rd;
    hi = (static_cast<uint64_t>(rd()) << 32) | rd();
    lo = (static_cast<uint64_t>(rd()) << 32) | rd();
  } catch (...) {
    const auto now = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const auto addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&hi));
    std::mt19937_64 fallback{now ^ addr};
    hi = fallback();
    lo = fallback();
  }
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

std::string GetOsArchitecture() {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
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

int64_t GetTotalMemoryMB() {
#if defined(_WIN32)
  MEMORYSTATUSEX ms{};
  ms.dwLength = sizeof(ms);
  if (GlobalMemoryStatusEx(&ms)) {
    return static_cast<int64_t>(ms.ullTotalPhys / (1024 * 1024));
  }
  return 0;
#elif defined(__APPLE__)
  int64_t mem = 0;
  size_t len = sizeof(mem);
  if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) {
    return mem / (1024 * 1024);
  }
  return 0;
#else
  struct sysinfo si{};
  if (sysinfo(&si) == 0) {
    return static_cast<int64_t>((static_cast<uint64_t>(si.totalram) * si.mem_unit) / (1024 * 1024));
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

// Per-user directory shared by the Microsoft AI dev-tools family (ONNX Runtime, Olive) under
// "Microsoft/DeveloperTools/.onnxruntime". Holds the generated device-id file for desktop POSIX and
// the 1DS offline cache when available (on Windows, the device id itself lives in the registry — see
// below; Android/iOS use the SDK/platform device id).
// Returns an empty path when no per-user location is available, so the caller skips persistence
// rather than writing to a predictable world-writable temp path. Never throws.
std::filesystem::path GetDeviceIdStorageDir() {
#if defined(_WIN32)
  const char* base = std::getenv("LOCALAPPDATA");
  if (!base) return {};
  return std::filesystem::path(base) / "Microsoft" / "DeveloperTools" / ".onnxruntime";
#else
  std::filesystem::path home;
  if (const char* h = std::getenv("HOME"); h != nullptr && h[0] != '\0') {
    home = h;
  } else {
    struct passwd pwd{};
    struct passwd* result = nullptr;
    const long sc = sysconf(_SC_GETPW_R_SIZE_MAX);
    constexpr size_t kDefaultPwBufferSize = 16384;
    constexpr size_t kMaxPwBufferSize = 1 << 20;
    const size_t pw_buffer_size =
        sc > 0 ? std::min(static_cast<size_t>(sc), kMaxPwBufferSize) : kDefaultPwBufferSize;
    std::vector<char> buf(pw_buffer_size);
    if (getpwuid_r(getuid(), &pwd, buf.data(), buf.size(), &result) == 0 &&
        result != nullptr && result->pw_dir != nullptr && result->pw_dir[0] != '\0') {
      home = result->pw_dir;
    }
  }
  if (home.empty()) return {};

#if defined(__APPLE__)
  return home / "Library" / "Application Support" / "Microsoft" / "DeveloperTools" / ".onnxruntime";
#else  // Linux / Android / other POSIX
  // Follow the XDG Base Directory spec, matching ONNX Runtime: prefer
  // $XDG_CACHE_HOME, otherwise ~/.cache.
  if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && xdg[0] != '\0') {
    return std::filesystem::path(xdg) / "Microsoft" / "DeveloperTools" / ".onnxruntime";
  }
  return home / ".cache" / "Microsoft" / "DeveloperTools" / ".onnxruntime";
#endif
#endif
}

// Basic v4-UUID shape check (8-4-4-4-12 hex with hyphens); used to reject a corrupted stored value.
bool IsValidUuid(const std::string& s) {
  if (s.size() != 36) return false;
  for (size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (i == 8 || i == 13 || i == 18 || i == 23) {
      if (c != '-') return false;
    } else if (!std::isxdigit(static_cast<unsigned char>(c))) {
      return false;
    }
  }
  return true;
}

// Provenance of the persistent device id for this run, surfaced as ProcessInfo's deviceIdStatus so we
// can gauge how often the id is freshly created vs reused (and detect corruption). Mirrors ONNX Runtime.
enum class DeviceIdStatus { New,
                            Existing,
                            Corrupted,
                            Failed,
                            Platform };

const char* DeviceIdStatusString(DeviceIdStatus s) {
  switch (s) {
    case DeviceIdStatus::New:
      return "New";
    case DeviceIdStatus::Existing:
      return "Existing";
    case DeviceIdStatus::Corrupted:
      return "Corrupted";
    case DeviceIdStatus::Failed:
      return "Failed";
    case DeviceIdStatus::Platform:
      return "Platform";
  }
  return "Unknown";
}

#if defined(_WIN32)
class ScopedWinHandle {
 public:
  explicit ScopedWinHandle(HANDLE handle = nullptr) : handle_{handle} {}
  ~ScopedWinHandle() {
    if (handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
  }

  ScopedWinHandle(const ScopedWinHandle&) = delete;
  ScopedWinHandle& operator=(const ScopedWinHandle&) = delete;

  HANDLE Get() const { return handle_; }

 private:
  HANDLE handle_{};
};

class ScopedDeviceIdMutex {
 public:
  ScopedDeviceIdMutex() {
    HANDLE token{};
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) return;
    ScopedWinHandle token_handle{token};

    DWORD size = 0;
    GetTokenInformation(token_handle.Get(), TokenUser, nullptr, 0, &size);
    std::vector<unsigned char> token_info(size);
    if (size == 0 ||
        !GetTokenInformation(token_handle.Get(), TokenUser, token_info.data(), size, &size)) {
      return;
    }

    const auto* token_user = reinterpret_cast<const TOKEN_USER*>(token_info.data());
    if (!IsValidSid(token_user->User.Sid)) return;
    const DWORD sid_size = GetLengthSid(token_user->User.Sid);
    uint64_t sid_hash = 14695981039346656037ULL;
    const auto* sid_bytes = static_cast<const unsigned char*>(token_user->User.Sid);
    for (DWORD i = 0; i < sid_size; ++i) {
      sid_hash ^= sid_bytes[i];
      sid_hash *= 1099511628211ULL;
    }

    std::array<wchar_t, 96> mutex_name{};
    _snwprintf_s(mutex_name.data(), mutex_name.size(), _TRUNCATE,
                 L"Global\\Microsoft.DeveloperTools.OnnxRuntime.DeviceId.%016llx",
                 static_cast<unsigned long long>(sid_hash));

    handle_ = CreateMutexW(nullptr, FALSE, mutex_name.data());
    if (handle_ == nullptr) return;

    const DWORD wait_result = WaitForSingleObject(handle_, 1000);
    acquired_ = wait_result == WAIT_OBJECT_0 || wait_result == WAIT_ABANDONED;
  }

  ~ScopedDeviceIdMutex() {
    if (acquired_) ReleaseMutex(handle_);
    if (handle_ != nullptr) CloseHandle(handle_);
  }

  ScopedDeviceIdMutex(const ScopedDeviceIdMutex&) = delete;
  ScopedDeviceIdMutex& operator=(const ScopedDeviceIdMutex&) = delete;

  explicit operator bool() const { return acquired_; }

 private:
  HANDLE handle_{};
  bool acquired_{};
};

// Read (or create) the persistent device id in the Windows registry at
// HKCU\SOFTWARE\Microsoft\DeveloperTools\.onnxruntime : deviceid (REG_SZ), matching Olive and the
// wider Microsoft AI dev-tools family so a machine reports one shared device id. HKCU is per-user;
// the raw UUID is never sent (callers hash it) and can be reset by deleting the value.
std::string GetOrCreateWindowsDeviceId(DeviceIdStatus& status) {
  static constexpr const char* kSubKey = "SOFTWARE\\Microsoft\\DeveloperTools\\.onnxruntime";
  static constexpr const char* kValueName = "deviceid";

  bool corrupted = false;  // a value was present but not a valid UUID
  const auto read_existing = [&]() -> std::string {
    HKEY key{};
    if (RegOpenKeyExA(HKEY_CURRENT_USER, kSubKey, 0, KEY_READ | KEY_WOW64_64KEY, &key) == ERROR_SUCCESS) {
      char buf[256]{};
      DWORD type = 0;
      DWORD size = sizeof(buf);
      const LSTATUS query =
          RegQueryValueExA(key, kValueName, nullptr, &type, reinterpret_cast<LPBYTE>(buf), &size);
      RegCloseKey(key);
      if (query == ERROR_SUCCESS && type == REG_SZ && size > 0) {
        // A REG_SZ value is not guaranteed to include its terminating null within `size`.
        buf[(size < sizeof(buf)) ? size : (sizeof(buf) - 1)] = '\0';
        std::string existing(buf);
        while (!existing.empty() && (existing.back() == '\0' || existing.back() == '\r' ||
                                     existing.back() == '\n' || existing.back() == ' ')) {
          existing.pop_back();
        }
        if (IsValidUuid(existing)) return existing;
        corrupted = true;  // present but invalid -> regenerate
      }
    }
    return {};
  };

  if (std::string existing = read_existing(); !existing.empty()) {
    status = DeviceIdStatus::Existing;
    return existing;
  }

  ScopedDeviceIdMutex mutex;
  if (!mutex) {
    if (std::string existing = read_existing(); !existing.empty()) {
      status = DeviceIdStatus::Existing;
      return existing;
    }
    status = DeviceIdStatus::Failed;
    return GenerateUuidV4();
  }

  // Another process may have published the value while this process waited.
  corrupted = false;
  if (std::string existing = read_existing(); !existing.empty()) {
    status = DeviceIdStatus::Existing;
    return existing;
  }

  std::string uuid = GenerateUuidV4();
  bool wrote = false;
  HKEY write_key{};
  if (RegCreateKeyExA(HKEY_CURRENT_USER, kSubKey, 0, nullptr, REG_OPTION_NON_VOLATILE,
                      KEY_WRITE | KEY_WOW64_64KEY, nullptr, &write_key, nullptr) == ERROR_SUCCESS) {
    wrote = RegSetValueExA(write_key, kValueName, 0, REG_SZ,
                           reinterpret_cast<const BYTE*>(uuid.c_str()),
                           static_cast<DWORD>(uuid.size() + 1)) == ERROR_SUCCESS;
    RegCloseKey(write_key);
  }
  // Corrupted (invalid-and-regenerated) is preserved over New even after a successful rewrite so
  // callers can still observe that the stored id had to be replaced.
  status = !wrote ? DeviceIdStatus::Failed : (corrupted ? DeviceIdStatus::Corrupted : DeviceIdStatus::New);
  return uuid;
}
#endif  // _WIN32

#if !defined(_WIN32)
bool CreateDirectoryTreeOwnerOnly(const std::filesystem::path& dir, bool leaf = true) {
  if (dir.empty()) return false;

  std::error_code ec;
  if (leaf && std::filesystem::is_symlink(dir, ec)) return false;
  ec.clear();
  if (std::filesystem::exists(dir, ec)) {
    if (!std::filesystem::is_directory(dir, ec)) return false;
    if (leaf) {
      ec.clear();
      std::filesystem::permissions(dir, std::filesystem::perms::owner_all,
                                   std::filesystem::perm_options::replace, ec);
      if (ec) return false;
    }
    return true;
  }

  const std::filesystem::path parent = dir.parent_path();
  if (!parent.empty() && parent != dir && !CreateDirectoryTreeOwnerOnly(parent, false)) {
    return false;
  }

  const std::string dir_path = dir.string();
  if (mkdir(dir_path.c_str(), S_IRWXU) != 0 && errno != EEXIST) return false;

  ec.clear();
  if (leaf && std::filesystem::is_symlink(dir, ec)) return false;
  ec.clear();
  if (!std::filesystem::is_directory(dir, ec)) return false;
  if (leaf) {
    std::filesystem::permissions(dir, std::filesystem::perms::owner_all,
                                 std::filesystem::perm_options::replace, ec);
    if (ec) return false;
  }
  return true;
}

enum class DeviceIdPublishResult {
  Created,
  AlreadyExists,
  Failed,
};

DeviceIdPublishResult PublishDeviceIdFileNoFollow(const std::filesystem::path& file,
                                                  const std::string& uuid,
                                                  bool replace_existing) {
  std::filesystem::path temp = file;
  temp += ".tmp." + GenerateUuidV4();

  int flags = O_WRONLY | O_CREAT | O_EXCL;
#ifdef O_NOFOLLOW
  flags |= O_NOFOLLOW;
#endif
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
  const std::string temp_path = temp.string();
  int fd = open(temp_path.c_str(), flags, S_IRUSR | S_IWUSR);
  if (fd < 0) return DeviceIdPublishResult::Failed;

  bool wrote = true;
  const char* data = uuid.data();
  size_t remaining = uuid.size();
  while (remaining > 0) {
    const ssize_t n = write(fd, data, remaining);
    if (n <= 0) {
      wrote = false;
      break;
    }
    data += n;
    remaining -= static_cast<size_t>(n);
  }
  if (close(fd) != 0) wrote = false;

  std::error_code ec;
  if (!wrote) {
    std::filesystem::remove(temp, ec);
    return DeviceIdPublishResult::Failed;
  }

  if (replace_existing) {
    std::filesystem::rename(temp, file, ec);
  } else {
    std::filesystem::create_hard_link(temp, file, ec);
  }
  if (ec) {
    const bool already_exists = !replace_existing && ec == std::errc::file_exists;
    std::filesystem::remove(temp, ec);
    return already_exists ? DeviceIdPublishResult::AlreadyExists : DeviceIdPublishResult::Failed;
  }
  std::filesystem::remove(temp, ec);

  ec.clear();
  std::filesystem::permissions(file,
                               std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
                               std::filesystem::perm_options::replace, ec);
  return DeviceIdPublishResult::Created;
}
#endif  // !_WIN32

// Return a persistent desktop device id: a locally-generated random UUID (NOT a hardware identifier such
// as the machine GUID). On Windows it lives in the registry (see GetOrCreateWindowsDeviceId); on POSIX
// it is a "deviceid" file under the shared Microsoft/DeveloperTools/.onnxruntime directory, matching
// ONNX Runtime and Olive. Because it is a generated UUID it is not a device fingerprint and can be reset
// by clearing the registry value / deleting the file. The raw UUID is never sent — callers hash it.
// `status` records the id's provenance for this run (New/Existing/Corrupted/Failed).
std::string GetOrCreatePersistentDeviceId(DeviceIdStatus& status) {
#if defined(_WIN32)
  return GetOrCreateWindowsDeviceId(status);
#else
  std::error_code ec;
  const std::filesystem::path dir = GetDeviceIdStorageDir();
  if (dir.empty()) {
    // No per-user storage location (e.g. HOME unset). Use an ephemeral UUID instead of persisting to a
    // predictable world-writable temp path, which would be a target for pre-seeding or symlink attacks.
    status = DeviceIdStatus::Failed;
    return GenerateUuidV4();
  }
  if (std::filesystem::is_symlink(dir, ec)) {
    status = DeviceIdStatus::Failed;
    return GenerateUuidV4();
  }
  const std::filesystem::path file = dir / "deviceid";

  ec.clear();
  if (std::filesystem::is_symlink(file, ec)) {
    status = DeviceIdStatus::Failed;
    return GenerateUuidV4();
  }

  bool file_existed = false;
  {
    std::ifstream in(file);
    if (in.is_open()) {
      file_existed = true;
      const auto size = std::filesystem::file_size(file, ec);
      if (!ec && size <= 64) {
        std::string uuid(static_cast<size_t>(size), '\0');
        in.read(uuid.data(), static_cast<std::streamsize>(uuid.size()));
        uuid.erase(std::find_if_not(uuid.rbegin(), uuid.rend(),
                                    [](unsigned char c) { return std::isspace(c); })
                       .base(),
                   uuid.end());
        uuid.erase(uuid.begin(), std::find_if_not(uuid.begin(), uuid.end(),
                                                  [](unsigned char c) { return std::isspace(c); }));
        if (IsValidUuid(uuid)) {
          status = DeviceIdStatus::Existing;
          return uuid;
        }
      }
    }
  }

  std::string uuid = GenerateUuidV4();
  if (!CreateDirectoryTreeOwnerOnly(dir)) {
    status = DeviceIdStatus::Failed;
    return uuid;
  }
  ec.clear();
  if (std::filesystem::is_symlink(file, ec)) {
    status = DeviceIdStatus::Failed;
    return uuid;
  }
  const DeviceIdPublishResult publish_result = PublishDeviceIdFileNoFollow(file, uuid, file_existed);
  if (publish_result == DeviceIdPublishResult::AlreadyExists) {
    // Another process published a complete first-run id first. Read back the winner so all
    // concurrent processes report the same value.
    ec.clear();
    std::ifstream winner;
    if (!std::filesystem::is_symlink(file, ec)) {
      winner.open(file);
    }
    std::string winner_uuid;
    if (winner.is_open() && std::getline(winner, winner_uuid)) {
      winner_uuid.erase(std::find_if_not(winner_uuid.rbegin(), winner_uuid.rend(),
                                         [](unsigned char c) { return std::isspace(c); })
                            .base(),
                        winner_uuid.end());
      winner_uuid.erase(winner_uuid.begin(),
                        std::find_if_not(winner_uuid.begin(), winner_uuid.end(),
                                         [](unsigned char c) { return std::isspace(c); }));
      if (IsValidUuid(winner_uuid)) {
        status = DeviceIdStatus::Existing;
        return winner_uuid;
      }
    }
    status = DeviceIdStatus::Failed;
    return uuid;
  }

  // A file that existed but held an invalid id is reported as Corrupted; a fresh write as New.
  status = publish_result != DeviceIdPublishResult::Created
               ? DeviceIdStatus::Failed
               : (file_existed ? DeviceIdStatus::Corrupted : DeviceIdStatus::New);
  return uuid;
#endif  // _WIN32
}

// Host executable name (basename only, never the full path — the path could embed a user name). Used
// to understand which applications embed onnxruntime-genai. Mirrors ONNX Runtime's processName.
std::string GetProcessName() {
#if defined(_WIN32)
  char path[MAX_PATH]{};
  const DWORD n = GetModuleFileNameA(nullptr, path, static_cast<DWORD>(sizeof(path)));
  if (n == 0) return "unknown";
  const std::string full(path, n);
  const size_t slash = full.find_last_of("\\/");
  return slash == std::string::npos ? full : full.substr(slash + 1);
#elif defined(__APPLE__)
  const char* name = getprogname();
  return name != nullptr ? std::string(name) : "unknown";
#else  // Linux / other POSIX
  // /proc/self/cmdline holds the null-separated argv; argv[0]'s basename is the executable name.
  std::ifstream cmdline("/proc/self/cmdline", std::ios::binary);
  std::string arg0;
  if (cmdline && std::getline(cmdline, arg0, '\0') && !arg0.empty()) {
    const size_t slash = arg0.find_last_of('/');
    return slash == std::string::npos ? arg0 : arg0.substr(slash + 1);
  }
  return "unknown";
#endif
}

bool UsePlatformDeviceId() {
#if defined(__ANDROID__)
  return true;
#elif defined(__APPLE__) && TARGET_OS_IPHONE
  return true;
#else
  return false;
#endif
}

}  // namespace

std::string GetTelemetryStorageDir() {
  const std::filesystem::path dir = GetDeviceIdStorageDir();
  if (dir.empty()) return {};
#if !defined(_WIN32)
  if (!CreateDirectoryTreeOwnerOnly(dir)) return {};
#else
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
#endif
  return dir.string();
}

const DeviceInfo& GetDeviceInfo() {
  static DeviceInfo info = [] {
    DeviceInfo di;
    DeviceIdStatus id_status = DeviceIdStatus::Failed;
    if (UsePlatformDeviceId()) {
      id_status = DeviceIdStatus::Platform;
    } else {
      // "c:" prefix marks a custom device id; the value is a stable hash of a
      // locally-generated UUID plus a product salt (no hardware identifier is ever sent).
      di.device_id = "c:" + Fnv1aHex(std::string{kDeviceIdHashSalt} + GetOrCreatePersistentDeviceId(id_status));
    }
    di.device_id_status = DeviceIdStatusString(id_status);
    di.os_architecture = GetOsArchitecture();
    di.processor_count = GetProcessorCount();
    di.total_memory_mb = GetTotalMemoryMB();
    di.cpu_model = GetCpuModel();
    di.process_name = GetProcessName();
    return di;
  }();
  return info;
}

}  // namespace Generators

#else

namespace Generators {

const DeviceInfo& GetDeviceInfo() {
  static const DeviceInfo info{};
  return info;
}

std::string GetTelemetryStorageDir() {
  return {};
}

}  // namespace Generators

#endif  // ORTGENAI_ENABLE_TELEMETRY
