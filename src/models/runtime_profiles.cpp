// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "runtime_profiles.h"

#include "smartptrs.h"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace Generators {

std::optional<uint64_t> ParseMeminfoBytes(std::string_view meminfo, std::string_view key) {
  while (!meminfo.empty()) {
    const size_t end = meminfo.find('\n');
    const std::string_view line = meminfo.substr(0, end);
    meminfo = end == std::string_view::npos ? std::string_view{} : meminfo.substr(end + 1);
    if (line.size() <= key.size() || line.compare(0, key.size(), key) != 0 || line[key.size()] != ':') {
      continue;
    }
    std::istringstream fields{std::string{line.substr(key.size() + 1)}};
    uint64_t value = 0;
    std::string unit;
    if (!(fields >> value)) return std::nullopt;
    fields >> unit;
    if (unit.empty()) return value;
    if (unit != "kB" || value > std::numeric_limits<uint64_t>::max() / 1024) return std::nullopt;
    return value * 1024;
  }
  return std::nullopt;
}

void ResolveHostMemoryFacts(const HostMemoryReadings& readings, std::optional<uint64_t>& effective_host_bytes,
                            std::optional<uint64_t>& available_host_bytes) noexcept {
  const CgroupMemoryLimits& cgroup = readings.cgroup;
  if (cgroup.status == CgroupMemoryLimits::Status::kUnresolved) {
    effective_host_bytes.reset();
    available_host_bytes.reset();
    return;
  }

  const bool has_cgroup_limit = cgroup.limit && *cgroup.limit > 0;
  effective_host_bytes = readings.physical_total;
  if (has_cgroup_limit && (!effective_host_bytes || *cgroup.limit < *effective_host_bytes)) {
    effective_host_bytes = *cgroup.limit;
  }

  available_host_bytes = readings.physical_available;
  if (cgroup.headroom) {
    available_host_bytes = available_host_bytes ? std::min(*available_host_bytes, *cgroup.headroom) : *cgroup.headroom;
  }
  if (available_host_bytes && effective_host_bytes) {
    available_host_bytes = std::min(*available_host_bytes, *effective_host_bytes);
  }
}

namespace {

constexpr std::string_view kWhitespace = " \t\r\n";

std::string_view TrimWhitespace(std::string_view text) {
  const size_t first = text.find_first_not_of(kWhitespace);
  if (first == std::string_view::npos) return {};
  return text.substr(first, text.find_last_not_of(kWhitespace) - first + 1);
}

// Calls `visit` with each non-empty line of `text` until it returns false. Returns whether every call returned true.
template <typename Visit>
bool ForEachLine(std::string_view text, Visit&& visit) {
  while (!text.empty()) {
    const size_t end = text.find('\n');
    const std::string_view line = text.substr(0, end);
    text = end == std::string_view::npos ? std::string_view{} : text.substr(end + 1);
    if (!line.empty() && !visit(line)) return false;
  }
  return true;
}

// /proc/self/mountinfo escapes the spaces inside a field, so a field never contains one.
std::vector<std::string_view> SplitFields(std::string_view line) {
  std::vector<std::string_view> fields;
  size_t begin = 0;
  while (begin < line.size()) {
    const size_t end = std::min(line.find(' ', begin), line.size());
    if (end > begin) fields.push_back(line.substr(begin, end - begin));
    begin = end + 1;
  }
  return fields;
}

// Decodes the \ooo octal escapes that /proc/self/mountinfo uses for space, tab, newline and backslash.
std::optional<std::string> UnescapeMountinfoField(std::string_view field) {
  std::string decoded;
  for (size_t i = 0; i < field.size(); ++i) {
    if (field[i] != '\\') {
      decoded.push_back(field[i]);
      continue;
    }
    if (i + 3 >= field.size()) return std::nullopt;
    unsigned value = 0;
    for (size_t digit = 1; digit <= 3; ++digit) {
      const char c = field[i + digit];
      if (c < '0' || c > '7') return std::nullopt;
      value = value * 8 + static_cast<unsigned>(c - '0');
    }
    if (value > 0xFF) return std::nullopt;
    decoded.push_back(static_cast<char>(value));
    i += 3;
  }
  return decoded;
}

// Extracts the cgroup v2 path (the "0::<path>" line) from /proc/self/cgroup text into `path`, which stays empty on a
// cgroup v1 system. Returns false when the text has no lines or a line is not "<hierarchy id>:<controllers>:<path>".
bool ParseCgroup2Path(std::string_view text, std::string& path) {
  path.clear();
  size_t line_count = 0;
  const bool well_formed = ForEachLine(text, [&](std::string_view line) {
    ++line_count;
    const size_t first = line.find(':');
    const size_t second = first == std::string_view::npos ? first : line.find(':', first + 1);
    if (second == std::string_view::npos) return false;
    const std::string_view hierarchy = line.substr(0, first);
    if (hierarchy.empty() || hierarchy.find_first_not_of("0123456789") != std::string_view::npos) return false;
    if (hierarchy != "0" || second != first + 1) return true;
    if (second + 1 == line.size()) return false;
    path = std::string{line.substr(second + 1)};
    return true;
  });
  return well_formed && line_count > 0;
}

struct Cgroup2Mount {
  std::string root;  // the cgroup shown at the mount point, as a path from the cgroup hierarchy root
  std::string mount_point;
};

// Collects the cgroup2 mounts from /proc/self/mountinfo text. Returns false when the text has no mounts or a line is
// not "<id> <parent id> <major:minor> <root> <mount point> <options> [optional fields] - <fs type> <source> <options>".
bool ParseCgroup2Mounts(std::string_view mountinfo, std::vector<Cgroup2Mount>& mounts) {
  mounts.clear();
  size_t line_count = 0;
  const bool well_formed = ForEachLine(mountinfo, [&](std::string_view line) {
    ++line_count;
    const std::vector<std::string_view> fields = SplitFields(line);
    size_t separator = 6;
    while (separator < fields.size() && fields[separator] != "-") ++separator;
    if (separator + 1 >= fields.size()) return false;
    if (fields[separator + 1] != "cgroup2") return true;
    auto root = UnescapeMountinfoField(fields[3]);
    auto mount_point = UnescapeMountinfoField(fields[4]);
    if (!root || !mount_point) return false;
    mounts.push_back(Cgroup2Mount{std::move(*root), std::move(*mount_point)});
    return true;
  });
  return well_formed && line_count > 0;
}

// Splits an absolute cgroup path into its components ("/" has none). A path that is not absolute, or that has an empty,
// "." or ".." component (what a process outside the cgroup namespace root sees), does not name a directory.
std::optional<std::vector<std::string>> SplitCgroupPath(std::string_view path) {
  if (path.empty() || path.front() != '/') return std::nullopt;
  std::vector<std::string> components;
  path.remove_prefix(1);
  while (!path.empty()) {
    const size_t end = path.find('/');
    const std::string_view component = path.substr(0, end);
    if (component.empty() || component == "." || component == "..") return std::nullopt;
    components.emplace_back(component);
    path = end == std::string_view::npos ? std::string_view{} : path.substr(end + 1);
  }
  return components;
}

std::string JoinPath(const std::string& directory, std::string_view name) {
  std::string path = directory;
  if (path.empty() || path.back() != '/') path.push_back('/');
  path.append(name);
  return path;
}

// Parses an unsigned decimal, such as the content of memory.current.
std::optional<uint64_t> ParseCgroupBytes(std::string_view text) {
  text = TrimWhitespace(text);
  if (text.empty()) return std::nullopt;
  const char* const end = text.data() + text.size();
  uint64_t value = 0;
  const auto result = std::from_chars(text.data(), end, value);
  if (result.ec != std::errc{} || result.ptr != end) return std::nullopt;
  return value;
}

}  // namespace

CgroupMemoryLimits ResolveProcessCgroupMemory(const HostFileReader& read_file) {
  using Status = CgroupMemoryLimits::Status;
  const auto with_status = [](Status status) {
    CgroupMemoryLimits limits;
    limits.status = status;
    return limits;
  };

  const auto cgroup_text = read_file("/proc/self/cgroup");
  std::string cgroup_path;
  if (!cgroup_text || !ParseCgroup2Path(*cgroup_text, cgroup_path)) return with_status(Status::kUnresolved);
  if (cgroup_path.empty()) return with_status(Status::kNotApplicable);

  const auto mountinfo = read_file("/proc/self/mountinfo");
  std::vector<Cgroup2Mount> mounts;
  if (!mountinfo || !ParseCgroup2Mounts(*mountinfo, mounts)) return with_status(Status::kUnresolved);
  if (mounts.empty()) return with_status(Status::kNotApplicable);

  const auto process_path = SplitCgroupPath(cgroup_path);
  if (!process_path) return with_status(Status::kUnresolved);

  // Of the mounts that show the process's cgroup, take the one rooted highest, which shows the most ancestors.
  const Cgroup2Mount* mount = nullptr;
  size_t mount_depth = 0;
  for (const Cgroup2Mount& candidate : mounts) {
    const auto root = SplitCgroupPath(candidate.root);
    if (!root || root->size() > process_path->size() ||
        !std::equal(root->begin(), root->end(), process_path->begin())) {
      continue;
    }
    if (!mount || root->size() < mount_depth) {
      mount = &candidate;
      mount_depth = root->size();
    }
  }
  if (!mount) return with_status(Status::kUnresolved);

  // The cgroup directories from the mount root down to the process's own cgroup.
  std::vector<std::string> directories{mount->mount_point};
  for (size_t i = mount_depth; i < process_path->size(); ++i) {
    directories.push_back(JoinPath(directories.back(), (*process_path)[i]));
  }
  // Every cgroup directory has cgroup.controllers, so its absence means the path does not lead to the process's cgroup.
  if (!read_file(JoinPath(directories.back(), "cgroup.controllers"))) return with_status(Status::kUnresolved);

  CgroupMemoryLimits limits = with_status(Status::kResolved);
  for (const std::string& directory : directories) {
    // The hierarchy root and the cgroups without the memory controller have no memory.max.
    const auto max_text = read_file(JoinPath(directory, "memory.max"));
    if (!max_text || TrimWhitespace(*max_text) == "max") continue;
    const auto max_bytes = ParseCgroupBytes(*max_text);
    if (!max_bytes) return with_status(Status::kUnresolved);
    if (*max_bytes == 0) continue;  // a zero limit is ignored, as in ResolveHostMemoryFacts

    const auto current_text = read_file(JoinPath(directory, "memory.current"));
    const auto current_bytes = current_text ? ParseCgroupBytes(*current_text) : std::nullopt;
    if (!current_bytes) return with_status(Status::kUnresolved);
    const uint64_t headroom = *max_bytes > *current_bytes ? *max_bytes - *current_bytes : 0;
    limits.limit = limits.limit ? std::min(*limits.limit, *max_bytes) : *max_bytes;
    limits.headroom = limits.headroom ? std::min(*limits.headroom, headroom) : headroom;
  }
  return limits;
}

namespace {

#if defined(__linux__)
std::optional<std::string> ReadHostFile(const std::string& path) {
  std::ifstream file(path);
  if (!file) return std::nullopt;
  std::ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

std::optional<uint64_t> ReadMeminfoBytes(std::string_view key) {
  const auto meminfo = ReadHostFile("/proc/meminfo");
  return meminfo ? ParseMeminfoBytes(*meminfo, key) : std::nullopt;
}
#endif

void QueryHostMemoryFacts(std::optional<uint64_t>& effective_host_bytes,
                          std::optional<uint64_t>& available_host_bytes) noexcept {
  HostMemoryReadings readings;
#if defined(_WIN32)
  MEMORYSTATUSEX stat{};
  stat.dwLength = sizeof(stat);
  if (GlobalMemoryStatusEx(&stat)) {
    readings.physical_total = static_cast<uint64_t>(stat.ullTotalPhys);
    readings.physical_available = static_cast<uint64_t>(stat.ullAvailPhys);
  }
#else
  const long page_sz = sysconf(_SC_PAGESIZE);
  const long phys_pages = sysconf(_SC_PHYS_PAGES);
  if (page_sz > 0 && phys_pages > 0) {
    readings.physical_total = static_cast<uint64_t>(page_sz) * static_cast<uint64_t>(phys_pages);
  }
#if defined(__linux__)
  readings.physical_available = ReadMeminfoBytes("MemAvailable");
  readings.cgroup = ResolveProcessCgroupMemory(ReadHostFile);
#endif
#endif
  ResolveHostMemoryFacts(readings, effective_host_bytes, available_host_bytes);
}

}  // namespace

void ApplyRuntimeProfileForSelectedDevice(Config& config, DeviceInterface& device) {
  if (config.runtime_profiles.empty()) return;
  if (device.GetType() != DeviceType::CUDA) {
    throw std::runtime_error(
        "runtime_profiles require CUDA to be the primary execution provider");
  }
  for (const auto& provider : config.model.decoder.session_options.provider_options) {
    if (provider.name != "cuda") continue;
    if (provider.device_filtering_options && provider.device_filtering_options->hardware_device_id) {
      throw std::runtime_error(
          "runtime_profiles do not support CUDA hardware_device_id filtering; use CUDA_VISIBLE_DEVICES instead");
    }
    for (const auto& [name, value] : provider.options) {
      if (name == "device_id" && value != "0") {
        throw std::runtime_error(
            "runtime_profiles require CUDA device_id 0; use CUDA_VISIBLE_DEVICES instead");
      }
    }
  }
  if (device.GetDeviceId(nullptr) != 0) {
    throw std::runtime_error(
        "runtime_profiles require current CUDA device 0; use CUDA_VISIBLE_DEVICES instead");
  }

  size_t available_device_memory_bytes{};
  size_t total_device_memory_bytes{};
  device.GetAvailableMemory(available_device_memory_bytes, total_device_memory_bytes);
  std::optional<bool> is_integrated;
  if (std::any_of(config.runtime_profiles.begin(), config.runtime_profiles.end(), [](const auto& profile) {
        return profile.eligibility.is_integrated.has_value();
      })) {
    std::string error;
    is_integrated = device.GetIsIntegrated(error);
    if (!is_integrated) {
      std::cerr << "Warning: runtime profile integrated-device detection failed: " << error
                << "; using base configuration\n";
      config.runtime_profiles.clear();
      return;
    }
  }
  std::optional<uint64_t> effective_host_bytes;
  std::optional<uint64_t> available_host_bytes;
  QueryHostMemoryFacts(effective_host_bytes, available_host_bytes);
  ApplyRuntimeProfile(config, RuntimeProfileDeviceFacts{
                                  total_device_memory_bytes,
                                  is_integrated,
                                  available_device_memory_bytes,
                                  effective_host_bytes,
                                  available_host_bytes});
  config.runtime_profiles.clear();
}

}  // namespace Generators