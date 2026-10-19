// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__linux__) && !defined(QWEN38_GRAPH_NO_POSIX)
#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Generators::qwen38_numa {

inline bool Enabled(const char* name, bool default_value) {
  const char* value = std::getenv(name);
  return value ? std::strcmp(value, "0") != 0 : default_value;
}

struct PlacementReport {
  std::vector<int> nodes;
  std::vector<size_t> pages_before;
  std::vector<size_t> pages_after;
  size_t pages{};
  size_t misplaced_before{};
  size_t misplaced_after{};
  size_t move_attempts{};
  size_t move_failures{};
  size_t unknown_pages{};
  int policy_error{};
  int query_error{};
  int move_error{};
  int restore_error{};
  bool policy_applied{};
};

#if defined(__linux__) && !defined(QWEN38_GRAPH_NO_POSIX)
constexpr size_t kMaxNodes = 1024;
constexpr size_t kWordBits = sizeof(unsigned long) * 8;
using NodeMask = std::array<unsigned long, kMaxNodes / kWordBits>;

inline std::vector<int> AllowedNodes(NodeMask& mask) {
  if (syscall(SYS_get_mempolicy, nullptr, mask.data(), kMaxNodes, nullptr, MPOL_F_MEMS_ALLOWED) != 0) return {};
  std::vector<int> nodes;
  for (size_t n = 0; n < kMaxNodes; ++n) {
    if (mask[n / kWordBits] & (1UL << (n % kWordBits))) nodes.push_back(static_cast<int>(n));
  }
  return nodes;
}

// MAP_SHARED file faults use the faulting thread's policy, not a VMA mbind policy.
// The caller's policy must be restored before CUDA can create any helper threads.
class ScopedInterleavePolicy {
 public:
  explicit ScopedInterleavePolicy(const NodeMask& nodes) {
    if (syscall(SYS_get_mempolicy, &saved_mode_, saved_mask_.data(), kMaxNodes, nullptr, 0) != 0 ||
        syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, nodes.data(), kMaxNodes) != 0) {
      error_ = errno;
      return;
    }
    active_ = true;
  }
  ~ScopedInterleavePolicy() { Restore(); }
  ScopedInterleavePolicy(const ScopedInterleavePolicy&) = delete;
  ScopedInterleavePolicy& operator=(const ScopedInterleavePolicy&) = delete;
  int Error() const { return error_; }
  int Restore() noexcept {
    if (!active_) return 0;
    const bool empty = std::all_of(saved_mask_.begin(), saved_mask_.end(), [](unsigned long v) { return v == 0; });
    if (syscall(SYS_set_mempolicy, saved_mode_, empty ? nullptr : saved_mask_.data(), empty ? 0 : kMaxNodes) != 0)
      return errno;
    active_ = false;
    return 0;
  }

 private:
  NodeMask saved_mask_{};
  int saved_mode_{};
  int error_{};
  bool active_{};
};

inline void CountPlacement(int node, const std::vector<int>& nodes, std::vector<size_t>& counts) {
  const auto it = std::find(nodes.begin(), nodes.end(), node);
  if (it != nodes.end()) ++counts[static_cast<size_t>(it - nodes.begin())];
}

// Call only on owned, page-aligned, read-only file ranges BEFORE cudaHostRegister.
// Migration is bounded and never requests MOVE_ALL: another process's shared/pinned
// pages may prevent placement, which is reported rather than treated as success.
inline PlacementReport PrepareFileRange(const void* address, size_t bytes, size_t file_offset,
                                       bool interleave, bool migrate) {
  PlacementReport report;
  NodeMask allowed{};
  report.nodes = AllowedNodes(allowed);
  if (report.nodes.empty()) {
    report.policy_error = errno;
    return report;
  }
  report.pages_before.resize(report.nodes.size());
  report.pages_after.resize(report.nodes.size());
  const long page_size_long = sysconf(_SC_PAGESIZE);
  if (page_size_long <= 0) {
    report.policy_error = EINVAL;
    return report;
  }
  const size_t page_size = static_cast<size_t>(page_size_long);
  if (reinterpret_cast<uintptr_t>(address) % page_size != 0 || file_offset % page_size != 0) {
    report.policy_error = EINVAL;
    return report;
  }
  report.pages = bytes / page_size + (bytes % page_size != 0);
  // Audit-only mode does not change allocation policy or migrate pages.
  auto populate = [&] {
    auto* data = static_cast<const volatile uint8_t*>(address);
    for (size_t page = 0; page < report.pages; ++page) static_cast<void>(data[page * page_size]);
  };
  if (interleave && report.nodes.size() > 1) {
    ScopedInterleavePolicy policy(allowed);
    report.policy_error = policy.Error();
    report.policy_applied = report.policy_error == 0;
    populate();
    report.restore_error = policy.Restore();
  } else {
    populate();
  }

  constexpr size_t kBatchPages = 4096;
  std::array<void*, kBatchPages> pages{};
  std::array<void*, kBatchPages> move_pages{};
  std::array<int, kBatchPages> targets{};
  std::array<int, kBatchPages> status{};
  std::array<int, kBatchPages> move_status{};
  const auto* base = static_cast<const uint8_t*>(address);
  for (size_t first = 0; first < report.pages; first += kBatchPages) {
    const size_t count = std::min(kBatchPages, report.pages - first);
    for (size_t i = 0; i < count; ++i) pages[i] = const_cast<uint8_t*>(base + (first + i) * page_size);
    if (syscall(SYS_move_pages, 0, count, pages.data(), nullptr, status.data(), 0) < 0) {
      report.query_error = errno;
      report.unknown_pages += report.pages - first;
      break;
    }
    size_t to_move = 0;
    for (size_t i = 0; i < count; ++i) {
      const int target = report.nodes[(file_offset / page_size + first + i) % report.nodes.size()];
      CountPlacement(status[i], report.nodes, report.pages_before);
      if (status[i] >= 0 && status[i] != target) {
        ++report.misplaced_before;
        if (interleave && migrate && report.nodes.size() > 1 && report.move_error == 0) {
          move_pages[to_move] = pages[i];
          targets[to_move++] = target;
        }
      }
    }
    if (to_move > 0) {
      report.move_attempts += to_move;
      if (syscall(SYS_move_pages, 0, to_move, move_pages.data(), targets.data(), move_status.data(), MPOL_MF_MOVE) < 0) {
        report.move_error = errno;
        report.move_failures += to_move;
      } else {
        for (size_t i = 0; i < to_move; ++i) {
          if (move_status[i] != targets[i]) ++report.move_failures;
        }
      }
      // Query again: a successful syscall does not imply every page moved.
      if (syscall(SYS_move_pages, 0, count, pages.data(), nullptr, status.data(), 0) < 0) {
        report.query_error = errno;
        report.unknown_pages += report.pages - first;
        break;
      }
    }
    for (size_t i = 0; i < count; ++i) {
      CountPlacement(status[i], report.nodes, report.pages_after);
      if (status[i] < 0) ++report.unknown_pages;
      else if (status[i] != report.nodes[(file_offset / page_size + first + i) % report.nodes.size()])
        ++report.misplaced_after;
    }
  }
  return report;
}
#else
inline PlacementReport PrepareFileRange(const void*, size_t, size_t, bool, bool) {
  PlacementReport report;
  report.policy_error = ENOSYS;
  return report;
}
#endif

}  // namespace Generators::qwen38_numa
