// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>

#include "smartptrs.h"

namespace Generators {

// The host mirrors a drafter fills before it uploads one of its step tensors (row maps, block table, ...).
//
// A mirror may be rewritten only once the upload that reads it has run. A small upload carries its bytes in the launch; a
// larger one is read from the mirror when the device reaches it, which can be well after the host queued it. A drafter whose
// steps end in a synchronizing read-back drains its uploads there and keeps one mirror per tensor. One that proposes for a
// request after the other without waiting for the device (the multi-slot engine of qwen38_flash_graph drafts slot after slot)
// must not: the second proposal would rewrite the mirror the first is still being uploaded from and hand it another request's
// block table. It takes a fresh mirror per upload instead, which a device that recycles mirrors only after their upload has
// run (DeviceInterface::RecyclesHostMirrorsAfterUpload) hands out again once it is safe, without waiting for the device.
class Dflash2HostStaging {
 public:
  // Whether every upload takes a fresh mirror, on a device that recycles them safely; the mirror of a tensor is kept and
  // reused otherwise.
  void SetFreshMirrorPerUpload(bool fresh) noexcept { fresh_ = fresh; }
  bool FreshMirrorPerUpload() const noexcept { return fresh_; }

  // The mirror to fill for the next upload of `bytes` bytes to the front of `device_memory`, which `owner` holds in
  // `capacity` bytes. A fresh mirror lives as long as the returned span (and the copy from it); a kept one is the same for
  // every call with this `owner`.
  DeviceSpan<uint8_t> Acquire(DeviceInterface& device, const void* owner, void* device_memory, size_t capacity,
                              size_t bytes) {
    auto* memory = static_cast<uint8_t*>(device_memory);
    if (fresh_ && device.RecyclesHostMirrorsAfterUpload(bytes)) {
      kept_.erase(owner);
      return device.WrapMemory(std::span<uint8_t>{memory, bytes});
    }
    auto [it, inserted] = kept_.try_emplace(owner);
    if (inserted) it->second = device.WrapMemory(std::span<uint8_t>{memory, capacity});
    return it->second.subspan(0, bytes);
  }

  // `owner` is about to give its device memory up: the mirror kept for it goes with it.
  void Forget(const void* owner) noexcept { kept_.erase(owner); }

  size_t KeptMirrors() const noexcept { return kept_.size(); }

 private:
  bool fresh_{false};
  std::unordered_map<const void*, DeviceSpan<uint8_t>> kept_;
};

}  // namespace Generators
