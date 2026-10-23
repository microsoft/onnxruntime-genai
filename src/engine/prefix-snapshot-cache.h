// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "../models/model.h"

namespace Generators {

// Storage policy is independent of prefix identity and the contents of the snapshot.
class PrefixSnapshotCache {
 public:
  enum class Tier : size_t { Vram,
                             Ram,
                             Disk };
  enum class State { Empty,
                     Resident,
                     Evicted };

  struct Accounting {
    std::array<size_t, 3> used{};
    std::shared_ptr<void> transfer_buffer;
    bool transfer_buffer_pinned{};
  };

  struct Entry {
    Entry();
    Entry(const Entry&) = delete;
    Entry& operator=(const Entry&) = delete;
    virtual ~Entry();

    struct Range {
      size_t offset;
      size_t tensor_offset;
      size_t bytes;
    };

    void Write(size_t layer, size_t offset, Tensor& source, size_t source_offset, size_t bytes);
    void Read(size_t layer, size_t offset, Tensor& destination, size_t destination_offset, size_t bytes) const;
    void Write(size_t layer, Tensor& source, std::span<const Range> ranges);
    void Read(size_t layer, Tensor& destination, std::span<const Range> ranges) const;
    void Flush();

    State state{State::Empty};
    Tier tier{Tier::Ram};
    size_t bytes{};
    uint64_t recency{};
    std::vector<std::vector<int64_t>> shapes;
    ONNXTensorElementDataType type{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
    std::vector<std::shared_ptr<void>> host_buffers;
    std::vector<std::unique_ptr<Tensor>> caches;

   private:
    friend class PrefixSnapshotCache;
    struct DiskStorage;
    std::shared_ptr<Accounting> accounting;
    std::unique_ptr<DiskStorage> disk;
    std::string component;
    std::vector<size_t> layer_offsets;
    void Release() noexcept;
    void TakeStorage(Entry& source) noexcept;
    void* TransferBuffer(DeviceInterface* device) const;
  };

  PrefixSnapshotCache(std::array<size_t, 3> limits, size_t max_entries,
                      std::filesystem::path directory = {}, bool device_resident = false);
  bool Allocate(const std::shared_ptr<Entry>& entry, DeviceInterface* device,
                std::string_view component = {},
                size_t component_max_entries = std::numeric_limits<size_t>::max());
  void Touch(const Entry& entry) noexcept;
  void Discard(const Entry& entry) noexcept;
  const std::array<size_t, 3>& Used() const noexcept { return accounting_->used; }

 private:
  void LogEvent(const char* action, const Entry* entry = nullptr,
                const char* reason = "none", int64_t elapsed_us = 0) const noexcept;
  bool MakeRoom(Tier tier, size_t bytes, DeviceInterface* device);
  void Move(Entry& entry, Tier tier, DeviceInterface* device);
  void AllocateStorage(Entry& entry, Tier tier, DeviceInterface* device);
  void Prune();
  size_t Count(std::string_view component) const;
  bool EvictOldest(std::string_view component = {});
  bool ReuseOldest(Entry& destination, std::string_view component = {});

  std::array<size_t, 3> limits_;
  size_t max_entries_;
  std::filesystem::path directory_;
  bool device_resident_;
  std::shared_ptr<Accounting> accounting_{std::make_shared<Accounting>()};
  // The prefix index is the only non-lease owner. Extra strong references protect adoption.
  std::vector<std::weak_ptr<Entry>> entries_;
  uint64_t clock_{};
};

}  // namespace Generators
