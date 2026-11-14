// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <any>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "../engine/prefix_snapshot_cache.h"
#include "../engine/prefix-checkpoint-index.h"

namespace Generators {

// Logical rows need not be contiguous: KV heads can each have their own sequence axis.
struct StateCheckpointLayout {
  enum class Kind { Whole,
                    Sequence,
                    SelectedRow };
  Kind kind{Kind::Whole};
  size_t axis{};
  size_t compression{1};
  size_t extra_rows{};
  bool clamp{};

  bool operator==(const StateCheckpointLayout& other) const {
    return kind == other.kind && axis == other.axis && compression == other.compression &&
           extra_rows == other.extra_rows && clamp == other.clamp;
  }

  std::vector<PrefixSnapshotCache::Entry::Range> Ranges(
      std::span<const int64_t> shape, size_t element_bytes, size_t tensor_bytes,
      size_t length, size_t selected_row = 0) const;
};

class StateCheckpointDescription {
 public:
  struct Buffer {
    uint8_t* address{};
    size_t bytes{};
    std::vector<int64_t> shape{};
    ONNXTensorElementDataType type{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
    StateCheckpointLayout layout{};
    size_t first_position{};
    // Omitted means that sequence coverage ends at the checkpoint's committed length.
    std::optional<size_t> rows{};
    size_t selected_row{};
  };

  struct Snapshot : PrefixSnapshotCache::Entry {
    struct SavedBuffer {
      std::string name;
      std::vector<int64_t> shape;
      ONNXTensorElementDataType type{};
      StateCheckpointLayout layout;
      size_t bytes{};
      std::vector<Range> ranges;
      size_t first_position{};
      size_t rows{};
      size_t group{};
    };
    const void* owner{};
    size_t length{};
    size_t binding_count{};
    size_t group_count{};
    std::vector<SavedBuffer> buffers;
    std::vector<std::any> values;
  };

  explicit StateCheckpointDescription(const void* owner) : owner_{owner} {}
  void BindBuffer(std::string name, std::function<Buffer()> view);
  void BindTensors(std::map<std::string, std::unique_ptr<OrtValue>>& tensors, DeviceInterface* device);

  template <typename T>
  void BindValue(T& value) {
    BindValue<T>([&value]() { return value; }, [&value](const T& saved) { value = saved; });
  }

  template <typename T>
  void BindValue(std::function<T()> capture, std::function<void(const T&)> restore) {
    values_.push_back({[capture = std::move(capture)]() { return std::any{capture()}; },
                       [](const std::any& saved) { return std::any_cast<T>(&saved) != nullptr; },
                       [restore = std::move(restore)](const std::any& saved) { restore(std::any_cast<const T&>(saved)); }});
  }

  bool Capture(const std::shared_ptr<Snapshot>& snapshot, PrefixSnapshotCache& storage,
               DeviceInterface* device, size_t length, std::string_view component) const;
  void Restore(const Snapshot& snapshot, DeviceInterface* device) const;

 private:
  struct Binding {
    std::string name;
    std::function<Buffer()> view;
  };
  struct Value {
    std::function<std::any()> capture;
    std::function<bool(const std::any&)> compatible;
    std::function<void(const std::any&)> restore;
  };
  struct TensorGroup {
    std::map<std::string, std::unique_ptr<OrtValue>>* tensors;
    DeviceInterface* device;
  };
  const void* owner_;
  std::vector<Binding> bindings_;
  std::vector<Value> values_;
  std::vector<TensorGroup> groups_;
};

class StateCheckpointCache {
 public:
  struct Entry : StateCheckpointDescription::Snapshot {
    std::vector<int32_t> tokens;
    std::span<const int32_t> Tokens() const { return tokens; }
    bool IsResident() const { return state == PrefixSnapshotCache::State::Resident; }
  };
  StateCheckpointCache(std::shared_ptr<PrefixSnapshotCache> storage, size_t granularity,
                       std::string component = "native");
  std::shared_ptr<Entry> Find(std::span<const int32_t> tokens);
  void Capture(std::span<const int32_t> tokens, const StateCheckpointDescription& description,
               DeviceInterface* device);
  void Restore(const Entry& entry, const StateCheckpointDescription& description, DeviceInterface* device);
  const PrefixCacheMetrics& Metrics() const { return metrics_; }

 private:
  struct IndexedEntry {
    std::shared_ptr<Entry> checkpoint;
    std::span<const int32_t> Tokens() const { return checkpoint->Tokens(); }
    bool IsResident() const { return checkpoint->IsResident(); }
  };
  std::shared_ptr<PrefixSnapshotCache> storage_;
  size_t granularity_;
  std::string component_;
  PrefixCheckpointIndex<IndexedEntry> entries_;
  PrefixCacheMetrics metrics_;
};

}  // namespace Generators
