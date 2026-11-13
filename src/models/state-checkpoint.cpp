// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "state-checkpoint.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace Generators {
namespace {

size_t Multiply(size_t left, size_t right) {
  if (right != 0 && left > std::numeric_limits<size_t>::max() / right)
    throw std::overflow_error("State checkpoint size overflow.");
  return left * right;
}

std::unique_ptr<OrtValue> ByteView(DeviceInterface* device, void* address, size_t bytes) {
  if (bytes > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
    throw std::overflow_error("State checkpoint buffer is too large.");
  return OrtValue::CreateTensor(
      device->GetAllocator().GetInfo(), address, bytes,
      std::array<int64_t, 1>{static_cast<int64_t>(bytes)}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
}

}  // namespace

std::vector<PrefixSnapshotCache::Entry::Range> StateCheckpointLayout::Ranges(
    std::span<const int64_t> shape, size_t element_bytes, size_t tensor_bytes,
    size_t length, size_t selected_row) const {
  if (kind == Kind::Whole) return {{0, 0, tensor_bytes}};
  if (axis >= shape.size() || element_bytes == 0 || compression == 0)
    throw std::invalid_argument("Invalid state checkpoint layout.");
  size_t outer = 1, row_bytes = element_bytes;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0) throw std::invalid_argument("State checkpoint dimensions must be positive.");
    if (i < axis) outer = Multiply(outer, static_cast<size_t>(shape[i]));
    if (i > axis) row_bytes = Multiply(row_bytes, static_cast<size_t>(shape[i]));
  }
  const size_t capacity = static_cast<size_t>(shape[axis]);
  const size_t stride = Multiply(capacity, row_bytes);
  if (Multiply(outer, stride) != tensor_bytes)
    throw std::invalid_argument("State checkpoint shape does not match its buffer.");
  size_t rows = 1;
  if (kind == Kind::Sequence) {
    rows = length / compression;
    if (extra_rows > std::numeric_limits<size_t>::max() - rows)
      throw std::overflow_error("State checkpoint row count overflow.");
    rows += extra_rows;
    if (clamp) rows = std::min(rows, capacity);
    if (rows > capacity) throw std::out_of_range("State checkpoint exceeds sequence capacity.");
  } else if (selected_row >= capacity) {
    throw std::out_of_range("State checkpoint row is outside its buffer.");
  }
  const size_t kept = Multiply(rows, row_bytes);
  std::vector<PrefixSnapshotCache::Entry::Range> ranges;
  ranges.reserve(outer);
  for (size_t i = 0; i < outer; ++i)
    ranges.push_back({Multiply(i, kept),
                      Multiply(i, stride) + (kind == Kind::SelectedRow ? Multiply(selected_row, row_bytes) : 0),
                      kept});
  return ranges;
}

void StateCheckpointDescription::BindBuffer(std::string name, std::function<Buffer()> view) {
  for (const auto& binding : bindings_)
    if (binding.name == name) throw std::invalid_argument("Duplicate checkpoint state binding.");
  bindings_.push_back({std::move(name), std::move(view)});
}

void StateCheckpointDescription::BindTensors(
    std::map<std::string, std::unique_ptr<OrtValue>>& tensors, DeviceInterface* device) {
  groups_.push_back({&tensors, device});
}

bool StateCheckpointDescription::Capture(
    const std::shared_ptr<Snapshot>& snapshot, PrefixSnapshotCache& storage,
    DeviceInterface* device, size_t length, std::string_view component) const {
  if (!snapshot || snapshot->state != PrefixSnapshotCache::State::Empty || !snapshot->buffers.empty())
    throw std::invalid_argument("State checkpoint must be empty before capture.");
  snapshot->owner = owner_;
  snapshot->length = length;
  snapshot->binding_count = bindings_.size();
  snapshot->group_count = groups_.size();
  snapshot->type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  std::vector<Buffer> sources;
  const auto append = [&](const std::string& name, Buffer source, size_t group) {
    if (source.bytes == 0) return;
    if (!source.address || source.first_position > length)
      throw std::invalid_argument("Invalid committed state checkpoint extent.");
    const size_t rows = source.rows.value_or(length - source.first_position);
    auto ranges = source.layout.Ranges(source.shape, Ort::SizeOf(source.type),
                                       source.bytes, rows, source.selected_row);
    if (std::none_of(ranges.begin(), ranges.end(), [](const auto& range) { return range.bytes != 0; })) return;
    snapshot->AddPackedBuffer(ranges);
    snapshot->buffers.push_back(
        {name, source.shape, source.type, source.layout, source.bytes, std::move(ranges),
         source.first_position, rows, group});
    sources.push_back(std::move(source));
  };
  for (const auto& binding : bindings_) append(binding.name, binding.view(), 0);
  for (size_t group = 0; group < groups_.size(); ++group) {
    for (const auto& [name, value] : *groups_[group].tensors) {
      const auto info = value->GetTensorTypeAndShapeInfo();
      const auto type = info->GetElementType();
      append(name, {static_cast<uint8_t*>(value->GetTensorMutableRawData()), Multiply(info->GetElementCount(), Ort::SizeOf(type)), info->GetShape(), type},
             group + 1);
    }
  }
  for (const auto& value : values_) snapshot->values.push_back(value.capture());
  if (!storage.Allocate(snapshot, device, component)) return false;
  try {
    for (size_t i = 0; i < sources.size(); ++i) {
      Tensor source{ByteView(device, sources[i].address, sources[i].bytes)};
      source.p_device_ = device;
      snapshot->Write(i, source, snapshot->buffers[i].ranges);
    }
    snapshot->Flush();
  } catch (...) {
    device->Synchronize();
    storage.Discard(*snapshot);
    throw;
  }
  return true;
}

void StateCheckpointDescription::Restore(const Snapshot& snapshot, DeviceInterface* device) const {
  if (snapshot.owner != owner_ || snapshot.state != PrefixSnapshotCache::State::Resident ||
      snapshot.values.size() != values_.size() || snapshot.binding_count != bindings_.size() ||
      snapshot.group_count != groups_.size())
    throw std::runtime_error("Incompatible state checkpoint.");
  for (size_t i = 0; i < values_.size(); ++i)
    if (!values_[i].compatible(snapshot.values[i]))
      throw std::runtime_error("Incompatible state checkpoint control value.");
  for (const auto& binding : bindings_) {
    if (std::any_of(snapshot.buffers.begin(), snapshot.buffers.end(),
                    [&](const auto& saved) { return saved.group == 0 && saved.name == binding.name; }))
      continue;
    const auto buffer = binding.view();
    if (buffer.bytes == 0) continue;
    if (buffer.first_position > snapshot.length)
      throw std::runtime_error("Incompatible checkpoint state extent.");
    const auto ranges = buffer.layout.Ranges(
        buffer.shape, Ort::SizeOf(buffer.type), buffer.bytes,
        buffer.rows.value_or(snapshot.length - buffer.first_position), buffer.selected_row);
    if (std::any_of(ranges.begin(), ranges.end(), [](const auto& range) { return range.bytes != 0; }))
      throw std::runtime_error("Missing checkpoint state buffer.");
  }

  struct Destination {
    Buffer buffer;
    std::vector<PrefixSnapshotCache::Entry::Range> ranges;
  };
  std::vector<std::map<std::string, std::unique_ptr<OrtValue>>> groups(groups_.size());
  std::vector<Destination> destinations;
  destinations.reserve(snapshot.buffers.size());
  for (const auto& saved : snapshot.buffers) {
    if (saved.group != 0) {
      if (saved.group > groups_.size() || groups_[saved.group - 1].device != device)
        throw std::runtime_error("Incompatible checkpoint tensor group.");
      auto value = OrtValue::CreateTensor(device->GetAllocator(), saved.shape, saved.type);
      const auto info = value->GetTensorTypeAndShapeInfo();
      Buffer buffer{static_cast<uint8_t*>(value->GetTensorMutableRawData()),
                    Multiply(info->GetElementCount(), Ort::SizeOf(saved.type)), saved.shape, saved.type};
      groups[saved.group - 1].emplace(saved.name, std::move(value));
      destinations.push_back({std::move(buffer), saved.ranges});
      continue;
    }
    const auto binding = std::find_if(bindings_.begin(), bindings_.end(),
                                      [&](const Binding& candidate) { return candidate.name == saved.name; });
    if (binding == bindings_.end()) throw std::runtime_error("Missing checkpoint state binding.");
    auto buffer = binding->view();
    if (buffer.shape != saved.shape || buffer.type != saved.type ||
        buffer.bytes != saved.bytes || !(buffer.layout == saved.layout) ||
        !buffer.address || buffer.first_position < saved.first_position)
      throw std::runtime_error("Incompatible checkpoint state buffer.");
    for (const auto& range : saved.ranges)
      if (range.tensor_offset > buffer.bytes || range.bytes > buffer.bytes - range.tensor_offset)
        throw std::runtime_error("Checkpoint state exceeds its destination.");
    auto ranges = saved.ranges;
    if (buffer.first_position != saved.first_position) {
      // A newer auxiliary window needs only the overlapping suffix, rebased to row zero.
      if (buffer.layout.kind != StateCheckpointLayout::Kind::Sequence || buffer.layout.axis != 0 ||
          buffer.layout.compression != 1 || buffer.layout.extra_rows != 0 || buffer.layout.clamp ||
          saved.shape.empty() || saved.shape[0] <= 0 || ranges.size() != 1)
        throw std::runtime_error("Checkpoint extent cannot be rebased.");
      const size_t row_bytes = buffer.bytes / static_cast<size_t>(saved.shape[0]);
      const size_t skip = buffer.first_position - saved.first_position;
      ranges.clear();
      if (skip < saved.rows)
        ranges.push_back({Multiply(skip, row_bytes), 0, Multiply(saved.rows - skip, row_bytes)});
    }
    destinations.push_back({std::move(buffer), std::move(ranges)});
  }
  try {
    for (size_t i = 0; i < destinations.size(); ++i) {
      auto& buffer = destinations[i].buffer;
      Tensor destination{ByteView(device, buffer.address, buffer.bytes)};
      destination.p_device_ = device;
      if (!destinations[i].ranges.empty()) snapshot.Read(i, destination, destinations[i].ranges);
    }
    for (size_t i = 0; i < groups_.size(); ++i) groups_[i].tensors->swap(groups[i]);
    for (size_t i = 0; i < values_.size(); ++i) values_[i].restore(snapshot.values[i]);
  } catch (...) {
    device->Synchronize();
    throw;
  }
}

StateCheckpointCache::StateCheckpointCache(std::shared_ptr<PrefixSnapshotCache> storage, size_t granularity,
                                           std::string component)
    : storage_{std::move(storage)}, granularity_{granularity}, component_{std::move(component)} {
  if (!storage_ || granularity_ == 0) throw std::invalid_argument("Invalid checkpoint cache policy.");
}

std::shared_ptr<StateCheckpointCache::Entry> StateCheckpointCache::Find(std::span<const int32_t> tokens) {
  metrics_.evictions += entries_.Prune();
  ++metrics_.lookups;
  metrics_.queried_tokens += tokens.size() / granularity_ * granularity_;
  const auto* match = entries_.Find(tokens);
  if (match) ++metrics_.matches;
  return match ? match->checkpoint : nullptr;
}

void StateCheckpointCache::Capture(std::span<const int32_t> tokens,
                                   const StateCheckpointDescription& description, DeviceInterface* device) {
  if (tokens.empty() || tokens.size() % granularity_ != 0)
    throw std::runtime_error("Invalid committed checkpoint boundary.");
  device->Synchronize();
  metrics_.evictions += entries_.Prune();
  if (const auto* existing = entries_.Find(tokens); existing && existing->Tokens().size() == tokens.size()) return;
  entries_.TryPublish(metrics_, [&]() -> std::optional<IndexedEntry> {
    auto entry = std::make_shared<Entry>();
    entry->tokens.assign(tokens.begin(), tokens.end());
    if (!description.Capture(entry, *storage_, device, tokens.size(), component_)) {
      ++metrics_.retention_refusals;
      return std::nullopt;
    }
    device->Synchronize();
    metrics_.registered_blocks += tokens.size() / granularity_;
    return IndexedEntry{std::move(entry)};
  });
}

void StateCheckpointCache::Restore(const Entry& entry, const StateCheckpointDescription& description,
                                   DeviceInterface* device) {
  description.Restore(entry, device);
  device->Synchronize();
  storage_->Touch(entry);
  ++metrics_.hits;
  metrics_.matched_tokens += entry.tokens.size();
}

}  // namespace Generators
