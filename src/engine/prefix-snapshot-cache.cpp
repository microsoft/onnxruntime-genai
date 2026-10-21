// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "prefix-snapshot-cache.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <random>

#include "../logging.h"

namespace Generators {
namespace {

constexpr size_t kTransferBytes = 1024 * 1024;

size_t TensorBytes(const std::vector<int64_t>& shape, ONNXTensorElementDataType type) {
  size_t bytes = Ort::SizeOf(type);
  if (bytes == 0) throw std::invalid_argument("Prefix snapshot dtype has no fixed element size.");
  for (int64_t dimension : shape) {
    if (dimension <= 0 || static_cast<uint64_t>(dimension) > std::numeric_limits<size_t>::max() / bytes) {
      throw std::overflow_error("Prefix snapshot tensor size is invalid or overflows size_t.");
    }
    bytes *= static_cast<size_t>(dimension);
  }
  return bytes;
}

void CopyBytes(DeviceInterface* destination_device, void* destination,
               DeviceInterface* source_device, const void* source, size_t bytes) {
  const auto from = source_device->GetType(), to = destination_device->GetType();
  if (from == DeviceType::CUDA && to == DeviceType::CPU) {
    source_device->CopyDeviceToHost(destination, source, bytes);
  } else if (from == DeviceType::CPU && to == DeviceType::CUDA) {
    destination_device->CopyHostToDevice(destination, source, bytes);
  } else {
    auto destination_span = destination_device->WrapMemory(
        std::span<uint8_t>{static_cast<uint8_t*>(destination), bytes});
    auto source_span = source_device->WrapMemory(
        std::span<const uint8_t>{static_cast<const uint8_t*>(source), bytes});
    destination_span.CopyFrom(source_span);
  }
}

size_t LayerOffset(const PrefixSnapshotCache::Entry& entry, size_t layer, size_t offset, size_t bytes) {
  if (layer >= entry.shapes.size()) {
    throw std::out_of_range("Prefix snapshot layer is out of range.");
  }
  const size_t layer_bytes = TensorBytes(entry.shapes[layer], entry.type);
  if (offset > layer_bytes || bytes > layer_bytes - offset) {
    throw std::out_of_range("Prefix snapshot byte range is out of range.");
  }
  for (size_t i = 0; i < layer; ++i) {
    offset += TensorBytes(entry.shapes[i], entry.type);
  }
  return offset;
}

void CheckTensorRange(Tensor& tensor, size_t offset, size_t bytes) {
  const size_t size = tensor.GetByteSpan().size();
  if (offset > size || bytes > size - offset) {
    throw std::out_of_range("Prefix snapshot transfer tensor range is out of range.");
  }
}

}  // namespace

struct PrefixSnapshotCache::Entry::DiskStorage {
  std::filesystem::path directory;
  mutable std::fstream file;

  ~DiskStorage() {
    file.exceptions(std::ios::goodbit);
    file.close();
    if (directory.empty()) return;
    std::error_code error;
    std::filesystem::remove(directory / "payload", error);
    if (!error) std::filesystem::remove(directory, error);
    if (error && g_log.enabled && g_log.warning) {
      try {
        Log("warning", "Unable to remove prefix snapshot spill storage: %s", error.message().c_str());
      } catch (...) {
        // Cleanup diagnostics must not throw from the destructor.
      }
    }
  }
};

PrefixSnapshotCache::Entry::Entry() = default;
PrefixSnapshotCache::Entry::~Entry() { Release(); }

void PrefixSnapshotCache::Entry::Release() noexcept {
  caches.clear();
  host_buffers.clear();
  disk.reset();
  if (accounting) {
    accounting->used[static_cast<size_t>(tier)] -= bytes;
    accounting.reset();
  }
}

void PrefixSnapshotCache::Entry::TakeStorage(Entry& source) noexcept {
  Release();
  tier = source.tier;
  accounting = std::move(source.accounting);
  host_buffers = std::move(source.host_buffers);
  caches = std::move(source.caches);
  disk = std::move(source.disk);
}

void* PrefixSnapshotCache::Entry::TransferBuffer(DeviceInterface* device) const {
  const bool pinned = device->GetType() == DeviceType::CUDA;
  if (!accounting->transfer_buffer || (pinned && !accounting->transfer_buffer_pinned)) {
    auto* cpu = GetDeviceInterface(DeviceType::CPU);
    accounting->transfer_buffer = pinned
                                      ? device->AllocatePinnedHost(kTransferBytes)
                                      : std::shared_ptr<void>(cpu->GetAllocator().Alloc(kTransferBytes),
                                                              [cpu](void* pointer) { cpu->GetAllocator().Free(pointer); });
    accounting->transfer_buffer_pinned = pinned;
  }
  if (!accounting->transfer_buffer) throw std::bad_alloc{};
  return accounting->transfer_buffer.get();
}

void PrefixSnapshotCache::Entry::Write(
    size_t layer, size_t offset, Tensor& source, size_t source_offset, size_t count) {
  if (state == State::Evicted) throw std::logic_error("Cannot write an evicted prefix snapshot.");
  const size_t file_offset = LayerOffset(*this, layer, offset, count);
  CheckTensorRange(source, source_offset, count);
  auto* data = static_cast<const uint8_t*>(source.GetRawData()) + source_offset;
  if (tier != Tier::Disk) {
    if (!caches.at(layer)) throw std::logic_error("Prefix snapshot has no resident tensor.");
    CheckTensorRange(*caches[layer], offset, count);
    CopyBytes(caches.at(layer)->p_device_,
              static_cast<uint8_t*>(caches.at(layer)->GetMutableRawData()) + offset,
              source.p_device_, data, count);
    return;
  }
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  void* buffer = TransferBuffer(source.p_device_);
  disk->file.seekp(static_cast<std::streamoff>(file_offset));
  for (size_t copied = 0; copied < count;) {
    const size_t chunk = std::min(count - copied, kTransferBytes);
    CopyBytes(cpu, buffer, source.p_device_, data + copied, chunk);
    source.p_device_->Synchronize();
    disk->file.write(static_cast<const char*>(buffer), static_cast<std::streamsize>(chunk));
    copied += chunk;
  }
  disk->file.flush();
}

void PrefixSnapshotCache::Entry::Read(
    size_t layer, size_t offset, Tensor& destination, size_t destination_offset, size_t count) const {
  if (state == State::Evicted) throw std::logic_error("Cannot read an evicted prefix snapshot.");
  const size_t file_offset = LayerOffset(*this, layer, offset, count);
  CheckTensorRange(destination, destination_offset, count);
  auto* data = static_cast<uint8_t*>(destination.GetMutableRawData()) + destination_offset;
  if (tier != Tier::Disk) {
    if (!caches.at(layer)) throw std::logic_error("Prefix snapshot has no resident tensor.");
    CheckTensorRange(*caches[layer], offset, count);
    CopyBytes(destination.p_device_, data, caches.at(layer)->p_device_,
              static_cast<const uint8_t*>(caches.at(layer)->GetRawData()) + offset, count);
    return;
  }
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  void* buffer = TransferBuffer(destination.p_device_);
  disk->file.seekg(static_cast<std::streamoff>(file_offset));
  for (size_t copied = 0; copied < count;) {
    const size_t chunk = std::min(count - copied, kTransferBytes);
    disk->file.read(static_cast<char*>(buffer), static_cast<std::streamsize>(chunk));
    CopyBytes(destination.p_device_, data + copied, cpu, buffer, chunk);
    destination.p_device_->Synchronize();
    copied += chunk;
  }
}

PrefixSnapshotCache::PrefixSnapshotCache(
    std::array<size_t, 3> limits, size_t max_entries, std::filesystem::path directory)
    : limits_{limits}, max_entries_{max_entries}, directory_{std::move(directory)} {}

void PrefixSnapshotCache::Prune() {
  entries_.erase(std::remove_if(entries_.begin(), entries_.end(), [](const auto& weak) {
                   auto entry = weak.lock();
                   return !entry || entry->state == State::Evicted;
                 }),
                 entries_.end());
}

size_t PrefixSnapshotCache::Count(std::string_view component) const {
  return static_cast<size_t>(std::count_if(entries_.begin(), entries_.end(), [&](const auto& weak) {
    auto entry = weak.lock();
    return entry && entry->state == State::Resident &&
           (component.empty() || entry->component == component);
  }));
}

bool PrefixSnapshotCache::EvictOldest(std::string_view component) {
  std::weak_ptr<Entry>* selected = nullptr;
  uint64_t oldest = std::numeric_limits<uint64_t>::max();
  for (auto& weak : entries_) {
    if (weak.use_count() != 1) continue;
    auto entry = weak.lock();
    if (entry && (component.empty() || entry->component == component) &&
        entry->recency < oldest) {
      selected = &weak;
      oldest = entry->recency;
    }
  }
  if (!selected) return false;
  auto entry = selected->lock();
  entry->Release();
  entry->state = State::Evicted;
  Prune();
  return true;
}

bool PrefixSnapshotCache::ReuseOldest(Entry& destination, std::string_view component) {
  std::weak_ptr<Entry>* selected = nullptr;
  uint64_t oldest = std::numeric_limits<uint64_t>::max();
  for (auto& weak : entries_) {
    if (weak.use_count() != 1) continue;
    auto entry = weak.lock();
    if (entry && (component.empty() || entry->component == component) &&
        entry->state == State::Resident && entry->recency < oldest) {
      selected = &weak;
      oldest = entry->recency;
    }
  }
  if (!selected) return false;
  auto source = selected->lock();
  if (source->type != destination.type || source->shapes != destination.shapes) return false;
  destination.TakeStorage(*source);
  source->state = State::Evicted;
  Prune();
  return true;
}

bool PrefixSnapshotCache::MakeRoom(Tier tier, size_t bytes, DeviceInterface* device) {
  const size_t index = static_cast<size_t>(tier);
  if (bytes > limits_[index]) return false;
  while (bytes > limits_[index] - accounting_->used[index]) {
    std::weak_ptr<Entry>* selected = nullptr;
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (auto& weak : entries_) {
      if (weak.use_count() != 1) continue;
      auto entry = weak.lock();
      if (entry && entry->tier == tier && entry->state == State::Resident && entry->recency < oldest) {
        selected = &weak;
        oldest = entry->recency;
      }
    }
    if (!selected) return false;
    auto victim = selected->lock();
    bool moved = false;
    for (size_t lower = index + 1; lower < limits_.size(); ++lower) {
      if (MakeRoom(static_cast<Tier>(lower), victim->bytes, device)) {
        Move(*victim, static_cast<Tier>(lower), device);
        moved = true;
        break;
      }
    }
    if (!moved) {
      victim->Release();
      victim->state = State::Evicted;
    }
  }
  return true;
}

void PrefixSnapshotCache::AllocateStorage(Entry& entry, Tier tier, DeviceInterface* device) {
  entry.tier = tier;
  if (tier == Tier::Disk) {
    if (entry.bytes > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
      throw std::overflow_error("Prefix snapshot exceeds the disk stream offset range.");
    }
    auto disk = std::make_unique<Entry::DiskStorage>();
    const auto parent = directory_.empty() ? std::filesystem::temp_directory_path() : directory_;
    std::random_device random;
    for (size_t attempt = 0; attempt < 32; ++attempt) {
      const auto candidate = parent / ("oga-prefix-" + std::to_string(random()) + "-" + std::to_string(random()));
      if (!std::filesystem::create_directory(candidate)) continue;
      disk->directory = candidate;
      break;
    }
    if (disk->directory.empty()) throw std::runtime_error("Unable to create private prefix snapshot spill storage.");
    std::filesystem::permissions(disk->directory, std::filesystem::perms::owner_all,
                                 std::filesystem::perm_options::replace);
    disk->file.exceptions(std::ios::failbit | std::ios::badbit);
    disk->file.open(disk->directory / "payload", std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
    disk->file.seekp(static_cast<std::streamoff>(entry.bytes - 1));
    disk->file.put('\0');
    disk->file.flush();
    entry.disk = std::move(disk);
  } else {
    auto* storage_device = tier == Tier::Vram ? device : GetDeviceInterface(DeviceType::CPU);
    for (const auto& shape : entry.shapes) {
      auto tensor = std::make_unique<Tensor>(storage_device, entry.type);
      if (tier == Tier::Ram && device->GetType() == DeviceType::CUDA) {
        const size_t bytes = TensorBytes(shape, entry.type);
        auto buffer = device->AllocatePinnedHost(bytes);
        auto value = OrtValue::CreateTensor(storage_device->GetAllocator().GetInfo(), buffer.get(),
                                            bytes, shape, entry.type);
        entry.host_buffers.push_back(std::move(buffer));
        tensor = std::make_unique<Tensor>(std::move(value));
      } else {
        try {
          tensor->CreateTensor(shape);
        } catch (const Ort::Exception& error) {
          // ORT reports allocator failures through its status API, not std::bad_alloc.
          if (error.GetOrtErrorCode() != ORT_FAIL &&
              error.GetOrtErrorCode() != ORT_RUNTIME_EXCEPTION) {
            throw;
          }
          throw std::bad_alloc{};
        }
      }
      entry.caches.push_back(std::move(tensor));
    }
  }
  accounting_->used[static_cast<size_t>(tier)] += entry.bytes;
  entry.accounting = accounting_;
}

void PrefixSnapshotCache::Move(Entry& entry, Tier tier, DeviceInterface* device) {
  Entry staged;
  staged.shapes = entry.shapes;
  staged.type = entry.type;
  staged.bytes = entry.bytes;
  AllocateStorage(staged, tier, device);
  // A move only proceeds toward a slower tier, so disk is always the destination, never the source.
  try {
    for (size_t layer = 0; layer < entry.caches.size(); ++layer) {
      staged.Write(layer, 0, *entry.caches[layer], 0, TensorBytes(entry.shapes[layer], entry.type));
    }
    device->Synchronize();
  } catch (...) {
    device->Synchronize();
    throw;
  }
  entry.TakeStorage(staged);
}

bool PrefixSnapshotCache::Allocate(const std::shared_ptr<Entry>& entry, DeviceInterface* device,
                                   std::string_view component,
                                   size_t component_max_entries) {
  if (entry->state != State::Empty || entry->accounting) {
    throw std::logic_error("Prefix snapshot storage may only be allocated once.");
  }
  if (device->GetType() != DeviceType::CPU && device->GetType() != DeviceType::CUDA) {
    throw std::runtime_error("Tiered prefix snapshots currently support CPU and CUDA devices.");
  }
  if (limits_[0] != 0 && device->GetType() != DeviceType::CUDA) {
    throw std::runtime_error("A VRAM prefix snapshot budget requires a CUDA device.");
  }
  entry->bytes = 0;
  for (const auto& shape : entry->shapes) {
    const size_t bytes = TensorBytes(shape, entry->type);
    if (bytes > std::numeric_limits<size_t>::max() - entry->bytes) {
      throw std::overflow_error("Prefix snapshot payload size overflows size_t.");
    }
    entry->bytes += bytes;
  }
  if (entry->bytes == 0 || max_entries_ == 0 || component_max_entries == 0 ||
      std::none_of(limits_.begin(), limits_.end(), [&](size_t limit) { return limit >= entry->bytes; })) return false;
  Prune();
  entries_.reserve(entries_.size() + 1);
  const auto publish = [&] {
    entry->component = component;
    entry->state = State::Resident;
    entry->recency = ++clock_;
    entries_.push_back(entry);
    return true;
  };
  while (!component.empty() && Count(component) >= component_max_entries) {
    if (ReuseOldest(*entry, component)) {
      return publish();
    }
    if (!EvictOldest(component)) return false;
  }
  while (entries_.size() >= max_entries_) {
    if (ReuseOldest(*entry)) {
      return publish();
    }
    if (!EvictOldest()) return false;
  }
  for (size_t index = 0; index < limits_.size(); ++index) {
    if (!MakeRoom(static_cast<Tier>(index), entry->bytes, device)) continue;
    Entry staged;
    staged.shapes = entry->shapes;
    staged.type = entry->type;
    staged.bytes = entry->bytes;
    AllocateStorage(staged, static_cast<Tier>(index), device);
    entry->TakeStorage(staged);
    return publish();
  }
  return false;
}

void PrefixSnapshotCache::Touch(const Entry& entry) noexcept {
  for (auto& weak : entries_) {
    auto candidate = weak.lock();
    if (candidate.get() == &entry) {
      candidate->recency = ++clock_;
      return;
    }
  }
}

}  // namespace Generators
