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

using TransferRange = PrefixSnapshotCache::Entry::Range;

void CopyTensorRanges(Tensor& destination, Tensor& source,
                      std::span<const TransferRange> ranges, bool writing) {
  if (ranges.empty()) return;
  if (destination.p_device_->GetType() == DeviceType::WEBGPU ||
      source.p_device_->GetType() == DeviceType::WEBGPU) {
    auto source_span = source.GetByteSpan();
    auto destination_span = destination.GetByteSpan();
    size_t covered = 0;
    bool complete = true;
    for (const auto& range : ranges) {
      if ((writing ? range.offset : range.tensor_offset) != covered) complete = false;
      covered += range.bytes;
    }
    const auto source_cpu = source_span.CopyDeviceToCpu();
    auto destination_cpu = complete && covered == destination_span.size()
                               ? destination_span.CpuSpan()
                               : destination_span.CopyDeviceToCpu();
    for (const auto& range : ranges) {
      const size_t from = writing ? range.tensor_offset : range.offset;
      const size_t to = writing ? range.offset : range.tensor_offset;
      std::copy_n(source_cpu.data() + from, range.bytes, destination_cpu.data() + to);
    }
    destination_span.CopyCpuToDevice();
  } else {
    auto* destination_data = static_cast<uint8_t*>(destination.GetMutableRawData());
    const auto* source_data = static_cast<const uint8_t*>(source.GetRawData());
    for (const auto& range : ranges) {
      CopyBytes(destination.p_device_, destination_data + (writing ? range.offset : range.tensor_offset),
                source.p_device_, source_data + (writing ? range.tensor_offset : range.offset), range.bytes);
    }
  }
}

void CheckRanges(const std::vector<size_t>& offsets, size_t layer, Tensor& tensor,
                 std::span<const TransferRange> ranges) {
  if (offsets.empty() || layer >= offsets.size() - 1) {
    throw std::out_of_range("Prefix snapshot layer is out of range.");
  }
  const size_t layer_bytes = offsets[layer + 1] - offsets[layer];
  const size_t tensor_bytes = tensor.GetByteSpan().size();
  for (const auto& range : ranges) {
    if (range.offset > layer_bytes || range.bytes > layer_bytes - range.offset ||
        range.tensor_offset > tensor_bytes || range.bytes > tensor_bytes - range.tensor_offset) {
      throw std::out_of_range("Prefix snapshot transfer byte range is out of range.");
    }
  }
}

template <typename Transfer>
void TransferBatches(std::span<const TransferRange> ranges, Transfer&& transfer) {
  std::vector<TransferRange> batch;
  batch.reserve(ranges.size());
  size_t start = 0, filled = 0;
  const auto drain = [&] {
    if (filled == 0) return;
    transfer(start, std::span<const TransferRange>{batch}, filled);
    batch.clear();
    filled = 0;
  };
  for (const auto& range : ranges) {
    for (size_t copied = 0; copied < range.bytes;) {
      if (filled == kTransferBytes || (filled != 0 && start + filled != range.offset + copied)) drain();
      if (filled == 0) start = range.offset + copied;
      const size_t count = std::min(range.bytes - copied, kTransferBytes - filled);
      batch.push_back({filled, range.tensor_offset + copied, count});
      filled += count;
      copied += count;
    }
  }
  drain();
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
  const std::array ranges{Range{offset, source_offset, count}};
  Write(layer, source, ranges);
  Flush();
}

void PrefixSnapshotCache::Entry::Write(
    size_t layer, Tensor& source, std::span<const Range> ranges) {
  if (state == State::Evicted) throw std::logic_error("Cannot write an evicted prefix snapshot.");
  CheckRanges(layer_offsets, layer, source, ranges);
  if (tier != Tier::Disk) {
    if (!caches.at(layer)) throw std::logic_error("Prefix snapshot has no resident tensor.");
    if (caches[layer]->GetByteSpan().size() != layer_offsets[layer + 1] - layer_offsets[layer]) {
      throw std::logic_error("Prefix snapshot tensor has incompatible storage.");
    }
    CopyTensorRanges(*caches[layer], source, ranges, true);
    return;
  }
  const auto* data = static_cast<const uint8_t*>(source.GetRawData());
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  auto* buffer = static_cast<uint8_t*>(TransferBuffer(source.p_device_));
  TransferBatches(ranges, [&](size_t offset, std::span<const Range> batch, size_t count) {
    for (const auto& range : batch) {
      CopyBytes(cpu, buffer + range.offset, source.p_device_, data + range.tensor_offset, range.bytes);
    }
    source.p_device_->Synchronize();
    disk->file.seekp(static_cast<std::streamoff>(layer_offsets[layer] + offset));
    disk->file.write(reinterpret_cast<const char*>(buffer), static_cast<std::streamsize>(count));
  });
}

void PrefixSnapshotCache::Entry::Read(
    size_t layer, size_t offset, Tensor& destination, size_t destination_offset, size_t count) const {
  const std::array ranges{Range{offset, destination_offset, count}};
  Read(layer, destination, ranges);
}

void PrefixSnapshotCache::Entry::Read(
    size_t layer, Tensor& destination, std::span<const Range> ranges) const {
  if (state == State::Evicted) throw std::logic_error("Cannot read an evicted prefix snapshot.");
  CheckRanges(layer_offsets, layer, destination, ranges);
  if (tier != Tier::Disk) {
    if (!caches.at(layer)) throw std::logic_error("Prefix snapshot has no resident tensor.");
    if (caches[layer]->GetByteSpan().size() != layer_offsets[layer + 1] - layer_offsets[layer]) {
      throw std::logic_error("Prefix snapshot tensor has incompatible storage.");
    }
    CopyTensorRanges(destination, *caches[layer], ranges, false);
    return;
  }
  auto* data = static_cast<uint8_t*>(destination.GetMutableRawData());
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  auto* buffer = static_cast<uint8_t*>(TransferBuffer(destination.p_device_));
  TransferBatches(ranges, [&](size_t offset, std::span<const Range> batch, size_t count) {
    disk->file.seekg(static_cast<std::streamoff>(layer_offsets[layer] + offset));
    disk->file.read(reinterpret_cast<char*>(buffer), static_cast<std::streamsize>(count));
    for (const auto& range : batch) {
      CopyBytes(destination.p_device_, data + range.tensor_offset, cpu, buffer + range.offset, range.bytes);
    }
    destination.p_device_->Synchronize();
  });
}

void PrefixSnapshotCache::Entry::Flush() {
  if (disk) disk->file.flush();
}

PrefixSnapshotCache::PrefixSnapshotCache(
    std::array<size_t, 3> limits, size_t max_entries, std::filesystem::path directory,
    bool device_resident)
    : limits_{limits}, max_entries_{max_entries}, directory_{std::move(directory)}, device_resident_{device_resident} {}

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
    if (disk->directory.empty()) {
      throw std::system_error(std::make_error_code(std::errc::file_exists),
                              "Unable to create private prefix snapshot spill storage.");
    }
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
  staged.layer_offsets = entry.layer_offsets;
  AllocateStorage(staged, tier, device);
  // A move only proceeds toward a slower tier, so disk is always the destination, never the source.
  try {
    for (size_t layer = 0; layer < entry.caches.size(); ++layer) {
      const std::array ranges{Entry::Range{0, 0, entry.layer_offsets[layer + 1] - entry.layer_offsets[layer]}};
      staged.Write(layer, *entry.caches[layer], ranges);
    }
    staged.Flush();
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
  const bool webgpu = device->GetType() == DeviceType::WEBGPU;
  const bool device_resident = device_resident_ || webgpu;
  if (!webgpu && device->GetType() != DeviceType::CPU && device->GetType() != DeviceType::CUDA) {
    throw std::runtime_error("Prefix snapshots currently support CPU, CUDA, and WebGPU devices.");
  }
  if (!device_resident && limits_[0] != 0 && device->GetType() != DeviceType::CUDA) {
    throw std::runtime_error("A VRAM prefix snapshot budget requires a CUDA device.");
  }
  entry->bytes = 0;
  entry->layer_offsets.clear();
  entry->layer_offsets.push_back(0);
  for (const auto& shape : entry->shapes) {
    const size_t bytes = TensorBytes(shape, entry->type);
    if (bytes > std::numeric_limits<size_t>::max() - entry->bytes) {
      throw std::overflow_error("Prefix snapshot payload size overflows size_t.");
    }
    entry->bytes += bytes;
    entry->layer_offsets.push_back(entry->bytes);
  }
  if (entry->bytes == 0 || max_entries_ == 0 || component_max_entries == 0 ||
      (!device_resident && std::none_of(limits_.begin(), limits_.end(),
                                        [&](size_t limit) { return limit >= entry->bytes; }))) return false;
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
    if (!device_resident && !MakeRoom(static_cast<Tier>(index), entry->bytes, device)) continue;
    Entry staged;
    staged.shapes = entry->shapes;
    staged.type = entry->type;
    staged.bytes = entry->bytes;
    staged.layer_offsets = entry->layer_offsets;
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

void PrefixSnapshotCache::Discard(const Entry& entry) noexcept {
  for (auto& weak : entries_) {
    auto candidate = weak.lock();
    if (candidate.get() == &entry) {
      candidate->Release();
      candidate->state = State::Evicted;
      return;
    }
  }
}

}  // namespace Generators
