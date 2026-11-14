// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "prefix_snapshot_cache.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <random>

#ifndef _WIN32
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "../logging.h"
#include "../filesystem.h"

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

void PrefixSnapshotCache::Entry::AddPackedBuffer(std::span<const Range> ranges) {
  if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
    throw std::invalid_argument("Packed prefix snapshot buffers require the byte element type.");
  size_t packed = 0;
  for (const auto& range : ranges) {
    if (range.bytes > std::numeric_limits<size_t>::max() - packed)
      throw std::overflow_error("Prefix snapshot size overflow.");
    packed += range.bytes;
  }
  if (packed > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      packed > std::numeric_limits<size_t>::max() - bytes)
    throw std::overflow_error("Prefix snapshot size overflow.");
  shapes.push_back({static_cast<int64_t>(packed)});
  bytes += packed;
}

bool IsPrefixCachingRequested(const Config& config) {
  return config.engine.dynamic_batching &&
         config.engine.dynamic_batching->prefix_caching.value_or(!config.model.decoder.sliding_window.has_value());
}

std::shared_ptr<PrefixSnapshotCache> CreateNativePrefixSnapshotCache(
    const Config& config, const char* disabled_reason, std::string_view warning_component) {
  if (!IsPrefixCachingRequested(config)) return nullptr;
  const auto& batching = *config.engine.dynamic_batching;
  const auto& limits = batching.snapshot_cache;
  if (batching.max_snapshots.value_or(1) == 0)
    disabled_reason = "max_snapshots is zero";
  else if (!disabled_reason && limits.Configured() && !limits.Enabled())
    disabled_reason = "snapshot_cache tier budgets are all zero";
  if (disabled_reason) {
    if (!warning_component.empty() && batching.prefix_caching.value_or(false) && g_log.enabled && g_log.warning)
      Log("warning", "%.*s prefix caching is inactive: %s.",
          static_cast<int>(warning_component.size()), warning_component.data(), disabled_reason);
    return nullptr;
  }
  const bool resident = !limits.Configured();
  return std::make_shared<PrefixSnapshotCache>(
      std::array<size_t, 3>{limits.vram_bytes, limits.ram_bytes, limits.disk_bytes},
      batching.max_snapshots.value_or(resident ? 1 : std::numeric_limits<size_t>::max()),
      std::filesystem::path{fs::path{limits.directory}.c_str()}, resident);
}

PrefixSnapshotCache::PrefixSnapshotCache(
    std::array<size_t, 3> limits, size_t max_entries, std::filesystem::path directory,
    bool device_resident)
    : limits_{limits}, max_entries_{max_entries}, directory_{std::move(directory)}, device_resident_{device_resident} {
  if (limits_[static_cast<size_t>(Tier::Disk)] != 0 && !device_resident_) {
    try {
      directory_ = std::filesystem::absolute(
          directory_.empty() ? std::filesystem::temp_directory_path() : directory_);
      if (!std::filesystem::is_directory(directory_)) {
        throw std::filesystem::filesystem_error(
            "Expected an existing directory.", directory_,
            std::make_error_code(std::errc::not_a_directory));
      }
      Entry probe;
      probe.bytes = 1;
      probe.AllocateStorage(Tier::Disk, nullptr, directory_, accounting_);
    } catch (const std::system_error& error) {
      throw std::system_error(error.code(),
                              "Invalid engine.dynamic_batching.snapshot_cache.directory '" +
                                  directory_.string() + "': " + error.what());
    }
  }
  LogEvent("configured");
}

void PrefixSnapshotCache::LogEvent(
    const char* action, const Entry* entry, const char* reason, int64_t elapsed_us) const noexcept {
  if (!g_log.enabled) return;
  try {
    Log("engine_snapshot_cache",
        "action=%s reason=%s tier=%zu bytes=%zu used_vram_bytes=%zu used_ram_bytes=%zu "
        "used_disk_bytes=%zu limit_vram_bytes=%zu limit_ram_bytes=%zu limit_disk_bytes=%zu "
        "max_entries=%zu device_resident=%d elapsed_us=%lld",
        action, reason, entry ? static_cast<size_t>(entry->tier) : limits_.size(),
        entry ? entry->bytes : size_t{0},
        accounting_->used[0], accounting_->used[1], accounting_->used[2],
        limits_[0], limits_[1], limits_[2], max_entries_,
        static_cast<int>(device_resident_), static_cast<long long>(elapsed_us));
  } catch (...) {
    // Diagnostics must not change cache publication or cleanup outcomes.
  }
}

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
    if (entry && entry->state == State::Resident &&
        (component.empty() || entry->component == component) &&
        entry->recency < oldest) {
      selected = &weak;
      oldest = entry->recency;
    }
  }
  if (!selected) return false;
  auto entry = selected->lock();
  entry->Release();
  entry->state = State::Evicted;
  LogEvent("evicted", entry.get(), "count");
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
  if (source->tier == Tier::Disk && disk_writes_disabled_) return false;
  if (source->type != destination.type || source->shapes != destination.shapes) return false;
  destination.TakeStorage(*source);
  source->state = State::Evicted;
  LogEvent("reused", &destination, "count");
  Prune();
  return true;
}

size_t PrefixSnapshotCache::Limit(Tier tier) const {
  const size_t configured = limits_[static_cast<size_t>(tier)];
  if (tier != Tier::Vram || !device_memory_limit_) return configured;
  return device_resident_ ? *device_memory_limit_ : std::min(configured, *device_memory_limit_);
}

bool PrefixSnapshotCache::SetDeviceMemoryLimit(size_t bytes, DeviceInterface* device) {
  device_memory_limit_ = bytes;
  return MakeRoom(Tier::Vram, 0, device);
}

bool PrefixSnapshotCache::MakeRoom(Tier tier, size_t bytes, DeviceInterface* device) {
  if (tier == Tier::Disk && disk_writes_disabled_) return false;
  const size_t index = static_cast<size_t>(tier);
  const size_t limit = Limit(tier);
  if (bytes > limit) return false;
  while (accounting_->used[index] > limit || bytes > limit - accounting_->used[index]) {
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
        try {
          Move(*victim, static_cast<Tier>(lower), device);
        } catch (const std::bad_alloc&) {
          continue;
        } catch (const std::system_error& error) {
          if (static_cast<Tier>(lower) != Tier::Disk) throw;
          DisableDisk(error);
          continue;
        }
        moved = true;
        break;
      }
    }
    if (!moved) {
      victim->Release();
      victim->state = State::Evicted;
      LogEvent("evicted", victim.get(), "tier_capacity");
    }
  }
  return true;
}

void PrefixSnapshotCache::DisableDisk(const std::system_error& error) {
  disk_writes_disabled_ = true;
  LogEvent("disabled", nullptr, "disk_failure");
  if (g_log.enabled && g_log.warning) {
    Log("warning",
        "Prefix snapshot disk writes disabled until Engine recreation after an operational error "
        "in engine.dynamic_batching.snapshot_cache.directory '%s': %s",
        directory_.string().c_str(), error.what());
  }
}

void PrefixSnapshotCache::Entry::PrepareStorage() {
  if (state != State::Empty || accounting) {
    throw std::logic_error("Prefix snapshot storage may only be allocated once.");
  }
  bytes = 0;
  layer_offsets.clear();
  layer_offsets.reserve(shapes.size() + 1);
  layer_offsets.push_back(0);
  for (const auto& shape : shapes) {
    const size_t count = TensorBytes(shape, type);
    if (count > std::numeric_limits<size_t>::max() - bytes) {
      throw std::overflow_error("Prefix snapshot payload size overflows size_t.");
    }
    bytes += count;
    layer_offsets.push_back(bytes);
  }
}

void PrefixSnapshotCache::Entry::AllocateStorage(
    Tier storage_tier, DeviceInterface* device, const std::filesystem::path& directory,
    const std::shared_ptr<Accounting>& owner) {
  auto& entry = *this;
  entry.tier = storage_tier;
  if (storage_tier == Tier::Disk) {
    if (entry.bytes > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
      throw std::overflow_error("Prefix snapshot exceeds the disk stream offset range.");
    }
    auto spill = std::make_unique<DiskStorage>();
    const auto& parent = directory;
    std::random_device random;
    for (size_t attempt = 0; attempt < 32; ++attempt) {
      const auto candidate = parent / ("oga-prefix-" + std::to_string(random()) + "-" + std::to_string(random()));
#ifdef _WIN32
      if (!std::filesystem::create_directory(candidate)) continue;
#else
      if (::mkdir(candidate.c_str(), S_IRWXU) != 0) {
        if (errno == EEXIST) continue;
        throw std::filesystem::filesystem_error(
            "Unable to create private prefix snapshot spill storage.", candidate,
            std::error_code{errno, std::generic_category()});
      }
#endif
      spill->directory = candidate;
      break;
    }
    if (spill->directory.empty()) {
      throw std::system_error(std::make_error_code(std::errc::file_exists),
                              "Unable to create private prefix snapshot spill storage.");
    }
    std::filesystem::permissions(spill->directory, std::filesystem::perms::owner_all,
                                 std::filesystem::perm_options::replace);
    const auto payload = spill->directory / "payload";
    auto mode = std::ios::in | std::ios::out | std::ios::binary;
#ifdef _WIN32
    mode |= std::ios::trunc;
#else
    const int descriptor = ::open(payload.c_str(), O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_RDWR,
                                  S_IRUSR | S_IWUSR);
    if (descriptor == -1) {
      throw std::filesystem::filesystem_error(
          "Unable to create private prefix snapshot payload.", payload,
          std::error_code{errno, std::generic_category()});
    }
    if (::close(descriptor) != 0) {
      throw std::filesystem::filesystem_error(
          "Unable to close prefix snapshot payload.", payload,
          std::error_code{errno, std::generic_category()});
    }
#endif
    spill->file.exceptions(std::ios::failbit | std::ios::badbit);
    spill->file.open(payload, mode);
    spill->file.seekp(static_cast<std::streamoff>(entry.bytes - 1));
    spill->file.put('\0');
    spill->file.flush();
    entry.disk = std::move(spill);
  } else {
    auto* storage_device = storage_tier == Tier::Vram ? device : GetDeviceInterface(DeviceType::CPU);
    for (const auto& shape : entry.shapes) {
      auto tensor = std::make_unique<Tensor>(storage_device, entry.type);
      if ((storage_tier == Tier::Ram || storage_tier == Tier::Vram) && device->GetType() == DeviceType::CUDA) {
        const size_t tensor_bytes = TensorBytes(shape, entry.type);
        auto& allocator = storage_device->GetAllocator();
        std::shared_ptr<void> buffer;
        if (storage_tier == Tier::Vram) {
          // Arena Alloc retains backing after demotion. Reserve bypasses that pool and Free
          // releases the allocation, so logical snapshot reclamation also reclaims device memory.
          if (!allocator.Reserve) throw std::bad_alloc{};
          auto* memory = allocator.Reserve(&allocator, tensor_bytes);
          if (!memory) throw std::bad_alloc{};
          buffer = std::shared_ptr<void>(memory, [&allocator](void* pointer) { allocator.Free(pointer); });
        } else {
          buffer = device->AllocatePinnedHost(tensor_bytes);
        }
        tensor->ort_tensor_ = OrtValue::CreateTensor(storage_device->GetAllocator().GetInfo(), buffer.get(),
                                                     tensor_bytes, shape, entry.type);
        entry.host_buffers.push_back(std::move(buffer));
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
  owner->used[static_cast<size_t>(storage_tier)] += entry.bytes;
  entry.accounting = owner;
}

void PrefixSnapshotCache::Move(Entry& entry, Tier tier, DeviceInterface* device) {
  const auto start = std::chrono::steady_clock::now();
  Entry staged;
  staged.shapes = entry.shapes;
  staged.type = entry.type;
  staged.PrepareStorage();
  staged.AllocateStorage(tier, device, directory_, accounting_);
  LogEvent("move_staged", &staged);
  // A move only proceeds toward a slower tier, so disk is always the destination, never the source.
  try {
    for (size_t layer = 0; layer < entry.caches.size(); ++layer) {
      const std::array ranges{Entry::Range{0, 0, TensorBytes(entry.shapes[layer], entry.type)}};
      staged.Write(layer, *entry.caches[layer], ranges);
    }
    staged.Flush();
    device->Synchronize();
  } catch (...) {
    device->Synchronize();
    throw;
  }
  entry.TakeStorage(staged);
  LogEvent("demoted", &entry, "tier_capacity",
           std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now() - start)
               .count());
}

bool PrefixSnapshotCache::Allocate(const std::shared_ptr<Entry>& entry, DeviceInterface* device,
                                   std::string_view component,
                                   size_t component_max_entries,
                                   bool preserve_existing) {
  entry->PrepareStorage();
  const bool webgpu = device->GetType() == DeviceType::WEBGPU;
  const bool device_resident = device_resident_ || webgpu;
  if (!webgpu && device->GetType() != DeviceType::CPU && device->GetType() != DeviceType::CUDA) {
    throw std::runtime_error("Prefix snapshots currently support CPU, CUDA, and WebGPU devices.");
  }
  if (!device_resident && limits_[0] != 0 && device->GetType() != DeviceType::CUDA) {
    throw std::runtime_error("A VRAM prefix snapshot budget requires a CUDA device.");
  }
  const auto refuse = [&](const char* reason) {
    LogEvent("refused", entry.get(), reason);
    return false;
  };
  if (device_memory_limit_ &&
      (preserve_existing ? accounting_->used[0] > Limit(Tier::Vram)
                         : !MakeRoom(Tier::Vram, 0, device)))
    return refuse("device_memory_leased");
  if (entry->bytes == 0) return refuse("empty_payload");
  if (max_entries_ == 0 || component_max_entries == 0) return refuse("count_disabled");
  if ((!device_resident || device_memory_limit_) &&
      entry->bytes > Limit(Tier::Vram) && entry->bytes > limits_[1] && entry->bytes > limits_[2]) {
    return refuse("payload_exceeds_tiers");
  }
  Prune();
  if (preserve_existing &&
      (entries_.size() >= max_entries_ ||
       (!component.empty() && Count(component) >= component_max_entries))) {
    return refuse("preserving_existing_count");
  }
  entries_.reserve(entries_.size() + 1);
  const auto publish = [&] {
    entry->component = component;
    entry->state = State::Resident;
    entry->recency = ++clock_;
    entries_.push_back(entry);
    LogEvent("allocated", entry.get());
    return true;
  };
  while (!component.empty() && Count(component) >= component_max_entries) {
    if (ReuseOldest(*entry, component)) {
      return publish();
    }
    if (!EvictOldest(component)) return refuse("component_count_leased");
  }
  while (entries_.size() >= max_entries_) {
    if (ReuseOldest(*entry)) {
      return publish();
    }
    if (!EvictOldest()) return refuse("shared_count_leased");
  }
  for (size_t index = 0; index < limits_.size(); ++index) {
    if (static_cast<Tier>(index) == Tier::Disk && disk_writes_disabled_) continue;
    if (!device_resident || device_memory_limit_) {
      const auto tier = static_cast<Tier>(index);
      if (preserve_existing) {
        if (accounting_->used[index] > Limit(tier) ||
            entry->bytes > Limit(tier) - accounting_->used[index]) continue;
      } else if (!MakeRoom(tier, entry->bytes, device)) {
        continue;
      }
    }
    Entry staged;
    staged.shapes = entry->shapes;
    staged.type = entry->type;
    staged.PrepareStorage();
    try {
      staged.AllocateStorage(static_cast<Tier>(index), device, directory_, accounting_);
    } catch (const std::system_error& error) {
      if (static_cast<Tier>(index) != Tier::Disk) throw;
      DisableDisk(error);
      throw;
    }
    entry->TakeStorage(staged);
    return publish();
  }
  return refuse("tier_capacity_leased");
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
      LogEvent("discarded", candidate.get());
      return;
    }
  }
}

}  // namespace Generators
