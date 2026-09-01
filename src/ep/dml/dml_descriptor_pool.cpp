// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <assert.h>
#include <optional>
#include <wil/result.h>
#include "dml_descriptor_pool.h"

DmlDescriptorHeap::DmlDescriptorHeap(ID3D12DescriptorHeap* heap) : heap_(heap),
                                                                   capacity_(heap->GetDesc().NumDescriptors),
                                                                   head_cpu_handle_(heap->GetCPUDescriptorHandleForHeapStart()),
                                                                   head_gpu_handle_(heap->GetGPUDescriptorHandleForHeapStart()),
                                                                   heap_flags_(heap->GetDesc().Flags) {
  ComPtr<ID3D12Device> device;
  THROW_IF_FAILED(heap->GetDevice(IID_PPV_ARGS(device.GetAddressOf())));
  handle_increment_size_ = device->GetDescriptorHandleIncrementSize(heap->GetDesc().Type);
}

std::optional<DmlDescriptorRange> DmlDescriptorHeap::TryAllocDescriptors(
    uint32_t num_descriptors,
    DmlGpuEvent completion_event,
    D3D12_DESCRIPTOR_HEAP_FLAGS heap_flags) {
  // Bail if the desired heap creation flags are incompatible with the existing heap.
  if (heap_flags_ != heap_flags) {
    return std::nullopt;
  }

  if ((completion_event_.fence != nullptr) && (completion_event_.IsSignaled())) {
    // This class always allocates descriptors from the end of the heap.
    // If the most recent completion event is signaled, then all previous
    // allocations have completed; the entire capacity is available to use.
    size_ = 0;
    head_cpu_handle_ = heap_->GetCPUDescriptorHandleForHeapStart();
    head_gpu_handle_ = heap_->GetGPUDescriptorHandleForHeapStart();
  }

  // The caller will need to create a new heap if there is no space left in this one.
  uint32_t space_remaining = capacity_ - size_;
  if (space_remaining < num_descriptors) {
    return std::nullopt;
  }

  DmlDescriptorRange range = {heap_.Get(), head_cpu_handle_, head_gpu_handle_};

  size_ += num_descriptors;
  completion_event_ = completion_event;
  head_cpu_handle_.Offset(num_descriptors, handle_increment_size_);
  head_gpu_handle_.Offset(num_descriptors, handle_increment_size_);

  return range;
}

DescriptorPool::DescriptorPool(ID3D12Device* device, uint32_t initial_capacity) : device_(device),
                                                                                  initial_heap_capacity_(initial_capacity) {
  CreateHeap(initial_capacity, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
}

DmlDescriptorRange DescriptorPool::AllocDescriptors(
    uint32_t num_descriptors,
    DmlGpuEvent completion_event,
    D3D12_DESCRIPTOR_HEAP_FLAGS heap_flags) {
  // Attempt to allocate from an existing heap.
  for (DmlDescriptorHeap& heap : heaps_) {
    auto descriptor_range = heap.TryAllocDescriptors(num_descriptors, completion_event, heap_flags);
    if (descriptor_range.has_value()) {
      return descriptor_range.value();
    }
  }

  // A new descriptor heap must be created.
  uint32_t new_heap_capacity = std::max(num_descriptors, initial_heap_capacity_);
  CreateHeap(new_heap_capacity, heap_flags);
  auto descriptor_range = heaps_.back().TryAllocDescriptors(num_descriptors, completion_event, heap_flags);
  assert(descriptor_range.has_value());
  return descriptor_range.value();
}

void DescriptorPool::Trim() {
  // Remove any heaps that are not pending execution.
  auto it = std::remove_if(heaps_.begin(), heaps_.end(), [](const DmlDescriptorHeap& heap) {
    auto completion_event = heap.GetLastCompletionEvent();
    return !completion_event.fence || completion_event.IsSignaled();
  });

  heaps_.erase(it, heaps_.end());
}

void DescriptorPool::CreateHeap(uint32_t num_descriptors, D3D12_DESCRIPTOR_HEAP_FLAGS heap_flags) {
  // This pool only manages CBV/SRV/UAV descriptors.
  D3D12_DESCRIPTOR_HEAP_DESC desc = {};
  desc.Flags = heap_flags;
  desc.NumDescriptors = num_descriptors;
  desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;

  ComPtr<ID3D12DescriptorHeap> heap;
  THROW_IF_FAILED(device_->CreateDescriptorHeap(&desc, IID_PPV_ARGS(heap.GetAddressOf())));

  heaps_.push_back(DmlDescriptorHeap{heap.Get()});
}

uint32_t DescriptorPool::GetTotalCapacity() const {
  uint32_t capacity = 0;

  for (auto& heap : heaps_) {
    capacity += heap.GetCapacity();
  }

  return capacity;
}