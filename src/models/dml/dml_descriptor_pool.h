// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>
#include <d3d12.h>
#include <d3dx12.h>
#include "dml_gpu_event.h"

// A contiguous range of descriptors.
struct DmlDescriptorRange {
  ID3D12DescriptorHeap* heap;
  D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle;
  D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle;
};

// Wraps an ID3D12DescriptorHeap to allocate descriptor ranges.
class DmlDescriptorHeap {
 public:
  // Wraps an existing heap.
  explicit DmlDescriptorHeap(ID3D12DescriptorHeap* heap);

  // Reserves descriptors from the end of the heap. Returns nullopt if there is
  // no space left in the heap.
  std::optional<DmlDescriptorRange> TryAllocDescriptors(
      uint32_t num_descriptors,
      DmlGpuEvent completion_event,
      D3D12_DESCRIPTOR_HEAP_FLAGS heap_flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);

  DmlGpuEvent GetLastCompletionEvent() const {
    return completion_event_;
  }

  uint32_t GetCapacity() const {
    return capacity_;
  }

 private:
  ComPtr<ID3D12DescriptorHeap> heap_;
  uint32_t capacity_ = 0;
  uint32_t size_ = 0;
  uint32_t handle_increment_size_ = 0;
  CD3DX12_CPU_DESCRIPTOR_HANDLE head_cpu_handle_;
  CD3DX12_GPU_DESCRIPTOR_HANDLE head_gpu_handle_;
  D3D12_DESCRIPTOR_HEAP_FLAGS heap_flags_ = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

  // Most recent GPU completion event. Allocations are always done at the end,
  // so there is no fragmentation of the heap.
  DmlGpuEvent completion_event_;
};

// Manages a pool of CBV/SRV/UAV descriptors.
class DescriptorPool {
 public:
  DescriptorPool(ID3D12Device* device, uint32_t initial_capacity);

  // Reserves a contiguous range of descriptors from a single descriptor heap. The
  // lifetime of the referenced descriptor heap is managed by the DescriptorPool class.
  // The caller must supply a DmlGpuEvent that informs the pool when the reserved descriptors
  // are no longer required.
  DmlDescriptorRange AllocDescriptors(
      uint32_t num_descriptors,
      DmlGpuEvent completion_event,
      D3D12_DESCRIPTOR_HEAP_FLAGS heap_flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);

  // Releases all descriptor heaps that contain only descriptors which have completed
  // their work on the GPU.
  void Trim();

  // Returns the total capacity of all heaps.
  uint32_t GetTotalCapacity() const;

 private:
  ComPtr<ID3D12Device> device_;
  std::vector<DmlDescriptorHeap> heaps_;
  const uint32_t initial_heap_capacity_;

  void CreateHeap(uint32_t num_descriptors, D3D12_DESCRIPTOR_HEAP_FLAGS heap_flags);
};