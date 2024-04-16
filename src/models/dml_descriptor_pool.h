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
      uint32_t numDescriptors,
      DmlGpuEvent completionEvent,
      D3D12_DESCRIPTOR_HEAP_FLAGS heapFlags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);

  DmlGpuEvent GetLastCompletionEvent() const {
    return m_completionEvent;
  }

  uint32_t GetCapacity() const {
    return m_capacity;
  }

 private:
  ComPtr<ID3D12DescriptorHeap> m_heap;
  uint32_t m_capacity = 0;
  uint32_t m_size = 0;
  uint32_t m_handleIncrementSize = 0;
  CD3DX12_CPU_DESCRIPTOR_HANDLE m_headCpuHandle;
  CD3DX12_GPU_DESCRIPTOR_HANDLE m_headGpuHandle;
  D3D12_DESCRIPTOR_HEAP_FLAGS m_heapFlags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

  // Most recent GPU completion event. Allocations are always done at the end,
  // so there is no fragmentation of the heap.
  DmlGpuEvent m_completionEvent;
};

// Manages a pool of CBV/SRV/UAV descriptors.
class DescriptorPool {
 public:
  DescriptorPool(ID3D12Device* device, uint32_t initialCapacity);

  // Reserves a contiguous range of descriptors from a single descriptor heap. The
  // lifetime of the referenced descriptor heap is managed by the DescriptorPool class.
  // The caller must supply a DmlGpuEvent that informs the pool when the reserved descriptors
  // are no longer required.
  DmlDescriptorRange AllocDescriptors(
      uint32_t numDescriptors,
      DmlGpuEvent completionEvent,
      D3D12_DESCRIPTOR_HEAP_FLAGS heapFlags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);

  // Releases all descriptor heaps that contain only descriptors which have completed
  // their work on the GPU.
  void Trim();

  // Returns the total capacity of all heaps.
  uint32_t GetTotalCapacity() const;

 private:
  ComPtr<ID3D12Device> m_device;
  std::vector<DmlDescriptorHeap> m_heaps;
  const uint32_t m_initialHeapCapacity;

  void CreateHeap(uint32_t numDescriptors, D3D12_DESCRIPTOR_HEAP_FLAGS heapFlags);
};