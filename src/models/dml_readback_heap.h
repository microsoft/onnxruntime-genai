// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <d3d12.h>
#include "dml_execution_context.h"

// Because we never perform more than one readback at a time, we don't need anything fancy for managing the
// readback heap - just maintain a single resource and reallocate it if it's not big enough.
class DmlReadbackHeap {
 public:
  DmlReadbackHeap(ID3D12Device* device, DmlExecutionContext* executionContext);

  // Copies data from the specified GPU resource into CPU memory pointed-to by the span. This method will block
  // until the copy is complete.
  void ReadbackFromGpu(
      std::span<uint8_t> dst,
      ID3D12Resource* src,
      uint64_t srcOffset,
      D3D12_RESOURCE_STATES srcState);

  // Overload supporting batching
  void ReadbackFromGpu(
      std::span<void*> dst,
      std::span<const uint32_t> dstSizes,
      std::span<ID3D12Resource*> src,
      D3D12_RESOURCE_STATES srcState);

 private:
  void EnsureReadbackHeap(size_t size);

  static constexpr size_t c_initialCapacity = 1024 * 1024;  // 1MB

  ComPtr<ID3D12Device> m_device;
  DmlExecutionContext* execution_context_;

  ComPtr<ID3D12Resource> m_readbackHeap;
  size_t m_capacity = 0;
};
