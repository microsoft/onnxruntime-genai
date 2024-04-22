// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <assert.h>
#include <wil/result.h>
#include <stdexcept>
#include "dml_readback_heap.h"
#include "dml_execution_context.h"

static ComPtr<ID3D12Resource> CreateReadbackHeap(ID3D12Device* device, size_t size) {
  ComPtr<ID3D12Resource> readback_heap;
  auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
  auto buffer = CD3DX12_RESOURCE_DESC::Buffer(size);

  THROW_IF_FAILED(device->CreateCommittedResource(
      &heap,
      D3D12_HEAP_FLAG_NONE,
      &buffer,
      D3D12_RESOURCE_STATE_COPY_DEST,
      nullptr,
      IID_PPV_ARGS(readback_heap.ReleaseAndGetAddressOf())));

  return readback_heap;
}

DmlReadbackHeap::DmlReadbackHeap(ID3D12Device* device, DmlExecutionContext* execution_context)
    : device_(device),
      execution_context_(execution_context) {
}

static size_t ComputeNewCapacity(size_t existing_capacity, size_t desired_capacity) {
  size_t new_capacity = existing_capacity;

  while (new_capacity < desired_capacity) {
    if (new_capacity >= std::numeric_limits<size_t>::max() / 2) {
      // Overflow; there's no way we can satisfy this allocation request
      THROW_HR(E_OUTOFMEMORY);
    }

    new_capacity *= 2;  // geometric growth
  }

  return new_capacity;
}

void DmlReadbackHeap::EnsureReadbackHeap(size_t size) {
  if (!readback_heap_) {
    // Initialize the readback heap for the first time
    assert(capacity_ == 0);
    capacity_ = ComputeNewCapacity(c_initial_capacity, size);
    readback_heap_ = CreateReadbackHeap(device_.Get(), capacity_);
  } else if (capacity_ < size) {
    // Ensure there's sufficient capacity
    capacity_ = ComputeNewCapacity(capacity_, size);

    readback_heap_ = nullptr;
    readback_heap_ = CreateReadbackHeap(device_.Get(), capacity_);
  }

  assert(readback_heap_->GetDesc().Width >= size);
}

void DmlReadbackHeap::ReadbackFromGpu(
    std::span<uint8_t> dst,
    ID3D12Resource* src,
    uint64_t src_offset,
    D3D12_RESOURCE_STATES src_state) {
  assert(!dst.empty());

  EnsureReadbackHeap(dst.size());

  // Copy from the source resource into the readback heap
  execution_context_->CopyBufferRegion(
      readback_heap_.Get(),
      0,
      D3D12_RESOURCE_STATE_COPY_DEST,
      src,
      src_offset,
      src_state,
      dst.size());

  // Wait for completion and map the result
  execution_context_->Flush();
  execution_context_->GetCurrentCompletionEvent().WaitForSignal();
  execution_context_->ReleaseCompletedReferences();

  // Map the readback heap and copy it into the destination
  void* readback_heap_data = nullptr;
  THROW_IF_FAILED(readback_heap_->Map(0, nullptr, &readback_heap_data));
  memcpy(dst.data(), readback_heap_data, dst.size());
  readback_heap_->Unmap(0, nullptr);
}

void DmlReadbackHeap::ReadbackFromGpu(
    std::span<void*> dst,
    std::span<const uint32_t> dst_sizes,
    std::span<ID3D12Resource*> src,
    D3D12_RESOURCE_STATES src_state) {
  assert(dst.size() == src.size());
  assert(dst_sizes.size() == src.size());

  if (dst.empty()) {
    return;
  }

  uint32_t total_size = 0;
  for (auto size : dst_sizes) {
    total_size += size;
  }

  EnsureReadbackHeap(total_size);

  // Copy from the source resource into the readback heap
  uint32_t offset = 0;
  for (uint32_t i = 0; i < dst.size(); ++i) {
    execution_context_->CopyBufferRegion(
        readback_heap_.Get(),
        offset,
        D3D12_RESOURCE_STATE_COPY_DEST,
        src[i],
        0,
        src_state,
        dst_sizes[i]);

    offset += dst_sizes[i];
  }

  // Wait for completion and map the result
  execution_context_->Flush();
  execution_context_->GetCurrentCompletionEvent().WaitForSignal();
  execution_context_->ReleaseCompletedReferences();

  // Map the readback heap and copy it into the destination
  void* readback_heap_data = nullptr;
  THROW_IF_FAILED(readback_heap_->Map(0, nullptr, &readback_heap_data));

  // Copy from the source resource into the readback heap
  offset = 0;
  for (uint32_t i = 0; i < dst.size(); ++i) {
    memcpy(dst[i], static_cast<uint8_t*>(readback_heap_data) + offset, dst_sizes[i]);
    offset += dst_sizes[i];
  }

  readback_heap_->Unmap(0, nullptr);
}