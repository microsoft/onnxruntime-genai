// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <assert.h>
#include <wil/result.h>
#include <stdexcept>
#include <algorithm>
#include "dml_pooled_upload_heap.h"
#include "dml_execution_context.h"

DmlPooledUploadHeap::DmlPooledUploadHeap(ID3D12Device* device, DmlExecutionContext* execution_context)
    : device_(device), execution_context_(execution_context) {
}

static size_t Align(size_t offset, size_t alignment) {
  assert(alignment != 0);
  return (offset + alignment - 1) & ~(alignment - 1);
}

std::optional<size_t> DmlPooledUploadHeap::FindOffsetForAllocation(const Chunk& chunk, size_t size_in_bytes) {
  assert(size_in_bytes != 0);

  if (chunk.capacity_in_bytes < size_in_bytes) {
    // This chunk isn't even big enough to accommodate this allocation
    return std::nullopt;
  }

  if (chunk.allocations.empty()) {
    // The entire chunk is empty - allocate from the beginning
    return 0;
  }

  // Chunks are used as ring buffers, which means this allocation should go after the most recent previous
  // allocation

  const auto& last_allocation = chunk.allocations.back();
  size_t new_allocation_begin = last_allocation.offset_in_chunk + last_allocation.size_in_bytes;
  new_allocation_begin = Align(new_allocation_begin, c_allocation_alignment);

  if (new_allocation_begin + size_in_bytes < new_allocation_begin) {
    // Overflow
    return std::nullopt;
  }

  const auto& first_allocation = chunk.allocations.front();
  if (first_allocation.offset_in_chunk <= last_allocation.offset_in_chunk) {
    // This is the case where there's potentially free space at the beginning and end of the chunk, but not
    // the middle:
    // e.g.
    //   |------XXXXYYYZZ------|
    //          ^^^^   ^^
    //          first  last

    if (new_allocation_begin + size_in_bytes <= chunk.capacity_in_bytes) {
      // There's enough space between the end of the last allocation and the end of the chunk
      return new_allocation_begin;
    } else {
      // Otherwise there's not enough space at the end of the chunk - try the beginning of the chunk instead
      new_allocation_begin = 0;
      if (new_allocation_begin + size_in_bytes <= first_allocation.offset_in_chunk) {
        // There was enough space between the start of the buffer, and the start of the first allocation
        return new_allocation_begin;
      }
    }
  } else {
    // This is the case where there's potentially free space in the middle of the chunk, but not at the edges
    // e.g.
    //   |YYYZZ---------XXXX-|
    //       ^^         ^^^^
    //       last       first

    if (new_allocation_begin + size_in_bytes <= first_allocation.offset_in_chunk) {
      // There's enough space between the end of the last allocation, and the start of the first one
      return new_allocation_begin;
    }
  }

  // Not enough space in this chunk to accommodate the requested allocation
  return std::nullopt;
}

/* static */ DmlPooledUploadHeap::Chunk DmlPooledUploadHeap::CreateChunk(ID3D12Device* device, size_t size_in_bytes) {
  ComPtr<ID3D12Resource> upload_buffer;
  auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
  auto buffer = CD3DX12_RESOURCE_DESC::Buffer(size_in_bytes);

  THROW_IF_FAILED(device->CreateCommittedResource(
      &heap,
      D3D12_HEAP_FLAG_NONE,
      &buffer,
      D3D12_RESOURCE_STATE_GENERIC_READ,
      nullptr,
      IID_PPV_ARGS(upload_buffer.ReleaseAndGetAddressOf())));

  return Chunk{size_in_bytes, std::move(upload_buffer)};
}

std::pair<DmlPooledUploadHeap::Chunk*, size_t> DmlPooledUploadHeap::Reserve(size_t size_in_bytes) {
  // Try to find a chunk with enough free space to accommodate the requested allocation size
  for (Chunk& chunk : chunks_) {
    std::optional<size_t> offset_for_allocation = FindOffsetForAllocation(chunk, size_in_bytes);
    if (offset_for_allocation) {
      // There's enough space in this chunk - return
      return std::make_pair(&chunk, *offset_for_allocation);
    }
  }

  // No chunks were able to accommodate the allocation - create a new chunk and return that instead

  // At least double the capacity of the pool
  const size_t new_chunk_size = std::max({total_capacity_, c_min_chunk_size, size_in_bytes});
  chunks_.push_back(CreateChunk(device_.Get(), new_chunk_size));
  total_capacity_ += new_chunk_size;

  // Allocate from the beginning of the new chunk
  return std::make_pair(&chunks_.back(), 0);
}

void DmlPooledUploadHeap::ReclaimAllocations() {
  for (Chunk& chunk : chunks_) {
    auto* allocs = &chunk.allocations;

    // Remove all allocations which have had their fences signaled - this indicates that they are no longer
    // being used by the GPU. We can stop as soon as we find an allocation which is still in use, because we
    // only use a single command queue and executions always complete in the order they were submitted.
    while (!allocs->empty() && allocs->front().done_event.IsSignaled()) {
      allocs->pop_front();
    }
  }
}

DmlGpuEvent DmlPooledUploadHeap::BeginUploadToGpu(
    ID3D12Resource* dst,
    uint64_t dst_offset,
    D3D12_RESOURCE_STATES dst_state,
    std::span<const uint8_t> src) {
  assert(!src.empty());
  assert(dst->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER);

  InvariantChecker checker(this);

  ReclaimAllocations();

  // Allocate space from the upload heap
  Chunk* chunk = nullptr;
  size_t offset_in_chunk = 0;
  std::tie(chunk, offset_in_chunk) = Reserve(src.size());

  assert(chunk != nullptr);
  assert(offset_in_chunk + src.size() <= chunk->capacity_in_bytes);

  // Map the upload heap and copy the source data into it at the specified offset
  void* upload_heap_data = nullptr;
  THROW_IF_FAILED(chunk->resource->Map(0, nullptr, &upload_heap_data));
  memcpy(static_cast<byte*>(upload_heap_data) + offset_in_chunk, src.data(), src.size());
  chunk->resource->Unmap(0, nullptr);

  // Copy from the upload heap into the destination resource
  execution_context_->CopyBufferRegion(
      dst,
      dst_offset,
      dst_state,
      chunk->resource.Get(),
      offset_in_chunk,
      D3D12_RESOURCE_STATE_GENERIC_READ,
      src.size());

  DmlGpuEvent done_event = execution_context_->GetCurrentCompletionEvent();

  execution_context_->Flush();
  done_event.WaitForSignal();

  // Add an allocation entry to the chunk
  chunk->allocations.push_back(Allocation{static_cast<size_t>(src.size()), offset_in_chunk, done_event});

  return done_event;
}

void DmlPooledUploadHeap::Trim() {
  InvariantChecker checker(this);

  ReclaimAllocations();

  // Release any chunks which have no allocations
  auto it = std::remove_if(chunks_.begin(), chunks_.end(), [](const Chunk& c) {
    return c.allocations.empty();
  });
  chunks_.erase(it, chunks_.end());

  // Re-calculate total capacity
  total_capacity_ = 0;
  for (const auto& chunk : chunks_) {
    total_capacity_ += chunk.capacity_in_bytes;
  }
}

void DmlPooledUploadHeap::AssertInvariants() {
#ifdef _DEBUG

  auto chunk_capacity_comparer = [](const Chunk& lhs, const Chunk& rhs) {
    return lhs.capacity_in_bytes < rhs.capacity_in_bytes;
  };

  // Chunks should be sorted by ascending capacity
  assert(std::is_sorted(chunks_.begin(), chunks_.end(), chunk_capacity_comparer));

  // Allocations in a chunk should be sorted by ascending fence value
  for (const auto& chunk : chunks_) {
    auto alloc_fence_value_comparer = [](const Allocation& lhs, const Allocation& rhs) {
      return lhs.done_event.fence_value < rhs.done_event.fence_value;
    };
    assert(std::is_sorted(chunk.allocations.begin(), chunk.allocations.end(), alloc_fence_value_comparer));
  }

  // Validate chunk properties
  for (const auto& chunk : chunks_) {
    assert(chunk.resource != nullptr);
    assert(chunk.capacity_in_bytes == chunk.resource->GetDesc().Width);
  }

  // Validate allocation properties
  for (const auto& chunk : chunks_) {
    for (const auto& alloc : chunk.allocations) {
      assert(alloc.offset_in_chunk + alloc.size_in_bytes <= chunk.capacity_in_bytes);
      assert(alloc.offset_in_chunk % c_allocation_alignment == 0);  // Validate alignment
    }
  }

  // Validate no overlapping allocations
  for (const auto& chunk : chunks_) {
    auto alloc_offset_comparer = [](const Allocation& lhs, const Allocation& rhs) {
      return lhs.offset_in_chunk < rhs.offset_in_chunk;
    };

    std::vector<Allocation> allocations_sorted_by_offset(chunk.allocations.begin(), chunk.allocations.end());
    std::sort(allocations_sorted_by_offset.begin(), allocations_sorted_by_offset.end(), alloc_offset_comparer);

    for (size_t i = 1; i < allocations_sorted_by_offset.size(); ++i) {
      const auto& alloc = allocations_sorted_by_offset[i - 1];
      const auto& next_alloc = allocations_sorted_by_offset[i];
      assert(alloc.offset_in_chunk + alloc.size_in_bytes <= next_alloc.offset_in_chunk);
    }
  }

  // Validate total capacity of pool
  size_t calculated_capacity = 0;
  for (const auto& chunk : chunks_) {
    calculated_capacity += chunk.capacity_in_bytes;
  }
  assert(calculated_capacity == total_capacity_);

#endif  // #ifdef _DEBUG
}