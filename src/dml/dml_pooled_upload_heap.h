// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <list>
#include <vector>
#include <optional>
#include <memory>
#include "../span.h"
#include "dml_gpu_event.h"
#include "dml_execution_context.h"

// Implements a non-blocking, ring-buffer style upload heap for copying CPU data to GPU resources.
class DmlPooledUploadHeap {
 public:
  DmlPooledUploadHeap(ID3D12Device* device, DmlExecutionContext* execution_context);

  // Makes a copy of the source data and begins copying it into the destination resource, and returns a GpuEvent
  // which will become signaled when the copy is complete. The destination resource must be a default or readback
  // buffer.
  DmlGpuEvent BeginUploadToGpu(
      ID3D12Resource* dst,
      uint64_t dst_offset,
      D3D12_RESOURCE_STATES dst_state,
      std::span<const uint8_t> src);

  // Releases unused capacity.
  void Trim();

  size_t Capacity() const { return total_capacity_; }

 private:
  static constexpr size_t c_min_chunk_size = 1024 * 1024;  // 1MB
  static constexpr size_t c_allocation_alignment = 512;    // In bytes; as per D3D12 requirement for buffers

  // A suballoction from a chunk
  struct Allocation {
    size_t size_in_bytes;

    // The offset, in bytes, from the beginning of the chunk to the beginning of this allocation
    size_t offset_in_chunk;

    // The event that will be signaled to when the GPU is done executing work that uses this allocation
    DmlGpuEvent done_event;
  };

  // Represents a single contiguous upload heap from which we carve out suballocations. Ranges are suballocated
  // from the upload heap in a ring-buffer fashion.
  struct Chunk {
    size_t capacity_in_bytes;  // The total size of the upload heap, in bytes
    ComPtr<ID3D12Resource> resource;

    // Allocations are sorted by ascending fence value - that is, least to most recently allocated
    std::list<Allocation> allocations;
  };

  // Calls AssertInvariants on construction and again on destruction
  class InvariantChecker {
   public:
    InvariantChecker(DmlPooledUploadHeap* parent)
        : parent_(parent) {
      parent_->AssertInvariants();
    }

    ~InvariantChecker() {
      parent_->AssertInvariants();
    }

   private:
    DmlPooledUploadHeap* parent_;
  };

  // Attempts to find enough unused space in the supplied chunk to accommodate the given allocation size.
  // Returns the offset of that memory if successful, null if there wasn't enough space.
  static std::optional<size_t> FindOffsetForAllocation(const Chunk& chunk, size_t size_in_bytes);

  static Chunk CreateChunk(ID3D12Device* device, size_t size_in_bytes);

  // Finds or creates a chunk with enough space to accommodate an allocation of the given size, and returns a
  // pointer to the chunk and allocation offset.
  std::pair<Chunk*, size_t> Reserve(size_t size_in_bytes);

  void ReclaimAllocations();  // Frees all allocations which are no longer being used by the GPU.
  void AssertInvariants();

  ComPtr<ID3D12Device> device_;
  DmlExecutionContext* execution_context_;

  std::vector<Chunk> chunks_;  // sorted ascending by capacity (upload heap size)
  size_t total_capacity_ = 0;  // Total size of all chunks, in bytes
};