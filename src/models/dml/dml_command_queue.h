// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <d3d12.h>
#include <deque>
#include "../span.h"
#include "dml_gpu_event.h"

// Manages a D3D12 command queue and provides a waitable fence which is signaled with a monotonically increasing
// value once each execute completes on the GPU.
class DmlCommandQueue {
 public:
  // Creates a DmlCommandQueue object that wraps an existing D3D12 queue.
  DmlCommandQueue(ID3D12CommandQueue* existing_queue);

  D3D12_COMMAND_LIST_TYPE GetType() const { return type_; }
  ComPtr<ID3D12Fence> GetFence() const { return fence_; }
  uint64_t GetLastFenceValue() const { return last_fence_value_; }

  void ExecuteCommandList(ID3D12CommandList* command_list);
  void ExecuteCommandLists(std::span<ID3D12CommandList*> command_lists);

  // Queues a wait to block the GPU until the specified fence is signaled to a given value.
  void Wait(ID3D12Fence* fence, uint64_t value);

  // Returns an event that will become signaled when everything submitted to the queue thus far has
  // completed execution on the GPU.
  DmlGpuEvent GetCurrentCompletionEvent();

  // Returns an event that will become signaled after the next ExecuteCommandLists call.
  DmlGpuEvent GetNextCompletionEvent();

  void QueueReference(IUnknown* object, bool wait_for_unsubmitted_work);

  void Close();
  void ReleaseCompletedReferences();

 private:
  struct QueuedReference {
    uint64_t fence_value;
    ComPtr<IUnknown> object;
  };

  std::deque<QueuedReference> queued_references_;

  ComPtr<ID3D12CommandQueue> queue_;
  D3D12_COMMAND_LIST_TYPE type_;

  ComPtr<ID3D12Fence> fence_;
  uint64_t last_fence_value_ = 0;
  bool closing_ = false;
};