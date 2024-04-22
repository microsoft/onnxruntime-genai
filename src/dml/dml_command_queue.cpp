// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <assert.h>
#include <wil/result.h>
#include "dml_command_queue.h"

DmlCommandQueue::DmlCommandQueue(ID3D12CommandQueue* existing_queue)
    : queue_(existing_queue), type_(existing_queue->GetDesc().Type) {
  ComPtr<ID3D12Device> device;
  THROW_IF_FAILED(queue_->GetDevice(IID_PPV_ARGS(device.GetAddressOf())));
  THROW_IF_FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence_.ReleaseAndGetAddressOf())));
}

void DmlCommandQueue::ExecuteCommandList(ID3D12CommandList* command_list) {
  ExecuteCommandLists(std::span(&command_list, 1));
}

void DmlCommandQueue::ExecuteCommandLists(std::span<ID3D12CommandList*> command_lists) {
  queue_->ExecuteCommandLists(static_cast<uint32_t>(command_lists.size()), command_lists.data());

  ++last_fence_value_;
  THROW_IF_FAILED(queue_->Signal(fence_.Get(), last_fence_value_));
}

void DmlCommandQueue::Wait(ID3D12Fence* fence, uint64_t value) {
  THROW_IF_FAILED(queue_->Wait(fence, value));

  ++last_fence_value_;
  THROW_IF_FAILED(queue_->Signal(fence_.Get(), last_fence_value_));
}

DmlGpuEvent DmlCommandQueue::GetCurrentCompletionEvent() {
  return DmlGpuEvent{last_fence_value_, fence_};
}

DmlGpuEvent DmlCommandQueue::GetNextCompletionEvent() {
  return DmlGpuEvent{last_fence_value_ + 1, fence_};
}

void DmlCommandQueue::QueueReference(IUnknown* object, bool wait_for_unsubmitted_work) {
  // If the DmlCommandQueue is closing, then queued_references_ is being cleared -- it is not OK
  // to queue additional references at this time, since those references would be leaked. This
  // affects any objects in queued_references_ whose destructors indirectly call QueueReference;
  // for example, an allocation from BucketizedBufferAllocator attempts to queue a reference
  // to its underlying D3D resource when freed. Furthermore, these references are unnecessary
  // since Close() already blocks for scheduled GPU work before clearing queued_references_.
  if (!closing_) {
    QueuedReference queued_reference = {GetLastFenceValue(), object};

    // If something has been recorded into a command list but not submitted yet, it means that the *next* fence
    // value is the one to signal completion.
    if (wait_for_unsubmitted_work) {
      ++queued_reference.fence_value;
    }

    queued_references_.push_back(queued_reference);
  }
}

void DmlCommandQueue::Close() {
  // Wait for flushed work:
  assert(!closing_);
  closing_ = true;
  DmlGpuEvent event = GetCurrentCompletionEvent();
  event.WaitForSignal();
  queued_references_.clear();
  closing_ = false;
}

void DmlCommandQueue::ReleaseCompletedReferences() {
  uint64_t completed_value = GetFence()->GetCompletedValue();
  while (!queued_references_.empty() && queued_references_.front().fence_value <= completed_value) {
    queued_references_.pop_front();
  }
}