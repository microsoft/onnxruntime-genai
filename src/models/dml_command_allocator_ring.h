// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include "dml_gpu_event.h"

// A fixed-size ring of command allocators. Each time an allocator is retrieved, the allocator will
// be reset if its previously recorded commands have finished executing on the GPU.
template <size_t AllocatorCount>
class DmlCommandAllocatorRing {
 public:
  DmlCommandAllocatorRing(
      ID3D12Device* device,
      D3D12_COMMAND_LIST_TYPE commandListType,
      DmlGpuEvent initialEvent) {
    for (auto& info : m_commandAllocators) {
      THROW_IF_FAILED(device->CreateCommandAllocator(
          commandListType,
          IID_PPV_ARGS(info.allocator.ReleaseAndGetAddressOf())));

      info.completionEvent = initialEvent;
    }
  }

  ID3D12CommandAllocator* GetNextAllocator(DmlGpuEvent nextCompletionEvent) {
    size_t earliestOtherAllocator = (m_currentCommandAllocator + 1) % AllocatorCount;

    assert(!m_commandAllocators[m_currentCommandAllocator].completionEvent.IsSignaled() ||
           m_commandAllocators[earliestOtherAllocator].completionEvent.IsSignaled());

    if (m_commandAllocators[earliestOtherAllocator].completionEvent.IsSignaled()) {
      THROW_IF_FAILED(m_commandAllocators[earliestOtherAllocator].Get()->Reset());
      m_currentCommandAllocator = earliestOtherAllocator;
    }

    // Set the completion event for the current allocator so it can be reset eventually.
    m_commandAllocators[m_currentCommandAllocator].completionEvent = nextCompletionEvent;

    return m_commandAllocators[m_currentCommandAllocator].Get();
  }

  // Updates the completion event of the current allocator to a different value.  This is used when the caller
  // decides to issue an unrelated call to the queue such as ExecuteCommandLists which updates its fence between calling
  // GetNextAllocator and executing the work which it recorded using the allocator it received.
  void UpdateCurrentAllocatorCompletionEvent(DmlGpuEvent nextCompletionEvent) {
    m_commandAllocators[m_currentCommandAllocator].completionEvent = nextCompletionEvent;
  }

 private:
  struct CommandAllocatorInfo {
    ComPtr<ID3D12CommandAllocator> allocator;

    // The event which will be signaled when the last command list submitted using this allocator
    // completes execution on the GPU.
    DmlGpuEvent completionEvent = {};

    ID3D12CommandAllocator* Get() const { return allocator.Get(); }
  };

  std::array<CommandAllocatorInfo, AllocatorCount> m_commandAllocators;
  size_t m_currentCommandAllocator = 0;
};