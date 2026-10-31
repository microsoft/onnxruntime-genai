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
    for (auto& info : command_allocators_) {
      THROW_IF_FAILED(device->CreateCommandAllocator(
          commandListType,
          IID_PPV_ARGS(info.allocator.ReleaseAndGetAddressOf())));

      info.completion_event = initialEvent;
    }
  }

  ID3D12CommandAllocator* GetNextAllocator(DmlGpuEvent next_completion_event) {
    size_t earliest_other_allocator = (current_command_allocator_ + 1) % AllocatorCount;

    assert(!command_allocators_[current_command_allocator_].completion_event.IsSignaled() ||
           command_allocators_[earliest_other_allocator].completion_event.IsSignaled());

    if (command_allocators_[earliest_other_allocator].completion_event.IsSignaled()) {
      THROW_IF_FAILED(command_allocators_[earliest_other_allocator].Get()->Reset());
      current_command_allocator_ = earliest_other_allocator;
    }

    // Set the completion event for the current allocator so it can be reset eventually.
    command_allocators_[current_command_allocator_].completion_event = next_completion_event;

    return command_allocators_[current_command_allocator_].Get();
  }

  // Updates the completion event of the current allocator to a different value.  This is used when the caller
  // decides to issue an unrelated call to the queue such as ExecuteCommandLists which updates its fence between calling
  // GetNextAllocator and executing the work which it recorded using the allocator it received.
  void UpdateCurrentAllocatorCompletionEvent(DmlGpuEvent next_completion_event) {
    command_allocators_[current_command_allocator_].completion_event = next_completion_event;
  }

 private:
  struct CommandAllocatorInfo {
    ComPtr<ID3D12CommandAllocator> allocator;

    // The event which will be signaled when the last command list submitted using this allocator
    // completes execution on the GPU.
    DmlGpuEvent completion_event = {};

    ID3D12CommandAllocator* Get() const { return allocator.Get(); }
  };

  std::array<CommandAllocatorInfo, AllocatorCount> command_allocators_;
  size_t current_command_allocator_ = 0;
};