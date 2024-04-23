// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdexcept>
#include <assert.h>
#include "dml_execution_context.h"
#include "dml_command_queue.h"

DmlExecutionContext::DmlExecutionContext(
    ID3D12Device* d3d12_device,
    IDMLDevice* dml_device,
    ID3D12CommandQueue* queue,
    Ort::Allocator& device_allocator,
    const OrtDmlApi* ort_dml_api)
    : queue_(std::make_shared<DmlCommandQueue>(queue)), dml_recorder_(d3d12_device, dml_device, queue_, device_allocator, ort_dml_api) {
}

void DmlExecutionContext::CopyBufferRegion(
    ID3D12Resource* dst_buffer,
    uint64_t dst_offset,
    D3D12_RESOURCE_STATES dst_state,
    ID3D12Resource* src_buffer,
    uint64_t src_offset,
    D3D12_RESOURCE_STATES src_state,
    uint64_t byte_count) {
  assert(!closed_);

  SetCommandRecorder(&dml_recorder_);

  std::vector<D3D12_RESOURCE_BARRIER> barriers;

  if (!(dst_state & D3D12_RESOURCE_STATE_COPY_DEST)) {
    barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(dst_buffer, dst_state, D3D12_RESOURCE_STATE_COPY_DEST));
  }
  if (!(src_state & D3D12_RESOURCE_STATE_COPY_SOURCE)) {
    barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(src_buffer, src_state, D3D12_RESOURCE_STATE_COPY_SOURCE));
  }

  if (!barriers.empty()) {
    dml_recorder_.ResourceBarrier(barriers);
  }

  dml_recorder_.CopyBufferRegion(dst_buffer, dst_offset, src_buffer, src_offset, byte_count);

  // Reset barrier state
  if (!barriers.empty()) {
    for (auto& barrier : barriers) {
      std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
    }

    dml_recorder_.ResourceBarrier(barriers);
  }
}

void DmlExecutionContext::InitializeOperator(
    IDMLCompiledOperator* op,
    const DML_BINDING_DESC& persistent_resource_binding,
    const DML_BINDING_DESC& input_array_binding) {
  assert(!closed_);
  SetCommandRecorder(&dml_recorder_);

  dml_recorder_.InitializeOperator(op, persistent_resource_binding, input_array_binding);
}

void DmlExecutionContext::ExecuteCommandList(
    ID3D12GraphicsCommandList* command_list,
    _Outptr_ ID3D12Fence** fence,
    _Out_ uint64_t* completion_value) {
  assert(!closed_);

  SetCommandRecorder(&dml_recorder_);
  dml_recorder_.ExecuteCommandList(command_list, fence, completion_value);
}

void DmlExecutionContext::AddUAVBarrier() {
  assert(!closed_);
  SetCommandRecorder(&dml_recorder_);

  dml_recorder_.AddUAVBarrier();
}

void DmlExecutionContext::ResourceBarrier(std::span<const D3D12_RESOURCE_BARRIER> barriers) {
  assert(!closed_);
  SetCommandRecorder(&dml_recorder_);

  dml_recorder_.ResourceBarrier(barriers);
}

void DmlExecutionContext::GetCommandListForRecordingAndInvalidateState(ID3D12GraphicsCommandList** command_list) {
  assert(!closed_);
  SetCommandRecorder(&dml_recorder_);

  // Ensure the descriptor heap is reset to D3D as something external may change it before recording
  dml_recorder_.InvalidateDescriptorHeap();

  dml_recorder_.GetCommandList().CopyTo(command_list);
}

void DmlExecutionContext::SetCommandRecorder(DmlCommandRecorder* new_recorder) {
  assert(!closed_);

  // If changing which recorder is the current one, we need to flush the old one first. This is to ensure correct
  // ordering of operations on the command queue.
  if (current_recorder_ != new_recorder) {
    Flush();
    current_recorder_ = new_recorder;

    if (current_recorder_ != nullptr) {
      current_recorder_->Open();
    }
  }
}

void DmlExecutionContext::Flush() {
  assert(!closed_);

  if (!current_recorder_ || !current_recorder_->HasUnsubmittedWork()) {
    // Nothing to flush
    return;
  }

  current_recorder_->CloseAndExecute();
  ReleaseCompletedReferences();

  // Pre-emptively set the DML command recorder.  It's the only command recorder right now,
  // and doing this here causes work and allocations resetting the command list to occur at
  // a point where it's going to be parallelized with GPU work.
  current_recorder_ = nullptr;
  SetCommandRecorder(&dml_recorder_);
}

void DmlExecutionContext::QueueReference(IUnknown* object) {
  assert(!closed_);
  // If something has been recorded into a command list but not submitted yet, it means that the *next* fence
  // value is the one to signal completion.
  bool wait_for_unsubmitted_work = (current_recorder_ != nullptr);
  queue_->QueueReference(object, wait_for_unsubmitted_work);
}

void DmlExecutionContext::Close() {
  assert(!closed_);

  // Discard unflushed work and clear queued references.  This prevents the circular reference:
  // Kernel --> ProviderImpl -->  Context --> QueuedRefs --> Kernel
  queue_->Close();
  current_recorder_ = nullptr;
  closed_ = true;
}

DmlGpuEvent DmlExecutionContext::GetCurrentCompletionEvent() {
  assert(!closed_);

  DmlGpuEvent event = queue_->GetCurrentCompletionEvent();

  // If something has been recorded into a command list but not submitted yet, it means that the *next* fence
  // value is the one to signal completion.
  const bool unflushed_work_exists = (current_recorder_ != nullptr) && current_recorder_->HasUnsubmittedWork();
  if (unflushed_work_exists) {
    ++event.fence_value;
  }

  return event;
}

void DmlExecutionContext::ReleaseCompletedReferences() {
  assert(!closed_);
  queue_->ReleaseCompletedReferences();
}

D3D12_COMMAND_LIST_TYPE DmlExecutionContext::GetCommandListTypeForQueue() const {
  assert(!closed_);
  return queue_->GetType();
}