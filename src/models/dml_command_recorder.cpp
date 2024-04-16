// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <assert.h>
#include <wil/result.h>
#include "dml_command_recorder.h"
#include "dml_command_queue.h"

DmlCommandRecorder::DmlCommandRecorder(
    ID3D12Device* d3dDevice,
    std::shared_ptr<DmlCommandQueue> commandQueue)
    : m_queue(std::move(commandQueue)),
      m_d3dDevice(d3dDevice),
      m_descriptorPool(d3dDevice, 2048),
      m_commandAllocatorRing(d3dDevice, m_queue->GetType(), m_queue->GetCurrentCompletionEvent()) {
}

void DmlCommandRecorder::CopyBufferRegion(
    ID3D12Resource* dstBuffer,
    uint64_t dstOffset,
    ID3D12Resource* srcBuffer,
    uint64_t srcOffset,
    uint64_t byteCount) {
  m_currentCommandList->CopyBufferRegion(dstBuffer, dstOffset, srcBuffer, srcOffset, byteCount);
  m_operationsRecordedInCurrentCommandList = true;
}

void DmlCommandRecorder::ExecuteCommandList(
    ID3D12GraphicsCommandList* commandList,
    _Outptr_ ID3D12Fence** fence,
    _Out_ uint64_t* completionValue) {
  if (!m_operationsRecordedInCurrentCommandList) {
    // The caller can re-use relevant resources after the next set of work to be
    // flushed has completed.  Its command list hasn't been executed yet, just batched.
    DmlGpuEvent gpuEvent = m_queue->GetNextCompletionEvent();
    gpuEvent.fence.CopyTo(fence);
    *completionValue = gpuEvent.fenceValue;

    m_queue->ExecuteCommandLists(std::span<ID3D12CommandList*>(reinterpret_cast<ID3D12CommandList**>(&commandList), 1));

    // The fence value at which the current command allocator may be re-used will now be higher
    m_commandAllocatorRing.UpdateCurrentAllocatorCompletionEvent(m_queue->GetNextCompletionEvent());

    // Fail early if something horrifying happens
    THROW_IF_FAILED(m_d3dDevice->GetDeviceRemovedReason());

    return;
  }

  // Remember the descriptor heap and apply it to the next command list.  This avoids unnecessarily setting it onto
  // the D3D object lazily at a point when the operation may not be parallelized with GPU work.
  auto heap = m_currentDescriptorHeap;

  // Execute work in the current command list plus provided command list while closing the recorder.
  CloseAndExecute(commandList);
  Open();

  // Reset the descriptor heap opportunistically per above comment
  SetDescriptorHeap(heap);

  DmlGpuEvent gpuEvent = m_queue->GetCurrentCompletionEvent();
  gpuEvent.fence.CopyTo(fence);
  *completionValue = gpuEvent.fenceValue;
}

ComPtr<ID3D12GraphicsCommandList> DmlCommandRecorder::GetCommandList() {
  // Assume operations are added by the caller after this returns
  m_operationsRecordedInCurrentCommandList = true;
  return m_currentCommandList;
}

void DmlCommandRecorder::ResourceBarrier(std::span<const D3D12_RESOURCE_BARRIER> barriers) {
  m_currentCommandList->ResourceBarrier(static_cast<uint32_t>(barriers.size()), barriers.data());
  m_operationsRecordedInCurrentCommandList = true;
}

void DmlCommandRecorder::AddUAVBarrier() {
#pragma warning(suppress : 6387)
  auto barrier = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
  m_currentCommandList->ResourceBarrier(1, &barrier);
  m_operationsRecordedInCurrentCommandList = true;
}

void DmlCommandRecorder::Open() {
  assert(m_currentDescriptorHeap == nullptr);

  ID3D12CommandAllocator* allocator = m_commandAllocatorRing.GetNextAllocator(m_queue->GetNextCompletionEvent());

  if (!m_cachedCommandList) {
    THROW_IF_FAILED(m_d3dDevice->CreateCommandList(
        0,
        m_queue->GetType(),
        allocator,
        nullptr,
        IID_PPV_ARGS(m_currentCommandList.ReleaseAndGetAddressOf())));
  } else {
    m_currentCommandList = m_cachedCommandList;
    m_cachedCommandList = nullptr;
    THROW_IF_FAILED(m_currentCommandList->Reset(allocator, nullptr));
  }
}

void DmlCommandRecorder::CloseAndExecute() {
  CloseAndExecute(nullptr);
}

void DmlCommandRecorder::CloseAndExecute(_In_opt_ ID3D12GraphicsCommandList* commandList) {
  THROW_IF_FAILED(m_currentCommandList->Close());

  ID3D12GraphicsCommandList* commandListsToExecute[2] = {};
  uint32_t commandListsToExecuteCount = 0;

  if (m_operationsRecordedInCurrentCommandList) {
    commandListsToExecute[commandListsToExecuteCount++] = m_currentCommandList.Get();
  }

  if (commandList) {
    commandListsToExecute[commandListsToExecuteCount++] = commandList;
  }

  if (commandListsToExecuteCount > 0) {
    m_queue->ExecuteCommandLists(std::span<ID3D12CommandList*>(reinterpret_cast<ID3D12CommandList**>(commandListsToExecute), commandListsToExecuteCount));
  }

  m_cachedCommandList = m_currentCommandList;
  m_currentCommandList = nullptr;
  m_operationsRecordedInCurrentCommandList = false;

  // The descriptor heap must be set on the command list the next time it's opened.
  m_currentDescriptorHeap = nullptr;

  // Fail early if something horrifying happens
  THROW_IF_FAILED(m_d3dDevice->GetDeviceRemovedReason());
}

void DmlCommandRecorder::SetDescriptorHeap(ID3D12DescriptorHeap* descriptorHeap) {
  if (descriptorHeap != nullptr && descriptorHeap != m_currentDescriptorHeap) {
    m_currentDescriptorHeap = descriptorHeap;

    ID3D12DescriptorHeap* descriptorHeaps[] = {descriptorHeap};
    m_currentCommandList->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps), descriptorHeaps);
  }
}
