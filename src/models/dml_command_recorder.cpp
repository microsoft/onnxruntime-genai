// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define NOMINMAX
#include <assert.h>
#include <stdexcept>
#include <wil/result.h>
#include "dml_command_recorder.h"
#include "dml_command_queue.h"
#include "onnxruntime_api.h"

DmlCommandRecorder::DmlCommandRecorder(
    ID3D12Device* d3dDevice,
    IDMLDevice* dmlDevice,
    std::shared_ptr<DmlCommandQueue> commandQueue,
    Ort::Allocator& deviceAllocator,
    const OrtDmlApi* ortDmlApi)
    : m_queue(std::move(commandQueue)),
      m_d3dDevice(d3dDevice),
      m_dmlDevice(dmlDevice),
      m_descriptorPool(d3dDevice, 2048),
      m_commandAllocatorRing(d3dDevice, m_queue->GetType(), m_queue->GetCurrentCompletionEvent()),
      m_deviceAllocator(deviceAllocator),
      m_ortDmlApi(ortDmlApi) {
  THROW_IF_FAILED(dmlDevice->CreateOperatorInitializer(0, nullptr, IID_PPV_ARGS(&m_initializer)));
  THROW_IF_FAILED(dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(&m_recorder)));
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

void DmlCommandRecorder::InitializeOperator(
    IDMLCompiledOperator* op,
    const DML_BINDING_DESC& persistentResourceBinding,
    const DML_BINDING_DESC& inputArrayBinding) {
  // Reset the initializer to reference the input operator.
  IDMLCompiledOperator* ops[] = {op};
  THROW_IF_FAILED(m_initializer->Reset(ARRAYSIZE(ops), ops));

  DML_BINDING_PROPERTIES initBindingProps = m_initializer->GetBindingProperties();

  const uint32_t numDescriptors = initBindingProps.RequiredDescriptorCount;
  DmlDescriptorRange descriptorRange = m_descriptorPool.AllocDescriptors(
      numDescriptors,
      m_queue->GetNextCompletionEvent());

  // Create a binding table for initialization.
  DML_BINDING_TABLE_DESC bindingTableDesc = {};
  bindingTableDesc.Dispatchable = m_initializer.Get();
  bindingTableDesc.CPUDescriptorHandle = descriptorRange.cpuHandle;
  bindingTableDesc.GPUDescriptorHandle = descriptorRange.gpuHandle;
  bindingTableDesc.SizeInDescriptors = numDescriptors;

  ComPtr<IDMLBindingTable> bindingTable;
  THROW_IF_FAILED(m_dmlDevice->CreateBindingTable(&bindingTableDesc, IID_PPV_ARGS(&bindingTable)));

  // Create a temporary resource for initializing the op, if it's required.
  uint64_t temporaryResourceSize = initBindingProps.TemporaryResourceSize;
  if (temporaryResourceSize > 0) {
    // Allocate and immediately free a temporary buffer. The buffer resource will still be
    // alive (managed by the pool); freeing allows the resource to be shared with other operators.
    std::array<int64_t, 1> temporaryResourceShape = {static_cast<int64_t>(temporaryResourceSize)};

    ComPtr<ID3D12Resource> buffer;
    auto temp_resource = OrtValue::CreateTensor(m_deviceAllocator, temporaryResourceShape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    Ort::ThrowOnError(m_ortDmlApi->GetD3D12ResourceFromAllocation(&m_deviceAllocator, temp_resource->GetTensorMutableRawData(), &buffer));

    // Bind the temporary resource.
    DML_BUFFER_BINDING bufferBinding = {buffer.Get(), 0, temporaryResourceSize};
    DML_BINDING_DESC bindingDesc = {DML_BINDING_TYPE_BUFFER, &bufferBinding};
    bindingTable->BindTemporaryResource(&bindingDesc);
  }

  // Bind inputs, if provided.
  if (inputArrayBinding.Type != DML_BINDING_TYPE_NONE) {
    // An operator with inputs to bind MUST use a BUFFER_ARRAY.
    assert(inputArrayBinding.Type == DML_BINDING_TYPE_BUFFER_ARRAY);
    bindingTable->BindInputs(1, &inputArrayBinding);
  }

  // Bind the persistent resource, which is an output of initialization.
  if (persistentResourceBinding.Type != DML_BINDING_TYPE_NONE) {
    // Persistent resources MUST be bound as buffers.
    assert(persistentResourceBinding.Type == DML_BINDING_TYPE_BUFFER);
    bindingTable->BindOutputs(1, &persistentResourceBinding);
  }

  // Record the initialization work.
  SetDescriptorHeap(descriptorRange.heap);
  m_recorder->RecordDispatch(m_currentCommandList.Get(), m_initializer.Get(), bindingTable.Get());
  m_operationsRecordedInCurrentCommandList = true;

  // Barrier if there's an output (i.e. persistent resource), or if any temps are used.
  if ((persistentResourceBinding.Type != DML_BINDING_TYPE_NONE) ||
      (temporaryResourceSize > 0)) {
    auto uav = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
    m_currentCommandList->ResourceBarrier(1, &uav);
  }
}

void DmlCommandRecorder::ExecuteOperator(
    IDMLCompiledOperator* op,
    const DML_BINDING_DESC& persistentResourceBinding,
    std::span<const DML_BINDING_DESC> inputBindings,
    std::span<const DML_BINDING_DESC> outputBindings) {
  DML_BINDING_PROPERTIES execBindingProps = op->GetBindingProperties();

  const uint32_t numDescriptors = execBindingProps.RequiredDescriptorCount;
  DmlDescriptorRange descriptorRange = m_descriptorPool.AllocDescriptors(
      numDescriptors,
      m_queue->GetNextCompletionEvent());

  // Create a binding table for execution.
  DML_BINDING_TABLE_DESC bindingTableDesc = {};
  bindingTableDesc.Dispatchable = op;
  bindingTableDesc.CPUDescriptorHandle = descriptorRange.cpuHandle;
  bindingTableDesc.GPUDescriptorHandle = descriptorRange.gpuHandle;
  bindingTableDesc.SizeInDescriptors = numDescriptors;

  ComPtr<IDMLBindingTable> bindingTable;
  THROW_IF_FAILED(m_dmlDevice->CreateBindingTable(&bindingTableDesc, IID_PPV_ARGS(&bindingTable)));

  // Create a temporary resource for executing the op, if it's required.
  uint64_t temporaryResourceSize = execBindingProps.TemporaryResourceSize;
  if (temporaryResourceSize > 0) {
    // Allocate and immediately free a temporary buffer. The buffer resource will still be
    // alive (managed by the pool); freeing allows the resource to be shared with other operators.
    std::array<int64_t, 1> temporaryResourceShape = {static_cast<int64_t>(temporaryResourceSize)};

    ComPtr<ID3D12Resource> buffer;
    auto temp_resource = OrtValue::CreateTensor(m_deviceAllocator, temporaryResourceShape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    Ort::ThrowOnError(m_ortDmlApi->GetD3D12ResourceFromAllocation(&m_deviceAllocator, temp_resource->GetTensorMutableRawData(), &buffer));

    // Bind the temporary resource.
    DML_BUFFER_BINDING bufferBinding = {buffer.Get(), 0, temporaryResourceSize};
    DML_BINDING_DESC bindingDesc = {DML_BINDING_TYPE_BUFFER, &bufferBinding};
    bindingTable->BindTemporaryResource(&bindingDesc);
  }

  if (persistentResourceBinding.Type != DML_BINDING_TYPE_NONE) {
    bindingTable->BindPersistentResource(&persistentResourceBinding);
  }

  bindingTable->BindInputs(static_cast<uint32_t>(inputBindings.size()), inputBindings.data());
  bindingTable->BindOutputs(static_cast<uint32_t>(outputBindings.size()), outputBindings.data());

  // Record the execution work.
  SetDescriptorHeap(descriptorRange.heap);
  m_recorder->RecordDispatch(m_currentCommandList.Get(), op, bindingTable.Get());
  m_operationsRecordedInCurrentCommandList = true;

// Barrier all outputs.
#pragma warning(push)
#pragma warning(disable : 6387)
  auto uav = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
  m_currentCommandList->ResourceBarrier(1, &uav);
#pragma warning(pop)
}