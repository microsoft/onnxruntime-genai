// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <span>
#include <d3d12.h>
#include <DirectML.h>
#include "dml_command_allocator_ring.h"
#include "dml_descriptor_pool.h"
#include "dml_command_queue.h"
#include "dml_descriptor_pool.h"
#include "dml_provider_factory.h"
#include "../onnxruntime_api.h"

class DmlCommandRecorder {
 public:
  DmlCommandRecorder(
      ID3D12Device* d3dDevice,
      IDMLDevice* dmlDevice,
      std::shared_ptr<DmlCommandQueue> commandQueue,
      Ort::Allocator& deviceAllocator,
      const OrtDmlApi* ortDmlApi);

  void InitializeOperator(
      IDMLCompiledOperator* op,
      const DML_BINDING_DESC& persistentResourceBinding,
      const DML_BINDING_DESC& inputArrayBinding);

  void ExecuteOperator(
      IDMLCompiledOperator* op,
      const DML_BINDING_DESC& persistentResourceBinding,
      std::span<const DML_BINDING_DESC> inputBindings,
      std::span<const DML_BINDING_DESC> outputBindings);

  void CopyBufferRegion(
      ID3D12Resource* dstBuffer,
      uint64_t dstOffset,
      ID3D12Resource* srcBuffer,
      uint64_t srcOffset,
      uint64_t byteCount);

  void ExecuteCommandList(
      ID3D12GraphicsCommandList* commandList,
      _Outptr_ ID3D12Fence** fence,
      _Out_ uint64_t* completionValue);

  ComPtr<ID3D12GraphicsCommandList> GetCommandList();

  void ResourceBarrier(std::span<const D3D12_RESOURCE_BARRIER> barriers);
  void AddUAVBarrier();

  void Open();
  void CloseAndExecute();

  bool HasUnsubmittedWork() {
    return m_operationsRecordedInCurrentCommandList;
  }

  // Forces the descriptor heap to be reset to D3D before executing future operations
  void InvalidateDescriptorHeap() {
    m_currentDescriptorHeap = nullptr;
  }

 private:
  void CloseAndExecute(_In_opt_ ID3D12GraphicsCommandList* commandList);

  std::shared_ptr<DmlCommandQueue> m_queue;
  ComPtr<ID3D12Device> m_d3dDevice;
  Microsoft::WRL::ComPtr<IDMLDevice> m_dmlDevice;
  Microsoft::WRL::ComPtr<IDMLOperatorInitializer> m_initializer;
  Microsoft::WRL::ComPtr<IDMLCommandRecorder> m_recorder;

  // Descriptors are allocated from a pool. The current heap pointer is only used to avoid redundantly
  // setting the same heap; it does not have ownership of the heap object.
  DescriptorPool m_descriptorPool;
  ID3D12DescriptorHeap* m_currentDescriptorHeap = nullptr;

  DmlCommandAllocatorRing<2> m_commandAllocatorRing;

  // The command list currently being recorded into, and whether any command have been recorded yet.
  ComPtr<ID3D12GraphicsCommandList> m_currentCommandList;
  bool m_operationsRecordedInCurrentCommandList = false;

  // A cached command list which may be re-used.
  ComPtr<ID3D12GraphicsCommandList> m_cachedCommandList;

  Ort::Allocator& m_deviceAllocator;
  const OrtDmlApi* m_ortDmlApi;

  void SetDescriptorHeap(ID3D12DescriptorHeap* descriptorHeap);
};