#pragma once

#include <numeric>
#include <d3d12.h>
#include <wrl/client.h>
#include <vector>
#include "dml_execution_context.h"

using Microsoft::WRL::ComPtr;

class DmlIncrementValuesKernel {
 public:
  DmlIncrementValuesKernel(
      ID3D12Device* d3d12Device,
      DmlExecutionContext* executionContext,
      uint32_t elementCount,
      ONNXTensorElementDataType dtype,
      ID3D12Resource* values_resource);

  ID3D12GraphicsCommandList* GetCommandList() { return m_graphicsCommandList.Get(); }

 private:
  struct Constants {
    uint32_t elementCount;
    uint32_t startIndex;
  };

  ComPtr<ID3D12Device> m_device;
  ComPtr<ID3D12RootSignature> m_rootSignature;
  ComPtr<ID3D12PipelineState> m_pipelineState;
  Constants m_constants;
  DmlExecutionContext* m_executionContext;

  ComPtr<ID3D12GraphicsCommandList> m_graphicsCommandList;
  ComPtr<ID3D12CommandAllocator> m_commandAllocator;
  ComPtr<ID3D12DescriptorHeap> m_heap;

  ONNXTensorElementDataType dtype_;
  ComPtr<ID3D12Resource> m_values_resource;
  uint32_t m_totalElementCount;

  constexpr static uint32_t m_constantCount = sizeof(Constants) / sizeof(uint32_t);
  constexpr static uint32_t m_uavCount = 1;
};