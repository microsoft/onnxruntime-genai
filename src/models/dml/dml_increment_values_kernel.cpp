#include <d3dx12.h>
#include <assert.h>
#include <wil/result.h>
#include <stdexcept>
#include "dml_increment_values_kernel.h"
#include "dml_helpers.h"

namespace DmlIncrementValues_Int32 {
#include "generated_dml_shaders/increment_values_int32.h"
}

namespace DmlIncrementValues_Int64 {
#include "generated_dml_shaders/increment_values_int64.h"
}

DmlIncrementValuesKernel::DmlIncrementValuesKernel(
    ID3D12Device* d3d12Device,
    DmlExecutionContext* executionContext,
    uint32_t elementCount,
    ONNXTensorElementDataType dtype,
    ID3D12Resource* values_resource)
    : m_device(d3d12Device),
      m_executionContext(executionContext),
      dtype_(dtype),
      m_values_resource(values_resource) {
  m_constants.elementCount = elementCount;
  m_totalElementCount = elementCount;

  // Compute root signature.
  std::vector<CD3DX12_ROOT_PARAMETER1> rootParameters;
  rootParameters.resize(m_uavCount + 1);

  for (UINT i = 0; i < m_uavCount; i++) {
    rootParameters[i].InitAsUnorderedAccessView(i);
  }

  rootParameters[m_uavCount].InitAsConstants(m_constantCount, 0);

  CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
  desc.Init_1_1(static_cast<uint32_t>(rootParameters.size()), rootParameters.data());

  ComPtr<ID3DBlob> rootSignatureBlob;
  ComPtr<ID3DBlob> rootSignatureErrorBlob;
  THROW_IF_FAILED(D3D12SerializeVersionedRootSignature(
      &desc,
      rootSignatureBlob.GetAddressOf(),
      rootSignatureErrorBlob.GetAddressOf()));

  THROW_IF_FAILED(m_device->CreateRootSignature(
      0,
      rootSignatureBlob->GetBufferPointer(),
      rootSignatureBlob->GetBufferSize(),
      IID_PPV_ARGS(&m_rootSignature)));

  D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
  computePsoDesc.pRootSignature = m_rootSignature.Get();

  if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(DmlIncrementValues_Int32::g_CSMain, sizeof(DmlIncrementValues_Int32::g_CSMain));
  } else if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(DmlIncrementValues_Int64::g_CSMain, sizeof(DmlIncrementValues_Int64::g_CSMain));
  } else {
    THROW_HR(E_NOTIMPL);
  }

  THROW_IF_FAILED(m_device->CreateComputePipelineState(&computePsoDesc, IID_PPV_ARGS(&m_pipelineState)));

  THROW_IF_FAILED(d3d12Device->CreateCommandAllocator(
      D3D12_COMMAND_LIST_TYPE_COMPUTE,
      IID_PPV_ARGS(m_commandAllocator.ReleaseAndGetAddressOf())));

  THROW_IF_FAILED(d3d12Device->CreateCommandList(
      0,
      D3D12_COMMAND_LIST_TYPE_COMPUTE,
      m_commandAllocator.Get(),
      nullptr,
      IID_PPV_ARGS(m_graphicsCommandList.ReleaseAndGetAddressOf())));

  D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {};
  heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  heap_desc.NumDescriptors = m_uavCount;
  heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;

  THROW_IF_FAILED(d3d12Device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(m_heap.ReleaseAndGetAddressOf())));

  ID3D12DescriptorHeap* descriptorHeaps[] = {m_heap.Get()};
  m_graphicsCommandList->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps), descriptorHeaps);

  // Set the root signature and pipeline state
  m_graphicsCommandList->SetComputeRootSignature(m_rootSignature.Get());
  m_graphicsCommandList->SetPipelineState(m_pipelineState.Get());
  m_graphicsCommandList->SetComputeRootUnorderedAccessView(0, m_values_resource->GetGPUVirtualAddress());

  auto pendingElementCount = m_totalElementCount;
  auto constants = m_constants;

  // Dispatch up to the maximum number of threads per iteration until
  // all elements are completed
  while (pendingElementCount > 0) {
    constants.startIndex = m_totalElementCount - pendingElementCount;

    uint32_t dispatchSizeX;

    GetNextDispatchSize(
        pendingElementCount,
        256,
        dispatchSizeX,
        pendingElementCount);

    // Set root constants
    m_graphicsCommandList->SetComputeRoot32BitConstants(
        m_uavCount,       // root parameter index
        m_constantCount,  // Constant count
        &constants,
        0  // offset
    );

    m_graphicsCommandList->Dispatch(dispatchSizeX, 1, 1);
  }

  m_graphicsCommandList->Close();
}