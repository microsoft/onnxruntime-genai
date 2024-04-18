#define NOMINMAX
#include <d3dx12.h>
#include <assert.h>
#include <wil/result.h>
#include <stdexcept>
#include "dml_update_mask_kernel.h"

namespace DmlUpdateMask_Int32 {
#include "../generated_dml_shaders/update_mask_int32.h"
}

namespace DmlUpdateMask_Int64 {
#include "../generated_dml_shaders/update_mask_int64.h"
}

static void GetNextDispatchSize(
    uint32_t elementCount,
    uint32_t numThreads,
    uint32_t& dispatch,
    uint32_t& pendingElementCount) {
  // Max threads per workgroup is 2^10 (1024). Max dispatch per dimension is 2^16. Taken together, we can dispatch a maximum of
  // 2^26 (268,435,456) threads along a single dimension. This should suffice for a majority of the workload. Therefore, even
  // though it is possible to dispatch up to (2^16)^3 workgroups simultaneously, we stick to the simpler 1D dispatch alternative.
  assert(numThreads <= D3D12_CS_THREAD_GROUP_MAX_THREADS_PER_GROUP);

  const uint32_t maxThreadsPerDispatch = numThreads * D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;

  // Compute max dispatchable elements
  const uint32_t availableThreadCount = std::min(elementCount, maxThreadsPerDispatch);

  // Compute required thread group count
  uint32_t workGroupCount1D = (availableThreadCount + numThreads - 1) / numThreads;

  // Compute min dispatch size
  dispatch = workGroupCount1D;

  // With the dispatch size computed, compute the dispatched element count
  const uint32_t dispatchedElementCount = workGroupCount1D * numThreads;

  // Update the pending element count
  pendingElementCount = (dispatchedElementCount < elementCount) ? elementCount - dispatchedElementCount : 0;
}

DmlUpdateMaskKernel::DmlUpdateMaskKernel(
    ID3D12Device* d3d12Device,
    DmlExecutionContext* executionContext,
    uint32_t batch_size,
    uint32_t max_seq_len,
    ONNXTensorElementDataType dtype,
    uint32_t seqLen,
    ID3D12Resource* attention_mask_resource,
    ID3D12Resource* attention_mask_next_resource)
    : m_device(d3d12Device),
      m_executionContext(executionContext),
      dtype_(dtype),
      m_attention_mask_resource(attention_mask_resource),
      m_attention_mask_next_resource(attention_mask_next_resource) {
  m_constants.elementCount = batch_size * max_seq_len;
  m_constants.maxSeqLen = max_seq_len;
  m_constants.seqLen = seqLen;
  m_totalElementCount = batch_size * max_seq_len;

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
    computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(DmlUpdateMask_Int32::g_CSMain, sizeof(DmlUpdateMask_Int32::g_CSMain));
  } else if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(DmlUpdateMask_Int64::g_CSMain, sizeof(DmlUpdateMask_Int64::g_CSMain));
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
  heap_desc.NumDescriptors = 2;
  heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;

  THROW_IF_FAILED(d3d12Device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(m_heap.ReleaseAndGetAddressOf())));

  ID3D12DescriptorHeap* descriptorHeaps[] = {m_heap.Get()};
  m_graphicsCommandList->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps), descriptorHeaps);

  // Set the root signature and pipeline state
  m_graphicsCommandList->SetComputeRootSignature(m_rootSignature.Get());
  m_graphicsCommandList->SetPipelineState(m_pipelineState.Get());
  m_graphicsCommandList->SetComputeRootUnorderedAccessView(0, m_attention_mask_resource->GetGPUVirtualAddress());
  m_graphicsCommandList->SetComputeRootUnorderedAccessView(1, m_attention_mask_next_resource->GetGPUVirtualAddress());

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

  // Barrier all outputs.
  std::array<D3D12_RESOURCE_BARRIER, 1> output_barriers = {
      CD3DX12_RESOURCE_BARRIER::UAV(m_attention_mask_next_resource.Get()),
  };
  m_graphicsCommandList->ResourceBarrier(static_cast<uint32_t>(output_barriers.size()), output_barriers.data());

  // Copy the next mask to the current mask for next iteration
  if (dtype_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    m_graphicsCommandList->CopyBufferRegion(m_attention_mask_resource.Get(), 0, m_attention_mask_next_resource.Get(), 0, m_constants.elementCount * sizeof(int32_t));
  } else if (dtype_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    m_graphicsCommandList->CopyBufferRegion(m_attention_mask_resource.Get(), 0, m_attention_mask_next_resource.Get(), 0, m_constants.elementCount * sizeof(int64_t));
  } else {
    THROW_HR(E_NOTIMPL);
  }

  m_graphicsCommandList->Close();
}