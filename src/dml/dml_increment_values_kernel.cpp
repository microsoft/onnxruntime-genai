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
    ID3D12Device* d3d12_device,
    DmlExecutionContext* execution_context,
    uint32_t element_count,
    ONNXTensorElementDataType dtype,
    ID3D12Resource* values_resource)
    : device_(d3d12_device),
      execution_context_(execution_context),
      dtype_(dtype),
      values_resource_(values_resource) {
  constants_.element_count = element_count;
  total_element_count_ = element_count;

  // Compute root signature.
  std::vector<CD3DX12_ROOT_PARAMETER1> root_parameters;
  root_parameters.resize(uav_count_ + 1);

  for (UINT i = 0; i < uav_count_; i++) {
    root_parameters[i].InitAsUnorderedAccessView(i);
  }

  root_parameters[uav_count_].InitAsConstants(constant_count_, 0);

  CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
  desc.Init_1_1(static_cast<uint32_t>(root_parameters.size()), root_parameters.data());

  ComPtr<ID3DBlob> root_signature_blob;
  ComPtr<ID3DBlob> root_signature_error_blob;
  THROW_IF_FAILED(D3D12SerializeVersionedRootSignature(
      &desc,
      root_signature_blob.GetAddressOf(),
      root_signature_error_blob.GetAddressOf()));

  THROW_IF_FAILED(device_->CreateRootSignature(
      0,
      root_signature_blob->GetBufferPointer(),
      root_signature_blob->GetBufferSize(),
      IID_PPV_ARGS(&root_signature_)));

  D3D12_COMPUTE_PIPELINE_STATE_DESC compute_pso_desc = {};
  compute_pso_desc.pRootSignature = root_signature_.Get();

  if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    compute_pso_desc.CS = CD3DX12_SHADER_BYTECODE(DmlIncrementValues_Int32::g_CSMain, sizeof(DmlIncrementValues_Int32::g_CSMain));
  } else if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    compute_pso_desc.CS = CD3DX12_SHADER_BYTECODE(DmlIncrementValues_Int64::g_CSMain, sizeof(DmlIncrementValues_Int64::g_CSMain));
  } else {
    THROW_HR(E_NOTIMPL);
  }

  THROW_IF_FAILED(device_->CreateComputePipelineState(&compute_pso_desc, IID_PPV_ARGS(&pipeline_state_)));

  THROW_IF_FAILED(d3d12_device->CreateCommandAllocator(
      D3D12_COMMAND_LIST_TYPE_COMPUTE,
      IID_PPV_ARGS(command_allocator_.ReleaseAndGetAddressOf())));

  THROW_IF_FAILED(d3d12_device->CreateCommandList(
      0,
      D3D12_COMMAND_LIST_TYPE_COMPUTE,
      command_allocator_.Get(),
      nullptr,
      IID_PPV_ARGS(graphics_command_list_.ReleaseAndGetAddressOf())));

  D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {};
  heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  heap_desc.NumDescriptors = uav_count_;
  heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;

  THROW_IF_FAILED(d3d12_device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(heap_.ReleaseAndGetAddressOf())));

  ID3D12DescriptorHeap* descriptor_heaps[] = {heap_.Get()};
  graphics_command_list_->SetDescriptorHeaps(ARRAYSIZE(descriptor_heaps), descriptor_heaps);

  // Set the root signature and pipeline state
  graphics_command_list_->SetComputeRootSignature(root_signature_.Get());
  graphics_command_list_->SetPipelineState(pipeline_state_.Get());
  graphics_command_list_->SetComputeRootUnorderedAccessView(0, values_resource_->GetGPUVirtualAddress());

  auto pending_element_count = total_element_count_;
  auto constants = constants_;

  // Dispatch up to the maximum number of threads per iteration until
  // all elements are completed
  while (pending_element_count > 0) {
    constants.start_index = total_element_count_ - pending_element_count;

    uint32_t dispatch_size_x;

    DmlHelpers::GetNextDispatchSize(
        pending_element_count,
        256,
        dispatch_size_x,
        pending_element_count);

    // Set root constants
    graphics_command_list_->SetComputeRoot32BitConstants(
        uav_count_,       // root parameter index
        constant_count_,  // Constant count
        &constants,
        0  // offset
    );

    graphics_command_list_->Dispatch(dispatch_size_x, 1, 1);
  }

  graphics_command_list_->Close();
}