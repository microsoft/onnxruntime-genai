#include <d3dx12.h>
#include <assert.h>
#include <wil/result.h>
#include <stdexcept>
#include "dml_update_mask_kernel.h"
#include "dml_helpers.h"

namespace DmlUpdateMask_Int32 {
#include "generated_dml_shaders/update_mask_int32.h"
}

namespace DmlUpdateMask_Int64 {
#include "generated_dml_shaders/update_mask_int64.h"
}

DmlUpdateMaskKernel::DmlUpdateMaskKernel(
    ID3D12Device* d3d12_device,
    DmlExecutionContext* execution_context,
    uint32_t batch_size,
    uint32_t max_seq_len,
    ONNXTensorElementDataType dtype,
    uint32_t seq_len,
    ID3D12Resource* attention_mask_resource,
    ID3D12Resource* attention_mask_next_resource)
    : device_(d3d12_device),
      execution_context_(execution_context),
      dtype_(dtype),
      attention_mask_resource_(attention_mask_resource),
      attention_mask_next_resource_(attention_mask_next_resource) {
  constants_.element_count = batch_size * max_seq_len;
  constants_.max_seq_len = max_seq_len;
  constants_.seq_len = seq_len;
  total_element_count_ = batch_size * max_seq_len;

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
    compute_pso_desc.CS = CD3DX12_SHADER_BYTECODE(DmlUpdateMask_Int32::g_CSMain, sizeof(DmlUpdateMask_Int32::g_CSMain));
  } else if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    compute_pso_desc.CS = CD3DX12_SHADER_BYTECODE(DmlUpdateMask_Int64::g_CSMain, sizeof(DmlUpdateMask_Int64::g_CSMain));
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
  graphics_command_list_->SetComputeRootUnorderedAccessView(0, attention_mask_resource_->GetGPUVirtualAddress());
  graphics_command_list_->SetComputeRootUnorderedAccessView(1, attention_mask_next_resource_->GetGPUVirtualAddress());

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

  // Barrier before doing the copy
  std::array<D3D12_RESOURCE_BARRIER, 3> before_copy_barriers = {
      CD3DX12_RESOURCE_BARRIER::UAV(attention_mask_next_resource_.Get()),
      CD3DX12_RESOURCE_BARRIER::Transition(attention_mask_resource_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST),
      CD3DX12_RESOURCE_BARRIER::Transition(attention_mask_next_resource_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
  };
  graphics_command_list_->ResourceBarrier(static_cast<uint32_t>(before_copy_barriers.size()), before_copy_barriers.data());

  // Copy the next mask to the current mask for next iteration
  if (dtype_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    graphics_command_list_->CopyBufferRegion(attention_mask_resource_.Get(), 0, attention_mask_next_resource_.Get(), 0, constants_.element_count * sizeof(int32_t));
  } else if (dtype_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    graphics_command_list_->CopyBufferRegion(attention_mask_resource_.Get(), 0, attention_mask_next_resource_.Get(), 0, constants_.element_count * sizeof(int64_t));
  } else {
    THROW_HR(E_NOTIMPL);
  }

  // Barrier after doing the copy
  std::array<D3D12_RESOURCE_BARRIER, 2> after_copy_barriers = {
      CD3DX12_RESOURCE_BARRIER::Transition(attention_mask_resource_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
      CD3DX12_RESOURCE_BARRIER::Transition(attention_mask_next_resource_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
  };
  graphics_command_list_->ResourceBarrier(static_cast<uint32_t>(after_copy_barriers.size()), after_copy_barriers.data());

  graphics_command_list_->Close();
}