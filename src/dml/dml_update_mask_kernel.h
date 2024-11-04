#pragma once

#include <numeric>
#include <d3d12.h>
#include <wrl/client.h>
#include <vector>
#include "dml_execution_context.h"

using Microsoft::WRL::ComPtr;

class DmlUpdateMaskKernel {
 public:
  DmlUpdateMaskKernel(
      ID3D12Device* d3d12_device,
      DmlExecutionContext* execution_context,
      uint32_t batch_size,
      uint32_t max_seq_len,
      ONNXTensorElementDataType dtype,
      uint32_t seq_len,
      ID3D12Resource* attention_mask_resource,
      ID3D12Resource* attention_mask_next_resource);

  ID3D12GraphicsCommandList* GetCommandList() { return graphics_command_list_.Get(); }

 private:
  struct Constants {
    uint32_t max_seq_len;
    uint32_t seq_len;
    uint32_t element_count;
    uint32_t start_index;
  };

  ComPtr<ID3D12Device> device_;
  ComPtr<ID3D12RootSignature> root_signature_;
  ComPtr<ID3D12PipelineState> pipeline_state_;
  Constants constants_;
  DmlExecutionContext* execution_context_;

  ComPtr<ID3D12GraphicsCommandList> graphics_command_list_;
  ComPtr<ID3D12CommandAllocator> command_allocator_;
  ComPtr<ID3D12DescriptorHeap> heap_;

  ONNXTensorElementDataType dtype_;
  ComPtr<ID3D12Resource> attention_mask_resource_;
  ComPtr<ID3D12Resource> attention_mask_next_resource_;
  uint32_t total_element_count_;

  constexpr static uint32_t constant_count_ = sizeof(Constants) / sizeof(uint32_t);
  constexpr static uint32_t uav_count_ = 2;
};