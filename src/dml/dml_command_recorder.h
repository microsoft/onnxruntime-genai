// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <d3d12.h>
#include <DirectML.h>
#include "../span.h"
#include "dml_command_allocator_ring.h"
#include "dml_descriptor_pool.h"
#include "dml_command_queue.h"
#include "dml_descriptor_pool.h"
#include "dml_provider_factory.h"

struct OrtDmlApi;

namespace Ort {
struct Allocator;
}

class DmlCommandRecorder {
 public:
  DmlCommandRecorder(
      ID3D12Device* d3d_device,
      IDMLDevice* dml_device,
      std::shared_ptr<DmlCommandQueue> command_queue,
      Ort::Allocator& device_allocator,
      const OrtDmlApi* ort_dml_api);

  void InitializeOperator(
      IDMLCompiledOperator* op,
      const DML_BINDING_DESC& persistent_resource_binding,
      const DML_BINDING_DESC& input_array_binding);

  void ExecuteOperator(
      IDMLCompiledOperator* op,
      const DML_BINDING_DESC& persistent_resource_binding,
      std::span<const DML_BINDING_DESC> input_bindings,
      std::span<const DML_BINDING_DESC> output_bindings);

  void CopyBufferRegion(
      ID3D12Resource* dst_buffer,
      uint64_t dst_offset,
      ID3D12Resource* src_buffer,
      uint64_t src_offset,
      uint64_t byte_count);

  void ExecuteCommandList(
      ID3D12GraphicsCommandList* command_list,
      _Outptr_ ID3D12Fence** fence,
      _Out_ uint64_t* completion_value);

  ComPtr<ID3D12GraphicsCommandList> GetCommandList();

  void ResourceBarrier(std::span<const D3D12_RESOURCE_BARRIER> barriers);
  void AddUAVBarrier();

  void Open();
  void CloseAndExecute();

  bool HasUnsubmittedWork() {
    return operations_recorded_in_current_command_list;
  }

  // Forces the descriptor heap to be reset to D3D before executing future operations
  void InvalidateDescriptorHeap() {
    current_descriptor_heap_ = nullptr;
  }

 private:
  void CloseAndExecute(_In_opt_ ID3D12GraphicsCommandList* command_list);

  std::shared_ptr<DmlCommandQueue> queue_;
  ComPtr<ID3D12Device> d3d_device_;
  Microsoft::WRL::ComPtr<IDMLDevice> dml_device_;
  Microsoft::WRL::ComPtr<IDMLOperatorInitializer> initializer_;
  Microsoft::WRL::ComPtr<IDMLCommandRecorder> recorder_;

  // Descriptors are allocated from a pool. The current heap pointer is only used to avoid redundantly
  // setting the same heap; it does not have ownership of the heap object.
  DescriptorPool descriptor_pool_;
  ID3D12DescriptorHeap* current_descriptor_heap_ = nullptr;

  DmlCommandAllocatorRing<2> command_allocator_ring_;

  // The command list currently being recorded into, and whether any command have been recorded yet.
  ComPtr<ID3D12GraphicsCommandList> current_command_list_;
  bool operations_recorded_in_current_command_list = false;

  // A cached command list which may be re-used.
  ComPtr<ID3D12GraphicsCommandList> cached_command_list_;

  Ort::Allocator& device_allocator_;
  const OrtDmlApi* ort_dml_api_;

  void SetDescriptorHeap(ID3D12DescriptorHeap* descriptor_heap);
};