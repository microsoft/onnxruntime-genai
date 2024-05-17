// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <assert.h>
#include <stdexcept>
#include <wil/result.h>
#include "dml_command_recorder.h"
#include "dml_command_queue.h"
#include "../models/onnxruntime_api.h"

DmlCommandRecorder::DmlCommandRecorder(
    ID3D12Device* d3d_device,
    IDMLDevice* dml_device,
    std::shared_ptr<DmlCommandQueue> command_queue,
    Ort::Allocator& device_allocator,
    const OrtDmlApi* ort_dml_api)
    : queue_(std::move(command_queue)),
      d3d_device_(d3d_device),
      dml_device_(dml_device),
      descriptor_pool_(d3d_device, 2048),
      command_allocator_ring_(d3d_device, queue_->GetType(), queue_->GetCurrentCompletionEvent()),
      device_allocator_(device_allocator),
      ort_dml_api_(ort_dml_api) {
  THROW_IF_FAILED(dml_device->CreateOperatorInitializer(0, nullptr, IID_PPV_ARGS(&initializer_)));
  THROW_IF_FAILED(dml_device->CreateCommandRecorder(IID_PPV_ARGS(&recorder_)));
}

void DmlCommandRecorder::CopyBufferRegion(
    ID3D12Resource* dst_buffer,
    uint64_t dst_offset,
    ID3D12Resource* src_buffer,
    uint64_t src_offset,
    uint64_t byte_count) {
  current_command_list_->CopyBufferRegion(dst_buffer, dst_offset, src_buffer, src_offset, byte_count);
  operations_recorded_in_current_command_list = true;
}

void DmlCommandRecorder::ExecuteCommandList(
    ID3D12GraphicsCommandList* command_list,
    _Outptr_ ID3D12Fence** fence,
    _Out_ uint64_t* completion_value) {
  if (!operations_recorded_in_current_command_list) {
    // The caller can re-use relevant resources after the next set of work to be
    // flushed has completed.  Its command list hasn't been executed yet, just batched.
    DmlGpuEvent gpu_event = queue_->GetNextCompletionEvent();
    gpu_event.fence.CopyTo(fence);
    *completion_value = gpu_event.fence_value;

    queue_->ExecuteCommandLists(std::span<ID3D12CommandList*>(reinterpret_cast<ID3D12CommandList**>(&command_list), 1));

    // The fence value at which the current command allocator may be re-used will now be higher
    command_allocator_ring_.UpdateCurrentAllocatorCompletionEvent(queue_->GetNextCompletionEvent());

    // Fail early if something horrifying happens
    THROW_IF_FAILED(d3d_device_->GetDeviceRemovedReason());

    return;
  }

  // Remember the descriptor heap and apply it to the next command list.  This avoids unnecessarily setting it onto
  // the D3D object lazily at a point when the operation may not be parallelized with GPU work.
  auto heap = current_descriptor_heap_;

  // Execute work in the current command list plus provided command list while closing the recorder.
  CloseAndExecute(command_list);
  Open();

  // Reset the descriptor heap opportunistically per above comment
  SetDescriptorHeap(heap);

  DmlGpuEvent gpu_event = queue_->GetCurrentCompletionEvent();
  gpu_event.fence.CopyTo(fence);
  *completion_value = gpu_event.fence_value;
}

ComPtr<ID3D12GraphicsCommandList> DmlCommandRecorder::GetCommandList() {
  // Assume operations are added by the caller after this returns
  operations_recorded_in_current_command_list = true;
  return current_command_list_;
}

void DmlCommandRecorder::ResourceBarrier(std::span<const D3D12_RESOURCE_BARRIER> barriers) {
  current_command_list_->ResourceBarrier(static_cast<uint32_t>(barriers.size()), barriers.data());
  operations_recorded_in_current_command_list = true;
}

void DmlCommandRecorder::AddUAVBarrier() {
#pragma warning(suppress : 6387)
  auto barrier = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
  current_command_list_->ResourceBarrier(1, &barrier);
  operations_recorded_in_current_command_list = true;
}

void DmlCommandRecorder::Open() {
  assert(current_descriptor_heap_ == nullptr);

  ID3D12CommandAllocator* allocator = command_allocator_ring_.GetNextAllocator(queue_->GetNextCompletionEvent());

  if (!cached_command_list_) {
    THROW_IF_FAILED(d3d_device_->CreateCommandList(
        0,
        queue_->GetType(),
        allocator,
        nullptr,
        IID_PPV_ARGS(current_command_list_.ReleaseAndGetAddressOf())));
  } else {
    current_command_list_ = cached_command_list_;
    cached_command_list_ = nullptr;
    THROW_IF_FAILED(current_command_list_->Reset(allocator, nullptr));
  }
}

void DmlCommandRecorder::CloseAndExecute() {
  CloseAndExecute(nullptr);
}

void DmlCommandRecorder::CloseAndExecute(_In_opt_ ID3D12GraphicsCommandList* command_list) {
  THROW_IF_FAILED(current_command_list_->Close());

  ID3D12GraphicsCommandList* command_lists_to_execute[2] = {};
  uint32_t command_lists_to_execute_count = 0;

  if (operations_recorded_in_current_command_list) {
    command_lists_to_execute[command_lists_to_execute_count++] = current_command_list_.Get();
  }

  if (command_list) {
    command_lists_to_execute[command_lists_to_execute_count++] = command_list;
  }

  if (command_lists_to_execute_count > 0) {
    queue_->ExecuteCommandLists(std::span<ID3D12CommandList*>(reinterpret_cast<ID3D12CommandList**>(command_lists_to_execute), command_lists_to_execute_count));
  }

  cached_command_list_ = current_command_list_;
  current_command_list_ = nullptr;
  operations_recorded_in_current_command_list = false;

  // The descriptor heap must be set on the command list the next time it's opened.
  current_descriptor_heap_ = nullptr;

  // Fail early if something horrifying happens
  THROW_IF_FAILED(d3d_device_->GetDeviceRemovedReason());
}

void DmlCommandRecorder::SetDescriptorHeap(ID3D12DescriptorHeap* descriptor_heap) {
  if (descriptor_heap != nullptr && descriptor_heap != current_descriptor_heap_) {
    current_descriptor_heap_ = descriptor_heap;

    ID3D12DescriptorHeap* descriptor_heaps[] = {descriptor_heap};
    current_command_list_->SetDescriptorHeaps(ARRAYSIZE(descriptor_heaps), descriptor_heaps);
  }
}

void DmlCommandRecorder::InitializeOperator(
    IDMLCompiledOperator* op,
    const DML_BINDING_DESC& persistent_resource_binding,
    const DML_BINDING_DESC& input_array_binding) {
  // Reset the initializer to reference the input operator.
  IDMLCompiledOperator* ops[] = {op};
  THROW_IF_FAILED(initializer_->Reset(ARRAYSIZE(ops), ops));

  DML_BINDING_PROPERTIES init_binding_props = initializer_->GetBindingProperties();

  const uint32_t num_descriptors = init_binding_props.RequiredDescriptorCount;
  DmlDescriptorRange descriptor_range = descriptor_pool_.AllocDescriptors(
      num_descriptors,
      queue_->GetNextCompletionEvent());

  // Create a binding table for initialization.
  DML_BINDING_TABLE_DESC binding_table_desc = {};
  binding_table_desc.Dispatchable = initializer_.Get();
  binding_table_desc.CPUDescriptorHandle = descriptor_range.cpuHandle;
  binding_table_desc.GPUDescriptorHandle = descriptor_range.gpuHandle;
  binding_table_desc.SizeInDescriptors = num_descriptors;

  ComPtr<IDMLBindingTable> binding_table;
  THROW_IF_FAILED(dml_device_->CreateBindingTable(&binding_table_desc, IID_PPV_ARGS(&binding_table)));

  // Create a temporary resource for initializing the op, if it's required.
  uint64_t temporary_resource_size = init_binding_props.TemporaryResourceSize;
  if (temporary_resource_size > 0) {
    // Allocate and immediately free a temporary buffer. The buffer resource will still be
    // alive (managed by the pool); freeing allows the resource to be shared with other operators.
    std::array<int64_t, 1> temporary_resource_shape = {static_cast<int64_t>(temporary_resource_size)};

    ComPtr<ID3D12Resource> buffer;
    auto temp_resource = OrtValue::CreateTensor(device_allocator_, temporary_resource_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    Ort::ThrowOnError(ort_dml_api_->GetD3D12ResourceFromAllocation(&device_allocator_, temp_resource->GetTensorMutableRawData(), &buffer));

    // Bind the temporary resource.
    DML_BUFFER_BINDING buffer_binding = {buffer.Get(), 0, temporary_resource_size};
    DML_BINDING_DESC binding_desc = {DML_BINDING_TYPE_BUFFER, &buffer_binding};
    binding_table->BindTemporaryResource(&binding_desc);
  }

  // Bind inputs, if provided.
  if (input_array_binding.Type != DML_BINDING_TYPE_NONE) {
    // An operator with inputs to bind MUST use a BUFFER_ARRAY.
    assert(input_array_binding.Type == DML_BINDING_TYPE_BUFFER_ARRAY);
    binding_table->BindInputs(1, &input_array_binding);
  }

  // Bind the persistent resource, which is an output of initialization.
  if (persistent_resource_binding.Type != DML_BINDING_TYPE_NONE) {
    // Persistent resources MUST be bound as buffers.
    assert(persistent_resource_binding.Type == DML_BINDING_TYPE_BUFFER);
    binding_table->BindOutputs(1, &persistent_resource_binding);
  }

  // Record the initialization work.
  SetDescriptorHeap(descriptor_range.heap);
  recorder_->RecordDispatch(current_command_list_.Get(), initializer_.Get(), binding_table.Get());
  operations_recorded_in_current_command_list = true;

  // Barrier if there's an output (i.e. persistent resource), or if any temps are used.
  if ((persistent_resource_binding.Type != DML_BINDING_TYPE_NONE) ||
      (temporary_resource_size > 0)) {
    auto uav = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
    current_command_list_->ResourceBarrier(1, &uav);
  }
}

void DmlCommandRecorder::ExecuteOperator(
    IDMLCompiledOperator* op,
    const DML_BINDING_DESC& persistent_resource_binding,
    std::span<const DML_BINDING_DESC> input_bindings,
    std::span<const DML_BINDING_DESC> output_bindings) {
  DML_BINDING_PROPERTIES exec_binding_props = op->GetBindingProperties();

  const uint32_t num_descriptors = exec_binding_props.RequiredDescriptorCount;
  DmlDescriptorRange descriptor_range = descriptor_pool_.AllocDescriptors(
      num_descriptors,
      queue_->GetNextCompletionEvent());

  // Create a binding table for execution.
  DML_BINDING_TABLE_DESC binding_table_desc = {};
  binding_table_desc.Dispatchable = op;
  binding_table_desc.CPUDescriptorHandle = descriptor_range.cpuHandle;
  binding_table_desc.GPUDescriptorHandle = descriptor_range.gpuHandle;
  binding_table_desc.SizeInDescriptors = num_descriptors;

  ComPtr<IDMLBindingTable> binding_table;
  THROW_IF_FAILED(dml_device_->CreateBindingTable(&binding_table_desc, IID_PPV_ARGS(&binding_table)));

  // Create a temporary resource for executing the op, if it's required.
  uint64_t temporary_resource_size = exec_binding_props.TemporaryResourceSize;
  if (temporary_resource_size > 0) {
    // Allocate and immediately free a temporary buffer. The buffer resource will still be
    // alive (managed by the pool); freeing allows the resource to be shared with other operators.
    std::array<int64_t, 1> temporary_resource_shape = {static_cast<int64_t>(temporary_resource_size)};

    ComPtr<ID3D12Resource> buffer;
    auto temp_resource = OrtValue::CreateTensor(device_allocator_, temporary_resource_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    Ort::ThrowOnError(ort_dml_api_->GetD3D12ResourceFromAllocation(&device_allocator_, temp_resource->GetTensorMutableRawData(), &buffer));

    // Bind the temporary resource.
    DML_BUFFER_BINDING buffer_binding = {buffer.Get(), 0, temporary_resource_size};
    DML_BINDING_DESC binding_desc = {DML_BINDING_TYPE_BUFFER, &buffer_binding};
    binding_table->BindTemporaryResource(&binding_desc);
  }

  if (persistent_resource_binding.Type != DML_BINDING_TYPE_NONE) {
    binding_table->BindPersistentResource(&persistent_resource_binding);
  }

  binding_table->BindInputs(static_cast<uint32_t>(input_bindings.size()), input_bindings.data());
  binding_table->BindOutputs(static_cast<uint32_t>(output_bindings.size()), output_bindings.data());

  // Record the execution work.
  SetDescriptorHeap(descriptor_range.heap);
  recorder_->RecordDispatch(current_command_list_.Get(), op, binding_table.Get());
  operations_recorded_in_current_command_list = true;

// Barrier all outputs.
#pragma warning(push)
#pragma warning(disable : 6387)
  auto uav = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
  current_command_list_->ResourceBarrier(1, &uav);
#pragma warning(pop)
}