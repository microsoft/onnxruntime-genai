#pragma once

// TODO (pavignol): Refactor

#define NOMINMAX
#include <wrl/client.h>
#include <wil/result.h>
#include <d3d12.h>
#include <DirectML.h>
#include "dml_execution_context.h"

using Microsoft::WRL::ComPtr;

struct DmlReusedCommandListState {
  // Re-usable command list, supporting descriptor heap, and DML binding table to update that heap.
  Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiledOperator;
  Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> graphicsCommandList;
  Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocator;
  Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap;
  Microsoft::WRL::ComPtr<IDMLBindingTable> bindingTable;
  Microsoft::WRL::ComPtr<ID3D12Resource> persistentResource;
  OrtValue* previousOutput = nullptr;
};

struct DmlObjects {
  ComPtr<ID3D12Device> d3d12Device;
  ComPtr<ID3D12CommandQueue> commandQueue;
  ComPtr<ID3D12CommandAllocator> commandAllocator;
  ComPtr<ID3D12GraphicsCommandList> commandList;
  ComPtr<ID3D12Resource> upload_buffer;
};

inline DmlObjects CreateDmlObjects() {
  D3D12_COMMAND_QUEUE_DESC commandQueueDescription =
      {
          D3D12_COMMAND_LIST_TYPE_COMPUTE,
          0,
          D3D12_COMMAND_QUEUE_FLAG_NONE,
          0,
      };

  DmlObjects dmlObjects;

  THROW_IF_FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dmlObjects.d3d12Device)));
  THROW_IF_FAILED(dmlObjects.d3d12Device->CreateCommandQueue(&commandQueueDescription, IID_PPV_ARGS(&dmlObjects.commandQueue)));
  THROW_IF_FAILED(dmlObjects.d3d12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&dmlObjects.commandAllocator)));
  THROW_IF_FAILED(dmlObjects.d3d12Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, dmlObjects.commandAllocator.Get(), nullptr, IID_PPV_ARGS(&dmlObjects.commandList)));
  return dmlObjects;
}

inline DmlReusedCommandListState BuildReusableCommandList(
    IDMLDevice* dml_device,
    IDMLCompiledOperator* compiled_operator,
    ID3D12Resource* persistent_resource,
    std::optional<DML_BUFFER_BINDING> persistent_resource_binding) {
  DmlReusedCommandListState command_list_state{};

  DML_BINDING_PROPERTIES execBindingProps = compiled_operator->GetBindingProperties();

  D3D12_DESCRIPTOR_HEAP_DESC desc = {};
  desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  desc.NumDescriptors = execBindingProps.RequiredDescriptorCount;
  desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;

  ComPtr<ID3D12Device> d3d_device;
  THROW_IF_FAILED(dml_device->GetParentDevice(IID_PPV_ARGS(&d3d_device)));

  THROW_IF_FAILED(d3d_device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(command_list_state.heap.ReleaseAndGetAddressOf())));

  // Create a binding table for execution.
  DML_BINDING_TABLE_DESC bindingTableDesc = {};
  bindingTableDesc.Dispatchable = compiled_operator;
  bindingTableDesc.CPUDescriptorHandle = command_list_state.heap->GetCPUDescriptorHandleForHeapStart();
  bindingTableDesc.GPUDescriptorHandle = command_list_state.heap->GetGPUDescriptorHandleForHeapStart();
  bindingTableDesc.SizeInDescriptors = execBindingProps.RequiredDescriptorCount;

  THROW_IF_FAILED(dml_device->CreateBindingTable(&bindingTableDesc, IID_PPV_ARGS(&command_list_state.bindingTable)));

  THROW_IF_FAILED(d3d_device->CreateCommandAllocator(
      D3D12_COMMAND_LIST_TYPE_COMPUTE,
      IID_PPV_ARGS(command_list_state.commandAllocator.ReleaseAndGetAddressOf())));

  THROW_IF_FAILED(d3d_device->CreateCommandList(
      0,
      D3D12_COMMAND_LIST_TYPE_COMPUTE,
      command_list_state.commandAllocator.Get(),
      nullptr,
      IID_PPV_ARGS(command_list_state.graphicsCommandList.ReleaseAndGetAddressOf())));

  if (persistent_resource) {
    DML_BINDING_DESC persistentResourceBindingDesc = {DML_BINDING_TYPE_BUFFER, persistent_resource_binding ? &*persistent_resource_binding : nullptr};
    command_list_state.bindingTable->BindPersistentResource(&persistentResourceBindingDesc);
    command_list_state.persistentResource = persistent_resource;
  }

  ID3D12DescriptorHeap* descriptorHeaps[] = {command_list_state.heap.Get()};
  command_list_state.graphicsCommandList->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps), descriptorHeaps);

  ComPtr<IDMLCommandRecorder> recorder;
  THROW_IF_FAILED(dml_device->CreateCommandRecorder(IID_PPV_ARGS(recorder.GetAddressOf())));

  recorder->RecordDispatch(command_list_state.graphicsCommandList.Get(), compiled_operator, command_list_state.bindingTable.Get());
  command_list_state.compiledOperator = compiled_operator;

  THROW_IF_FAILED(command_list_state.graphicsCommandList->Close());

  return command_list_state;
}

inline void ExecuteReusableCommandList(
    DmlExecutionContext* execution_context,
    DmlReusedCommandListState& commandListState,
    OrtAllocator& allocator,
    const OrtDmlApi* ort_dml_api,
    std::span<ID3D12Resource*> input_resources,
    std::span<const uint64_t> input_sizes,
    std::span<ID3D12Resource*> output_resources,
    std::span<const uint64_t> output_sizes,
    bool bindings_changed) {
  assert(input_resources.size() == input_sizes.size());
  assert(output_resources.size() == output_sizes.size());

  DML_BINDING_PROPERTIES execBindingProps = commandListState.compiledOperator->GetBindingProperties();

  std::vector<DML_BUFFER_BINDING> inputBindings(input_resources.size());
  std::vector<DML_BINDING_DESC> inputBindingDescs(output_resources.size());

  std::vector<DML_BUFFER_BINDING> outputBindings(output_resources.size());
  std::vector<DML_BINDING_DESC> outputBindingDescs(output_resources.size());

  if (bindings_changed) {
    // Bind the inputs
    for (uint32_t i = 0; i < inputBindings.size(); ++i) {
      inputBindings[i].Buffer = input_resources[i];
      inputBindings[i].SizeInBytes = input_sizes[i];
      inputBindingDescs[i] = {DML_BINDING_TYPE_BUFFER, &inputBindings[i]};
    }

    commandListState.bindingTable->BindInputs(static_cast<uint32_t>(inputBindingDescs.size()), inputBindingDescs.data());

    // Bind the outputs
    for (uint32_t i = 0; i < outputBindings.size(); ++i) {
      outputBindings[i].Buffer = output_resources[i];
      outputBindings[i].SizeInBytes = output_sizes[i];
      outputBindingDescs[i] = {DML_BINDING_TYPE_BUFFER, &outputBindings[i]};
    }

    commandListState.bindingTable->BindOutputs(static_cast<uint32_t>(outputBindingDescs.size()), outputBindingDescs.data());

    // Create the temporary resource
    if (execBindingProps.TemporaryResourceSize > 0) {
      ComPtr<ID3D12Resource> temporary_resource;
      std::array<int64_t, 1> persistent_resource_shape = {static_cast<int64_t>(execBindingProps.TemporaryResourceSize)};
      auto persistent_tensor = OrtValue::CreateTensor(allocator, persistent_resource_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
      Ort::ThrowOnError(ort_dml_api->GetD3D12ResourceFromAllocation(&allocator, persistent_tensor->GetTensorMutableRawData(), &temporary_resource));
    }
  }

  // Execute the command list and if it succeeds, update the fence value at which this command may be
  // re-used.
  ComPtr<ID3D12Fence> fence;
  uint64_t completionValue;
  execution_context->ExecuteCommandList(commandListState.graphicsCommandList.Get(), fence.GetAddressOf(), &completionValue);
}

inline ComPtr<IDMLCompiledOperator> CreateCastOperator(
    IDMLDevice* dml_device,
    uint32_t num_elements,
    DML_TENSOR_DATA_TYPE source_data_type,
    DML_TENSOR_DATA_TYPE target_data_type) {
  // Create the input tensor desc
  DML_BUFFER_TENSOR_DESC input_buffer_desc{};
  input_buffer_desc.Sizes = &num_elements;
  input_buffer_desc.DimensionCount = 1;
  input_buffer_desc.DataType = source_data_type;

  switch (source_data_type) {
    case DML_TENSOR_DATA_TYPE_FLOAT16:
      input_buffer_desc.TotalTensorSizeInBytes = num_elements * sizeof(Ort::Float16_t);
      break;
    case DML_TENSOR_DATA_TYPE_FLOAT32:
      input_buffer_desc.TotalTensorSizeInBytes = num_elements * sizeof(float);
      break;
    default:
      THROW_HR(E_NOTIMPL);
  }

  DML_TENSOR_DESC input_tensor_desc = {DML_TENSOR_TYPE_BUFFER, &input_buffer_desc};

  // Create the output tensor desc
  DML_BUFFER_TENSOR_DESC output_buffer_desc{};
  output_buffer_desc.Sizes = &num_elements;
  output_buffer_desc.DimensionCount = 1;
  output_buffer_desc.DataType = target_data_type;

  switch (target_data_type) {
    case DML_TENSOR_DATA_TYPE_FLOAT16:
      output_buffer_desc.TotalTensorSizeInBytes = num_elements * sizeof(Ort::Float16_t);
      break;
    case DML_TENSOR_DATA_TYPE_FLOAT32:
      output_buffer_desc.TotalTensorSizeInBytes = num_elements * sizeof(float);
      break;
    default:
      THROW_HR(E_NOTIMPL);
  }

  DML_TENSOR_DESC output_tensor_desc = {DML_TENSOR_TYPE_BUFFER, &output_buffer_desc};

  DML_CAST_OPERATOR_DESC cast_op_desc{};
  cast_op_desc.InputTensor = &input_tensor_desc;
  cast_op_desc.OutputTensor = &output_tensor_desc;
  DML_OPERATOR_DESC cast_op_dml_desc = {DML_OPERATOR_CAST, &cast_op_desc};

  ComPtr<IDMLOperator> cast_op;
  THROW_IF_FAILED(dml_device->CreateOperator(&cast_op_dml_desc, IID_PPV_ARGS(&cast_op)));

  ComPtr<IDMLCompiledOperator> compiled_cast_op;
  THROW_IF_FAILED(dml_device->CompileOperator(cast_op.Get(), DML_EXECUTION_FLAG_DESCRIPTORS_VOLATILE, IID_PPV_ARGS(&compiled_cast_op)));

  return compiled_cast_op;
}
