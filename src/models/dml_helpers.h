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

inline void GetNextDispatchSize(
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

inline DML_TENSOR_DATA_TYPE OrtToDmlDataType(ONNXTensorElementDataType ort_dtype) {
  switch (ort_dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return DML_TENSOR_DATA_TYPE_FLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return DML_TENSOR_DATA_TYPE_FLOAT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return DML_TENSOR_DATA_TYPE_FLOAT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return DML_TENSOR_DATA_TYPE_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return DML_TENSOR_DATA_TYPE_UINT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return DML_TENSOR_DATA_TYPE_UINT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return DML_TENSOR_DATA_TYPE_UINT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return DML_TENSOR_DATA_TYPE_INT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return DML_TENSOR_DATA_TYPE_INT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return DML_TENSOR_DATA_TYPE_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return DML_TENSOR_DATA_TYPE_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return DML_TENSOR_DATA_TYPE_UINT8;
    default:
      THROW_HR(E_NOTIMPL);
  }
}

inline void DmlCastInputToOutput(
    DmlExecutionContext* execution_context,
    OrtAllocator& allocator,
    OrtValue& in,
    std::unique_ptr<OrtValue>& p_out,
    IDMLDevice* dml_device,
    const OrtDmlApi* ort_dml_api,
    DmlReusedCommandListState& command_list_state) {
  auto shape_info = in.GetTensorTypeAndShapeInfo();
  auto shape = shape_info->GetShape();
  assert(shape_info->GetElementType() == Ort::TypeToTensorType<Ort::Float16_t>::type);

  bool allocate_p_out = p_out == nullptr;
  if (p_out) {
    auto out_shape_info = p_out->GetTensorTypeAndShapeInfo();
    auto out_shape = out_shape_info->GetShape();
    allocate_p_out = shape != out_shape;
  }

  if (allocate_p_out) {
    p_out = OrtValue::CreateTensor<float>(allocator, shape);
  }

  int element_count = static_cast<int>(shape_info->GetElementCount());

  bool rebind = command_list_state.previousOutput != p_out.get();

  // If the sizes change, we need to recompile the operator and rebuild the command lists. It should only happen
  // once after the very first iteration.
  if (rebind) {
    auto dml_from_type = OrtToDmlDataType(in.GetTensorTypeAndShapeInfo()->GetElementType());
    auto dml_to_type = OrtToDmlDataType(p_out->GetTensorTypeAndShapeInfo()->GetElementType());
    auto compiled_cast_operator = CreateCastOperator(dml_device, element_count, dml_from_type, dml_to_type);

    ComPtr<ID3D12Resource> persistent_resource;
    uint64_t persistent_resource_size = compiled_cast_operator->GetBindingProperties().PersistentResourceSize;

    std::optional<DML_BUFFER_BINDING> persistent_resource_binding;

    if (persistent_resource_size > 0) {
      std::array<int64_t, 1> persistent_resource_shape = {static_cast<int64_t>(persistent_resource_size)};
      auto persistent_tensor = OrtValue::CreateTensor(allocator, persistent_resource_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
      Ort::ThrowOnError(ort_dml_api->GetD3D12ResourceFromAllocation(&allocator, persistent_tensor->GetTensorMutableRawData(), &persistent_resource));
      persistent_resource_binding = DML_BUFFER_BINDING{persistent_resource.Get(), 0, persistent_resource_size};
    }

    DML_BINDING_DESC persistent_resource_bindingDesc = persistent_resource_binding
                                                           ? DML_BINDING_DESC{DML_BINDING_TYPE_BUFFER, &*persistent_resource_binding}
                                                           : DML_BINDING_DESC{DML_BINDING_TYPE_NONE, nullptr};

    DML_BINDING_DESC inputArrayBindingDesc = DML_BINDING_DESC{DML_BINDING_TYPE_NONE, nullptr};
    execution_context->InitializeOperator(compiled_cast_operator.Get(), persistent_resource_bindingDesc, inputArrayBindingDesc);
    command_list_state = BuildReusableCommandList(dml_device, compiled_cast_operator.Get(), persistent_resource.Get(), persistent_resource_binding);
    command_list_state.previousOutput = p_out.get();
  }

  ComPtr<ID3D12Resource> source_resource;
  Ort::ThrowOnError(ort_dml_api->GetD3D12ResourceFromAllocation(&allocator, in.GetTensorMutableData<uint8_t>(), &source_resource));

  ComPtr<ID3D12Resource> target_resource;
  Ort::ThrowOnError(ort_dml_api->GetD3D12ResourceFromAllocation(&allocator, p_out->GetTensorMutableData<uint8_t>(), &target_resource));

  std::array<ID3D12Resource*, 1> input_resources = {source_resource.Get()};
  std::array<uint64_t, 1> input_sizes = {element_count * sizeof(Ort::Float16_t)};

  std::array<ID3D12Resource*, 1> output_resources = {target_resource.Get()};
  ;
  std::array<uint64_t, 1> output_sizes = {element_count * sizeof(float)};

  ExecuteReusableCommandList(execution_context, command_list_state, allocator, ort_dml_api, input_resources, input_sizes, output_resources, output_sizes, rebind);
}
