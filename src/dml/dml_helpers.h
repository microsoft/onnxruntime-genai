#pragma once

#include <winrt/base.h>
#include <d3d12.h>
#include <DirectML.h>
#include "dml_execution_context.h"


struct DmlReusedCommandListState {
  // Re-usable command list, supporting descriptor heap, and DML binding table to update that heap.
  winrt::com_ptr<IDMLCompiledOperator> compiled_operator;
  winrt::com_ptr<ID3D12GraphicsCommandList> graphics_command_list;
  winrt::com_ptr<ID3D12CommandAllocator> command_allocator;
  winrt::com_ptr<ID3D12DescriptorHeap> heap;
  winrt::com_ptr<IDMLBindingTable> binding_table;
  winrt::com_ptr<ID3D12Resource> persistent_resource;
  winrt::com_ptr<ID3D12Resource> source_resource;
  winrt::com_ptr<ID3D12Resource> target_resource;
  OrtValue* previousInput = nullptr;
  OrtValue* previousOutput = nullptr;
};

struct DmlObjects {
  winrt::com_ptr<ID3D12Device> d3d12_device;
  winrt::com_ptr<ID3D12CommandQueue> command_queue;
  winrt::com_ptr<ID3D12CommandAllocator> command_allocator;
  winrt::com_ptr<ID3D12GraphicsCommandList> command_list;
  winrt::com_ptr<ID3D12Resource> upload_buffer;
};

namespace DmlHelpers {
DmlObjects CreateDmlObjects(const std::string& current_module_path);

DmlReusedCommandListState BuildReusableCommandList(
    IDMLDevice* dml_device,
    IDMLCompiledOperator* compiled_operator,
    ID3D12Resource* persistent_resource,
    std::optional<DML_BUFFER_BINDING> persistent_resource_binding);

void ExecuteReusableCommandList(
    DmlExecutionContext* execution_context,
    DmlReusedCommandListState& command_list_state,
    OrtAllocator& allocator,
    const OrtDmlApi* ort_dml_api,
    std::span<ID3D12Resource*> input_resources,
    std::span<const uint64_t> input_sizes,
    std::span<ID3D12Resource*> output_resources,
    std::span<const uint64_t> output_sizes,
    bool bindings_changed);

winrt::com_ptr<IDMLCompiledOperator> CreateCastOperator(
    IDMLDevice* dml_device,
    uint32_t num_elements,
    DML_TENSOR_DATA_TYPE source_data_type,
    DML_TENSOR_DATA_TYPE target_data_type);

void GetNextDispatchSize(
    uint32_t element_count,
    uint32_t num_threads,
    uint32_t& dispatch,
    uint32_t& pending_element_count);

DML_TENSOR_DATA_TYPE OrtToDmlDataType(ONNXTensorElementDataType ort_dtype);

void DmlCastInputToOutput(
    DmlExecutionContext* execution_context,
    OrtAllocator& allocator,
    OrtValue& in,
    std::unique_ptr<OrtValue>& p_out,
    IDMLDevice* dml_device,
    const OrtDmlApi* ort_dml_api,
    DmlReusedCommandListState& command_list_state);

bool IsIntelDevice(ID3D12Device* d3d12_device);
}  // namespace DmlHelpers
