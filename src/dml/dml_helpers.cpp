#pragma once

#include <assert.h>
#include <stdexcept>
#include <dxcore.h>
#include <dxcore_interface.h>
#include <dxgi1_6.h>
#include "dml_helpers.h"
#include "dml_adapter_info.h"

namespace DmlHelpers {

static bool IsSoftwareAdapter(IDXGIAdapter1* adapter) {
  DXGI_ADAPTER_DESC1 desc = {};
  THROW_IF_FAILED(adapter->GetDesc1(&desc));

  // See here for documentation on filtering WARP adapter:
  // https://docs.microsoft.com/en-us/windows/desktop/direct3ddxgi/d3d10-graphics-programming-guide-dxgi#new-info-about-enumerating-adapters-for-windows-8
  const bool is_basic_render_driver_vendor_id = desc.VendorId == static_cast<UINT>(VendorID::Microsoft);
  const bool is_basic_render_driver_device_id = desc.DeviceId == 0x8c;
  return desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE || (is_basic_render_driver_vendor_id && is_basic_render_driver_device_id);
};

static std::vector<ComPtr<IDXGIAdapter1>> EnumerateAdapters(PLUID device_luid = nullptr) {
  ComPtr<IDXGIFactory4> dxgi_factory;
  THROW_IF_FAILED(CreateDXGIFactory(IID_PPV_ARGS(&dxgi_factory)));

  std::vector<ComPtr<IDXGIAdapter1>> adapter_infos;

  ComPtr<IDXGIFactory6> dxgi_factory6;
  if (SUCCEEDED(dxgi_factory.As(&dxgi_factory6)) && !device_luid) {
    // Enumerate adapters by performance. This only works in Windows 10 Version 1803 and later.
    ComPtr<IDXGIAdapter1> adapter;
    for (uint32_t adapter_index = 0;
         dxgi_factory6->EnumAdapterByGpuPreference(
             adapter_index,
             DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
             IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND;
         adapter_index++) {
      // Since we enumerate by performance, we can ignore everything that comes after the first software adapter, which includes the IDD
      // adapters. This is necessary for now because IDD (e.g. remote desktop) adapters don't have the DXGI_ADAPTER_FLAG_SOFTWARE flag,
      // even though they run on software.
      if (IsSoftwareAdapter(adapter.Get())) {
        break;
      }

      // Make sure that we are able to create the device
      ComPtr<ID3D12Device> d3d12_device;
      THROW_IF_FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d12_device)));

      if (d3d12_device) {
        adapter_infos.emplace_back(std::move(adapter));
      }
    }
  } else {
    // Enumerate adapters without ordering.
    ComPtr<IDXGIAdapter1> adapter;
    for (uint32_t adapter_index = 0; dxgi_factory->EnumAdapters1(adapter_index, &adapter) != DXGI_ERROR_NOT_FOUND; adapter_index++) {
      // We can't assume the ordering of hardware and software adapters, so keep looping. This path should only execute on Windows 10
      // version 1709 or earlier; IDD (e.g. remote desktop) adapters do not exist when taking this code path.
      if (IsSoftwareAdapter(adapter.Get())) {
        continue;
      }

      // Make sure that we are able to create the device
      ComPtr<ID3D12Device> d3d12_device;
      THROW_IF_FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d12_device)));

      if (d3d12_device && device_luid) {
        DXGI_ADAPTER_DESC1 description = {};
        THROW_IF_FAILED(adapter->GetDesc1(&description));

        // Check if current adapter LUID is the same as the target one
        if (device_luid->HighPart == description.AdapterLuid.HighPart && device_luid->LowPart == description.AdapterLuid.LowPart) {
          adapter_infos.emplace_back(std::move(adapter));
          break;
        }
      } else if (d3d12_device) {
        adapter_infos.emplace_back(std::move(adapter));
      }
    }
  }

  return adapter_infos;
}

static ComPtr<IDXGIAdapter1> CreateAdapter(PLUID device_luid = nullptr) {
  auto filtered_adapters = EnumerateAdapters(device_luid);
  if (filtered_adapters.empty()) {
    throw std::runtime_error("No adapter is available for DML.");
  }
  return filtered_adapters.front();
}

DmlObjects CreateDmlObjects(const std::string& current_module_path, PLUID device_luid) {
  D3D12_COMMAND_QUEUE_DESC command_queue_description = {
      D3D12_COMMAND_LIST_TYPE_COMPUTE,
      0,
      D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT,
      0,
  };

  DmlObjects dml_objects;

  auto adapter = CreateAdapter(device_luid);
  ComPtr<ID3D12SDKConfiguration1> d3d12_sdk_config;
  ComPtr<ID3D12DeviceFactory> d3d12_factory;

  // Get the version from https://devblogs.microsoft.com/directx/directx12agility/. We are currently using 1.614.0.
  constexpr uint32_t agility_sdk_version = 614;

  if (SUCCEEDED(D3D12GetInterface(CLSID_D3D12SDKConfiguration, IID_PPV_ARGS(&d3d12_sdk_config))) &&
      SUCCEEDED(d3d12_sdk_config->CreateDeviceFactory(agility_sdk_version, current_module_path.c_str(), IID_PPV_ARGS(&d3d12_factory)))) {
    THROW_IF_FAILED(d3d12_factory->CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dml_objects.d3d12_device)));
  } else {
    printf("Warning: Unable to create a device from version 1.614.0 of the DirectX 12 Agility SDK. You can still use this library, but some scenarios may not work.\n");
    printf("The given module path: %s", current_module_path.c_str());
    THROW_IF_FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dml_objects.d3d12_device)));
  }

  THROW_IF_FAILED(dml_objects.d3d12_device->CreateCommandQueue(&command_queue_description, IID_PPV_ARGS(&dml_objects.command_queue)));
  THROW_IF_FAILED(dml_objects.d3d12_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&dml_objects.command_allocator)));
  THROW_IF_FAILED(dml_objects.d3d12_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, dml_objects.command_allocator.Get(), nullptr, IID_PPV_ARGS(&dml_objects.command_list)));
  return dml_objects;
}

DmlReusedCommandListState BuildReusableCommandList(
    IDMLDevice* dml_device,
    IDMLCompiledOperator* compiled_operator,
    ID3D12Resource* persistent_resource,
    std::optional<DML_BUFFER_BINDING> persistent_resource_binding) {
  DmlReusedCommandListState command_list_state{};

  DML_BINDING_PROPERTIES exec_binding_props = compiled_operator->GetBindingProperties();

  D3D12_DESCRIPTOR_HEAP_DESC desc = {};
  desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  desc.NumDescriptors = exec_binding_props.RequiredDescriptorCount;
  desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;

  ComPtr<ID3D12Device> d3d_device;
  THROW_IF_FAILED(dml_device->GetParentDevice(IID_PPV_ARGS(&d3d_device)));

  THROW_IF_FAILED(d3d_device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(command_list_state.heap.ReleaseAndGetAddressOf())));

  // Create a binding table for execution.
  DML_BINDING_TABLE_DESC binding_table_desc = {};
  binding_table_desc.Dispatchable = compiled_operator;
  binding_table_desc.CPUDescriptorHandle = command_list_state.heap->GetCPUDescriptorHandleForHeapStart();
  binding_table_desc.GPUDescriptorHandle = command_list_state.heap->GetGPUDescriptorHandleForHeapStart();
  binding_table_desc.SizeInDescriptors = exec_binding_props.RequiredDescriptorCount;

  THROW_IF_FAILED(dml_device->CreateBindingTable(&binding_table_desc, IID_PPV_ARGS(&command_list_state.binding_table)));

  THROW_IF_FAILED(d3d_device->CreateCommandAllocator(
      D3D12_COMMAND_LIST_TYPE_COMPUTE,
      IID_PPV_ARGS(command_list_state.command_allocator.ReleaseAndGetAddressOf())));

  THROW_IF_FAILED(d3d_device->CreateCommandList(
      0,
      D3D12_COMMAND_LIST_TYPE_COMPUTE,
      command_list_state.command_allocator.Get(),
      nullptr,
      IID_PPV_ARGS(command_list_state.graphics_command_list.ReleaseAndGetAddressOf())));

  if (persistent_resource) {
    DML_BINDING_DESC persistent_resource_binding_desc = {DML_BINDING_TYPE_BUFFER, persistent_resource_binding ? &*persistent_resource_binding : nullptr};
    command_list_state.binding_table->BindPersistentResource(&persistent_resource_binding_desc);
    command_list_state.persistent_resource = persistent_resource;
  }

  ID3D12DescriptorHeap* descriptor_heaps[] = {command_list_state.heap.Get()};
  command_list_state.graphics_command_list->SetDescriptorHeaps(ARRAYSIZE(descriptor_heaps), descriptor_heaps);

  ComPtr<IDMLCommandRecorder> recorder;
  THROW_IF_FAILED(dml_device->CreateCommandRecorder(IID_PPV_ARGS(recorder.GetAddressOf())));

  recorder->RecordDispatch(command_list_state.graphics_command_list.Get(), compiled_operator, command_list_state.binding_table.Get());
  command_list_state.compiled_operator = compiled_operator;

  THROW_IF_FAILED(command_list_state.graphics_command_list->Close());

  return command_list_state;
}

void ExecuteReusableCommandList(
    DmlExecutionContext* execution_context,
    DmlReusedCommandListState& command_list_state,
    OrtAllocator& allocator,
    const OrtDmlApi* ort_dml_api,
    std::span<ID3D12Resource*> input_resources,
    std::span<const uint64_t> input_sizes,
    std::span<ID3D12Resource*> output_resources,
    std::span<const uint64_t> output_sizes,
    bool bindings_changed) {
  assert(input_resources.size() == input_sizes.size());
  assert(output_resources.size() == output_sizes.size());

  DML_BINDING_PROPERTIES exec_binding_props = command_list_state.compiled_operator->GetBindingProperties();

  std::vector<DML_BUFFER_BINDING> input_bindings(input_resources.size());
  std::vector<DML_BINDING_DESC> input_binding_descs(output_resources.size());

  std::vector<DML_BUFFER_BINDING> output_bindings(output_resources.size());
  std::vector<DML_BINDING_DESC> output_binding_descs(output_resources.size());

  if (bindings_changed) {
    // Bind the inputs
    for (uint32_t i = 0; i < input_bindings.size(); ++i) {
      input_bindings[i].Buffer = input_resources[i];
      input_bindings[i].SizeInBytes = input_sizes[i];
      input_binding_descs[i] = {DML_BINDING_TYPE_BUFFER, &input_bindings[i]};
    }

    command_list_state.binding_table->BindInputs(static_cast<uint32_t>(input_binding_descs.size()), input_binding_descs.data());

    // Bind the outputs
    for (uint32_t i = 0; i < output_bindings.size(); ++i) {
      output_bindings[i].Buffer = output_resources[i];
      output_bindings[i].SizeInBytes = output_sizes[i];
      output_binding_descs[i] = {DML_BINDING_TYPE_BUFFER, &output_bindings[i]};
    }

    command_list_state.binding_table->BindOutputs(static_cast<uint32_t>(output_binding_descs.size()), output_binding_descs.data());

    // Create the temporary resource
    if (exec_binding_props.TemporaryResourceSize > 0) {
      ComPtr<ID3D12Resource> temporary_resource;
      std::array<int64_t, 1> persistent_resource_shape = {static_cast<int64_t>(exec_binding_props.TemporaryResourceSize)};
      auto persistent_tensor = OrtValue::CreateTensor(allocator, persistent_resource_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
      Ort::ThrowOnError(ort_dml_api->GetD3D12ResourceFromAllocation(&allocator, persistent_tensor->GetTensorMutableRawData(), &temporary_resource));
    }
  }

  // Execute the command list and if it succeeds, update the fence value at which this command may be
  // re-used.
  ComPtr<ID3D12Fence> fence;
  uint64_t completion_value;
  execution_context->ExecuteCommandList(command_list_state.graphics_command_list.Get(), fence.GetAddressOf(), &completion_value);
}

// Copied from https://learn.microsoft.com/en-us/windows/ai/directml/dml-helper-functions#dmlcalcbuffertensorsize
static UINT64 DMLCalcBufferTensorSize(
    DML_TENSOR_DATA_TYPE dataType,
    UINT dimensionCount,
    _In_reads_(dimensionCount) const UINT* sizes,
    _In_reads_opt_(dimensionCount) const UINT* strides) {
  UINT elementSizeInBytes = 0;
  switch (dataType) {
    case DML_TENSOR_DATA_TYPE_FLOAT32:
    case DML_TENSOR_DATA_TYPE_UINT32:
    case DML_TENSOR_DATA_TYPE_INT32:
      elementSizeInBytes = 4;
      break;

    case DML_TENSOR_DATA_TYPE_FLOAT16:
    case DML_TENSOR_DATA_TYPE_UINT16:
    case DML_TENSOR_DATA_TYPE_INT16:
      elementSizeInBytes = 2;
      break;

    case DML_TENSOR_DATA_TYPE_UINT8:
    case DML_TENSOR_DATA_TYPE_INT8:
      elementSizeInBytes = 1;
      break;

    case DML_TENSOR_DATA_TYPE_FLOAT64:
    case DML_TENSOR_DATA_TYPE_UINT64:
    case DML_TENSOR_DATA_TYPE_INT64:
      elementSizeInBytes = 8;
      break;

    default:
      return 0;  // Invalid data type
  }

  UINT64 minimumImpliedSizeInBytes = 0;
  if (!strides) {
    minimumImpliedSizeInBytes = sizes[0];
    for (UINT i = 1; i < dimensionCount; ++i) {
      minimumImpliedSizeInBytes *= sizes[i];
    }
    minimumImpliedSizeInBytes *= elementSizeInBytes;
  } else {
    UINT indexOfLastElement = 0;
    for (UINT i = 0; i < dimensionCount; ++i) {
      indexOfLastElement += (sizes[i] - 1) * strides[i];
    }

    minimumImpliedSizeInBytes = (static_cast<UINT64>(indexOfLastElement) + 1) * elementSizeInBytes;
  }

  // Round up to the nearest 4 bytes.
  minimumImpliedSizeInBytes = (minimumImpliedSizeInBytes + 3) & ~3ull;

  return minimumImpliedSizeInBytes;
}

ComPtr<IDMLCompiledOperator> CreateCastOperator(
    IDMLDevice* dml_device,
    uint32_t num_elements,
    DML_TENSOR_DATA_TYPE source_data_type,
    DML_TENSOR_DATA_TYPE target_data_type) {
  // Create the input tensor desc
  DML_BUFFER_TENSOR_DESC input_buffer_desc{};
  input_buffer_desc.Sizes = &num_elements;
  input_buffer_desc.DimensionCount = 1;
  input_buffer_desc.DataType = source_data_type;
  input_buffer_desc.TotalTensorSizeInBytes = DMLCalcBufferTensorSize(source_data_type, 1, &num_elements, NULL);
  DML_TENSOR_DESC input_tensor_desc = {DML_TENSOR_TYPE_BUFFER, &input_buffer_desc};

  // Create the output tensor desc
  DML_BUFFER_TENSOR_DESC output_buffer_desc{};
  output_buffer_desc.Sizes = &num_elements;
  output_buffer_desc.DimensionCount = 1;
  output_buffer_desc.DataType = target_data_type;
  output_buffer_desc.TotalTensorSizeInBytes = DMLCalcBufferTensorSize(target_data_type, 1, &num_elements, NULL);
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

void GetNextDispatchSize(
    uint32_t element_count,
    uint32_t num_threads,
    uint32_t& dispatch,
    uint32_t& pending_element_count) {
  // Max threads per workgroup is 2^10 (1024). Max dispatch per dimension is 2^16. Taken together, we can dispatch a maximum of
  // 2^26 (268,435,456) threads along a single dimension. This should suffice for a majority of the workload. Therefore, even
  // though it is possible to dispatch up to (2^16)^3 workgroups simultaneously, we stick to the simpler 1D dispatch alternative.
  assert(num_threads <= D3D12_CS_THREAD_GROUP_MAX_THREADS_PER_GROUP);

  const uint32_t max_threads_per_dispatch = num_threads * D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;

  // Compute max dispatchable elements
  const uint32_t available_thread_count = std::min(element_count, max_threads_per_dispatch);

  // Compute required thread group count
  uint32_t workgroup_count_1d = (available_thread_count + num_threads - 1) / num_threads;

  // Compute min dispatch size
  dispatch = workgroup_count_1d;

  // With the dispatch size computed, compute the dispatched element count
  const uint32_t dispatched_element_count = workgroup_count_1d * num_threads;

  // Update the pending element count
  pending_element_count = (dispatched_element_count < element_count) ? element_count - dispatched_element_count : 0;
}

DML_TENSOR_DATA_TYPE OrtToDmlDataType(ONNXTensorElementDataType ort_dtype) {
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

void DmlCastInputToOutput(
    DmlExecutionContext* execution_context,
    OrtAllocator& allocator,
    OrtValue& in,
    std::unique_ptr<OrtValue>& p_out,
    IDMLDevice* dml_device,
    const OrtDmlApi* ort_dml_api,
    DmlReusedCommandListState& command_list_state) {
  auto shape_info = in.GetTensorTypeAndShapeInfo();
  auto shape = shape_info->GetShape();

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
  auto dml_from_type = DmlHelpers::OrtToDmlDataType(in.GetTensorTypeAndShapeInfo()->GetElementType());
  auto dml_to_type = DmlHelpers::OrtToDmlDataType(p_out->GetTensorTypeAndShapeInfo()->GetElementType());

  bool rebind = command_list_state.previousInput != &in || command_list_state.previousOutput != p_out.get();

  // If the sizes change, we need to recompile the operator and rebuild the command lists. It should only happen
  // once after the very first iteration.
  if (rebind) {
    auto compiled_cast_operator = DmlHelpers::CreateCastOperator(dml_device, element_count, dml_from_type, dml_to_type);

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

    DML_BINDING_DESC input_array_binding_desc = DML_BINDING_DESC{DML_BINDING_TYPE_NONE, nullptr};
    execution_context->InitializeOperator(compiled_cast_operator.Get(), persistent_resource_bindingDesc, input_array_binding_desc);
    command_list_state = DmlHelpers::BuildReusableCommandList(dml_device, compiled_cast_operator.Get(), persistent_resource.Get(), persistent_resource_binding);
    command_list_state.previousInput = &in;
    command_list_state.previousOutput = p_out.get();
  }

  ComPtr<ID3D12Resource> source_resource;
  Ort::ThrowOnError(ort_dml_api->GetD3D12ResourceFromAllocation(&allocator, in.GetTensorMutableData<uint8_t>(), &source_resource));

  ComPtr<ID3D12Resource> target_resource;
  Ort::ThrowOnError(ort_dml_api->GetD3D12ResourceFromAllocation(&allocator, p_out->GetTensorMutableData<uint8_t>(), &target_resource));

  std::array<ID3D12Resource*, 1> input_resources = {source_resource.Get()};
  std::array<uint64_t, 1> input_sizes = {DMLCalcBufferTensorSize(dml_from_type, 1, (uint32_t*)&element_count, NULL)};

  std::array<ID3D12Resource*, 1> output_resources = {target_resource.Get()};
  std::array<uint64_t, 1> output_sizes = {DMLCalcBufferTensorSize(dml_to_type, 1, (uint32_t*)&element_count, NULL)};

  // Make sure the source and target allocations are kept alive until the operation is done
  command_list_state.source_resource = std::move(source_resource);
  command_list_state.target_resource = std::move(target_resource);

  DmlHelpers::ExecuteReusableCommandList(
      execution_context,
      command_list_state,
      allocator,
      ort_dml_api,
      input_resources,
      input_sizes,
      output_resources,
      output_sizes,
      rebind);
}

bool IsIntelDevice(ID3D12Device* d3d12_device) {
  return AdapterInfo(d3d12_device).IsIntel();
}

}  // namespace DmlHelpers
