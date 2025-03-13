// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "../search.h"
#include "../models/utils.h"
#include "../cpu/interface.h"
#include "interface.h"
#include <cstdarg>

#include <wil/wrl.h>
#include "dml_provider_factory.h"
#include "../dml/dml_helpers.h"
#include "../dml/dml_execution_context.h"
#include "../dml/dml_pooled_upload_heap.h"
#include "../dml/dml_readback_heap.h"

std::string CurrentModulePath();

namespace Generators {
namespace Dml {  // If this was in a shared library it wouldn't need to be in its own namespace

Ort::Allocator* ort_allocator_{};
const char* device_label = "dml";

wil::unique_hmodule smart_directml_dll_;
DmlObjects dml_objects_;
const OrtDmlApi* dml_api_{};
std::unique_ptr<DmlPooledUploadHeap> dml_pooled_upload_heap_;
std::unique_ptr<DmlExecutionContext> dml_execution_context_;
std::unique_ptr<DmlReadbackHeap> dml_readback_heap_;
ComPtr<IDMLDevice> dml_device_;

struct GpuMemory final : DeviceBuffer {
  GpuMemory(size_t size) : owned_{true} {
    size_in_bytes_ = size;
    p_device_ = static_cast<uint8_t*>(ort_allocator_->Alloc(size_in_bytes_));
    Ort::ThrowOnError(dml_api_->GetD3D12ResourceFromAllocation(ort_allocator_, p_device_, &gpu_resource_));
  }

  GpuMemory(void* p, size_t size) : owned_{false} {
    size_in_bytes_ = size;
    p_device_ = static_cast<uint8_t*>(p);
    Ort::ThrowOnError(dml_api_->GetD3D12ResourceFromAllocation(ort_allocator_, p_device_, &gpu_resource_));
  }

  ~GpuMemory() override {
    if (owned_)
      ort_allocator_->Free(p_device_);
    if (p_cpu_)
      free(p_cpu_);
  }

  const char* GetType() const override { return device_label; }

  void AllocateCpu() override {
    if (!p_cpu_)
      p_cpu_ = static_cast<uint8_t*>(malloc(size_in_bytes_));
  }

  void CopyDeviceToCpu() override {
    AllocateCpu();
    dml_readback_heap_->ReadbackFromGpu(std::span(p_cpu_, size_in_bytes_), gpu_resource_.Get(), 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  }

  void CopyCpuToDevice() override {
    assert(p_cpu_);
    auto source = std::span(p_cpu_, size_in_bytes_);
    dml_pooled_upload_heap_->BeginUploadToGpu(gpu_resource_.Get(), 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, source);
  }

  void CopyFrom(size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) override {
    if (source.GetType() == device_label) {
      auto& source_gpu = dynamic_cast<GpuMemory&>(source);
      dml_execution_context_->CopyBufferRegion(
          gpu_resource_.Get(),
          begin_dest,
          D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
          source_gpu.gpu_resource_.Get(),
          begin_source,
          D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
          size_in_bytes);
    } else
      CopyThroughCpu(*this, begin_dest, source, begin_source, size_in_bytes);
  }

  void Zero() override {
    // TODO: Implement a zeroing that runs directly on DML vs going through CPU
    AllocateCpu();
    memset(p_cpu_, 0, size_in_bytes_);
    CopyCpuToDevice();
  }

  ComPtr<ID3D12Resource> gpu_resource_;
  bool owned_;  // If we own the memory, we delete it on destruction
};

struct InterfaceImpl : DeviceInterface {
  InterfaceImpl(LUID* p_device_luid) {
    Ort::ThrowOnError(Ort::api->GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dml_api_)));
    if (!dml_api_) {
      throw std::runtime_error("Unexpected nullptr getting OrtDmlApi");
    }

    dml_objects_ = DmlHelpers::CreateDmlObjects(CurrentModulePath(), p_device_luid);

    constexpr auto directml_dll = "DirectML.dll";
    smart_directml_dll_ = wil::unique_hmodule{LoadLibraryEx(directml_dll, nullptr, 0)};
    if (!smart_directml_dll_)
      throw std::runtime_error("DirectML.dll not found");

    auto dml_create_device1_fn = reinterpret_cast<decltype(&DMLCreateDevice1)>(GetProcAddress(smart_directml_dll_.get(), "DMLCreateDevice1"));
    THROW_LAST_ERROR_IF(!dml_create_device1_fn);
    THROW_IF_FAILED(dml_create_device1_fn(dml_objects_.d3d12_device.Get(), DML_CREATE_DEVICE_FLAG_NONE, DML_FEATURE_LEVEL_5_0, IID_PPV_ARGS(&dml_device_)));

    Ort::ThrowOnError(Ort::api->GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dml_api_)));
  }

  DeviceType GetType() const override { return DeviceType::DML; }

  void InitOrt(const OrtApi& api, Ort::Allocator& allocator) override {
    Ort::api = &api;
    assert(!ort_allocator_);
    ort_allocator_ = &allocator;

    dml_execution_context_ = std::make_unique<DmlExecutionContext>(
        dml_objects_.d3d12_device.Get(),
        dml_device_.Get(),
        dml_objects_.command_queue.Get(),
        *ort_allocator_,
        dml_api_);

    dml_pooled_upload_heap_ = std::make_unique<DmlPooledUploadHeap>(dml_objects_.d3d12_device.Get(), dml_execution_context_.get());
    dml_readback_heap_ = std::make_unique<DmlReadbackHeap>(dml_objects_.d3d12_device.Get(), dml_execution_context_.get());
  }

  Ort::Allocator& GetAllocator() override {
    return *ort_allocator_;
  }

  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return std::make_shared<GpuMemory>(size);
  }

  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* p, size_t size) override {
    return std::make_shared<GpuMemory>(p, size);
  }

  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override {
    return GetCpuInterface()->CreateGreedy(params);
  }

  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override {
    return GetCpuInterface()->CreateBeam(params);
  }

#if 0
  void UpdatePositionIDs() {
    ComPtr<ID3D12Resource> target_resource;
    Ort::ThrowOnError(model_.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(model_.allocator_device_, position_ids_->GetTensorMutableRawData(), &target_resource));

    dml_update_position_ids_kernel_ = DmlIncrementValuesKernel(
        model_.GetD3D12Device(),
        model_.GetDmlExecutionContext(),
        static_cast<uint32_t>(position_ids_shape_[0]),
        type_,
        target_resource.Get());

    // Execute the cached command list
    ComPtr<ID3D12Fence> fence;
    uint64_t completion_value;
    model_.GetDmlExecutionContext()->ExecuteCommandList(dml_update_position_ids_kernel_->GetCommandList(), &fence, &completion_value);
  }

  void UpdateAttentionMask(int total_length) {
    ComPtr<ID3D12Resource> attention_mask_resource;
    Ort::ThrowOnError(model_.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(model_.allocator_device_, attention_mask_->GetTensorMutableRawData(), &attention_mask_resource));
    ComPtr<ID3D12Resource> attention_mask_next_resource;
    Ort::ThrowOnError(model_.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(model_.allocator_device_, attention_mask_next_->GetTensorMutableRawData(), &attention_mask_next_resource));
    if (is_first_mask_update_) {
      dml_update_mask_kernel_ = DmlUpdateMaskKernel(
          model_.GetD3D12Device(),
          model_.GetDmlExecutionContext(),
          static_cast<uint32_t>(attention_mask_shape_[0]),
          static_cast<uint32_t>(attention_mask_shape_[1]),
          type_,
          total_length,
          attention_mask_resource.Get(),
          attention_mask_next_resource.Get());
      is_second_mask_update_ = true;
    } else if (is_second_mask_update_) {
      dml_update_mask_kernel_ = DmlUpdateMaskKernel(
          model_.GetD3D12Device(),
          model_.GetDmlExecutionContext(),
          static_cast<uint32_t>(attention_mask_shape_[0]),
          static_cast<uint32_t>(attention_mask_shape_[1]),
          type_,
          1,
          attention_mask_resource.Get(),
          attention_mask_next_resource.Get());
      is_second_mask_update_ = false;
    }
    ComPtr<ID3D12Fence> fence;
    uint64_t completion_value;
    model_.GetDmlExecutionContext()->ExecuteCommandList(dml_update_mask_kernel_->GetCommandList(), &fence, &completion_value);
  }
#endif

  void Synchronize() override {}
};

}  // namespace Dml

std::unique_ptr<Dml::InterfaceImpl> g_dml_device;

void InitDmlInterface(LUID* p_device_luid) {
  if (!g_dml_device)
    g_dml_device = std::make_unique<Dml::InterfaceImpl>(p_device_luid);
}

void SetDmlProvider(OrtSessionOptions& session_options) {
  Ort::ThrowOnError(Dml::dml_api_->SessionOptionsAppendExecutionProvider_DML1(&session_options, Dml::dml_device_.Get(), Dml::dml_objects_.command_queue.Get()));
}

DeviceInterface* GetDmlInterface() {
  return g_dml_device.get();
}

}  // namespace Generators
