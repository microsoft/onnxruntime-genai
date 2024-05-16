// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdexcept>
#include <wil/result.h>
#include "dml_allocator.h"
#include "dml_execution_context.h"

DmlAllocator::DmlAllocator(const OrtDmlApi* p_dml_api, ID3D12Device* d3d12_device)
    : p_dml_api_(p_dml_api),
      d3d12_device_(d3d12_device) {
  version = ORT_API_VERSION;
  OrtAllocator::Alloc = AllocImpl;
  OrtAllocator::Free = FreeImpl;
  OrtAllocator::Info = InfoImpl;

  Ort::ThrowOnError(Ort::api->CreateMemoryInfo("DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault, &memory_info_));
}

DmlAllocator::~DmlAllocator() {
  Ort::api->ReleaseMemoryInfo(memory_info_);
}

void* DmlAllocator::DmlAlloc(size_t size_in_bytes) {
  ComPtr<ID3D12Resource> resource;
  auto buffer = CD3DX12_RESOURCE_DESC::Buffer(size_in_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  auto heap_props = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
  THROW_IF_FAILED(d3d12_device_->CreateCommittedResource(
      &heap_props,
      D3D12_HEAP_FLAG_NONE,
      &buffer,
      D3D12_RESOURCE_STATE_COMMON,
      nullptr,
      IID_PPV_ARGS(resource.GetAddressOf())));

  void* allocation;
  Ort::ThrowOnError(p_dml_api_->CreateGPUAllocationFromD3DResource(resource.Get(), &allocation));

  if (outstanding_allocs_.size() == 879) {
    printf("FOUND!\n");
  }

  outstanding_allocs_.emplace(allocation, static_cast<int>(outstanding_allocs_.size()));

  resource.Detach();

  return allocation;
}

void DmlAllocator::DmlFree(void* allocation) {
  if (allocation) {
    // Extend the lifetime of the D3D12 resource until the workload is done executing
    ComPtr<ID3D12Resource> resource;
    Ort::ThrowOnError(p_dml_api_->GetD3D12ResourceFromAllocation(allocator_wrapper_, allocation, &resource));

    dml_execution_context_->QueueReference(resource.Get());
    resource->Release();

    // We free the allocation itself, even though the D3D12 resource may survive until the GPU is done executing
    // Ort::ThrowOnError(p_dml_api_->FreeGPUAllocation(allocation));

    outstanding_allocs_.erase(allocation);

    if (outstanding_allocs_.size() < 70) {
      printf("%d Remaining items\n", static_cast<int>(outstanding_allocs_.size()));
      PrintOutstandingAllocs();
    }
  }
}

OrtMemoryInfo* DmlAllocator::DmlInfo() const {
  return memory_info_;
}

void* ORT_API_CALL DmlAllocator::AllocImpl(struct OrtAllocator* this_, size_t size) {
  return static_cast<DmlAllocator*>(this_)->DmlAlloc(size);
}

void ORT_API_CALL DmlAllocator::FreeImpl(struct OrtAllocator* this_, void* p) {
  return static_cast<DmlAllocator*>(this_)->DmlFree(p);
}

const OrtMemoryInfo* ORT_API_CALL DmlAllocator::InfoImpl(const struct OrtAllocator* this_) {
  return static_cast<const DmlAllocator*>(this_)->DmlInfo();
}

void DmlAllocator::PrintOutstandingAllocs() {
  for (auto& kvp : outstanding_allocs_) {
    ComPtr<ID3D12Resource> resource;
    Ort::ThrowOnError(p_dml_api_->GetD3D12ResourceFromAllocation(allocator_wrapper_, kvp.first, &resource));

    wchar_t name[128] = {};
    UINT size = sizeof(name);
    resource->GetPrivateData(WKPDID_D3DDebugObjectNameW, &size, name);

    printf("%ls\n", name);
  }
}