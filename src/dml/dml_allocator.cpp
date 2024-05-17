// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdexcept>
#include <wil/result.h>
#include <d3dx12.h>
#include <span>
#include "dml_allocator.h"
#include "dml_provider_factory.h"
#include "../generators.h"

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
  Microsoft::WRL::ComPtr<ID3D12Resource> resource;
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

  resources_.push_back(std::move(resource));

  return allocation;
}

void DmlAllocator::DmlFree(void* allocation) {
  if (allocation) {
    // We free the allocation itself, even though the D3D12 resource may survive until the GPU is done executing
    Ort::ThrowOnError(p_dml_api_->FreeGPUAllocation(allocation));
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