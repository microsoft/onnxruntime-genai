// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdexcept>
#include <wil/result.h>
#include <d3dx12.h>
#include <span>
#include "dml_allocator.h"
#include "dml_provider_factory.h"
#include "../generators.h"

DmlAllocator::DmlAllocator(const OrtDmlApi* p_dml_api, ID3D12Device* d3d12_device, OrtMemoryInfo* memory_info)
    : p_dml_api_(p_dml_api),
      d3d12_device_(d3d12_device),
      memory_info_(memory_info) {
  version = ORT_API_VERSION;
  OrtAllocator::Alloc = AllocImpl;
  OrtAllocator::Free = FreeImpl;
  OrtAllocator::Info = InfoImpl;
  OrtAllocator::Reserve = ReserveImpl;
}

void* DmlAllocator::DmlAlloc(size_t size_in_bytes) {
  size_t rounded_size_in_bytes = (size_in_bytes + 3) & ~3;

  Microsoft::WRL::ComPtr<ID3D12Resource> resource;
  auto buffer = CD3DX12_RESOURCE_DESC::Buffer(rounded_size_in_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
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

  return allocation;
}

constexpr GUID initializer_resource_guid = {0x86b04a2b, 0x14d1, 0x4a4b, {0xb9, 0xcf, 0xf0, 0x01, 0xa3, 0x9a, 0xa1, 0x6d}};

void* DmlAllocator::DmlReserve(size_t size_in_bytes) {
  size_t rounded_size_in_bytes = (size_in_bytes + 3) & ~3;

  Microsoft::WRL::ComPtr<ID3D12Resource> resource;
  auto buffer = CD3DX12_RESOURCE_DESC::Buffer(rounded_size_in_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  auto heap_props = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
  THROW_IF_FAILED(d3d12_device_->CreateCommittedResource(
      &heap_props,
      D3D12_HEAP_FLAG_NONE,
      &buffer,
      D3D12_RESOURCE_STATE_COMMON,
      nullptr,
      IID_PPV_ARGS(resource.GetAddressOf())));

  // Tag this resource as an initializer, which means that we don't need to pool it when freeing it
  bool dummy = false;
  resource->SetPrivateData(initializer_resource_guid, sizeof(&dummy), &dummy);

  void* allocation;
  Ort::ThrowOnError(p_dml_api_->CreateGPUAllocationFromD3DResource(resource.Get(), &allocation));

  return allocation;
}

void DmlAllocator::DmlFree(void* allocation) {
  if (!allocation) {
    return;
  }

  Microsoft::WRL::ComPtr<ID3D12Resource> resource;
  Ort::ThrowOnError(p_dml_api_->GetD3D12ResourceFromAllocation(this, allocation, resource.GetAddressOf()));

  bool dummy = false;
  uint32_t data_size = sizeof(&dummy);
  if (resource->GetPrivateData(initializer_resource_guid, &data_size, &dummy) == S_OK) {
    Ort::ThrowOnError(p_dml_api_->FreeGPUAllocation(allocation));
  }

  /*
  // if (allocation) {
  //  We free the allocation itself, even though the D3D12 resource may survive until the GPU is done executing
  Microsoft::WRL::ComPtr<ID3D12Resource> resource;
  Ort::ThrowOnError(p_dml_api_->GetD3D12ResourceFromAllocation(this, allocation, resource.GetAddressOf()));

  resource->Release();
  Ort::ThrowOnError(p_dml_api_->FreeGPUAllocation(allocation));
  //}
  */
}

OrtMemoryInfo* DmlAllocator::DmlInfo() const {
  return memory_info_;
}

void* ORT_API_CALL DmlAllocator::AllocImpl(struct OrtAllocator* this_, size_t size) {
  return static_cast<DmlAllocator*>(this_)->DmlAlloc(size);
}

void* ORT_API_CALL DmlAllocator::ReserveImpl(struct OrtAllocator* this_, size_t size) {
  return static_cast<DmlAllocator*>(this_)->DmlReserve(size);
}

void ORT_API_CALL DmlAllocator::FreeImpl(struct OrtAllocator* this_, void* p) {
  return static_cast<DmlAllocator*>(this_)->DmlFree(p);
}

const OrtMemoryInfo* ORT_API_CALL DmlAllocator::InfoImpl(const struct OrtAllocator* this_) {
  return static_cast<const DmlAllocator*>(this_)->DmlInfo();
}