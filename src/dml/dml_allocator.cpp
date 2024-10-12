// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdexcept>
#include <wil/result.h>
#include <d3dx12.h>
#include <span>
#include "dml_allocator.h"
#include "dml_provider_factory.h"
#include "../generators.h"

static const uint32_t c_minResourceSizeExponent = 16;  // 2^16 = 64KB

static uint32_t GetBucketIndexFromSize(uint64_t size) {
  assert(size != 0);

  // Each bucket is twice as large as the previous one, in ascending order
  uint32_t index = static_cast<uint32_t>(ceil(log2(size)));
  assert((1ull << index) >= size);  // This must be true unless there were some strange rounding issues

  // The smallest bucket is 2^n bytes large, where n = c_minResourceSizeExponent
  index = std::max<uint32_t>(index, c_minResourceSizeExponent);
  index -= c_minResourceSizeExponent;

  return index;
}

static uint64_t GetBucketSizeFromIndex(uint32_t index) {
  return (1ull << (index + c_minResourceSizeExponent));
}

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

void* DmlAllocator::DmlAlloc(size_t size) {
  // For some reason lotus likes requesting 0 bytes of memory
  size = std::max<size_t>(1, size);
  uint64_t bucketSize = 0;

  // Use a pooled resource if the size (post rounding, if requested) matches a bucket size
  Bucket* bucket = nullptr;

  // Find the bucket for this allocation size
  uint32_t bucketIndex = GetBucketIndexFromSize(size);

  if (static_cast<uint32_t>(m_pool.size()) <= bucketIndex) {
    // Ensure there are sufficient buckets
    m_pool.resize(bucketIndex + 1);
  }

  bucket = &m_pool[bucketIndex];
  bucketSize = GetBucketSizeFromIndex(bucketIndex);

  Microsoft::WRL::ComPtr<ID3D12Resource> resource;

  if (bucket->resources.empty()) {
    // No more resources in this bucket - allocate a new one
    auto buffer = CD3DX12_RESOURCE_DESC::Buffer(bucketSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto heap_props = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    THROW_IF_FAILED(d3d12_device_->CreateCommittedResource(
        &heap_props,
        D3D12_HEAP_FLAG_NONE,
        &buffer,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(resource.GetAddressOf())));
  } else {
    // Retrieve a resource from the bucket
    resource = std::move(bucket->resources.back());
    bucket->resources.pop_back();
  }

  assert(resource->GetDesc().Width == bucketSize);
  assert(resource != nullptr);

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
  Ort::ThrowOnError(p_dml_api_->FreeGPUAllocation(allocation));

  // If this was an initializer, don't return it to the pool
  bool dummy = false;
  uint32_t data_size = sizeof(&dummy);
  if (resource->GetPrivateData(initializer_resource_guid, &data_size, &dummy) == S_OK) {
    return;
  }

  // Free the resource to the pool if its size matches a bucket size
  uint32_t bucketIndex = GetBucketIndexFromSize(resource->GetDesc().Width);
  assert(static_cast<uint32_t>(m_pool.size()) > bucketIndex);

  // Return the resource to the bucket
  Bucket* bucket = &m_pool[bucketIndex];
  bucket->resources.push_back(std::move(resource));
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