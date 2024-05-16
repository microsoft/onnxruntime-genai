// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <wrl/client.h>
#include <unordered_map>
#include <d3d12.h>
#include "onnxruntime_c_api.h"
#include "dml_execution_context.h"

struct OrtDmlApi;
class DmlExecutionContext;

struct DmlAllocator : public OrtAllocator {
  DmlAllocator(const OrtDmlApi* p_dml_api, ID3D12Device* d3d12_device);
  ~DmlAllocator();

  void SetExecutionContext(DmlExecutionContext* dml_execution_context) { dml_execution_context_ = dml_execution_context; }
  void SetWrapper(OrtAllocator* allocator_wrapper) { allocator_wrapper_ = allocator_wrapper; }
  void* DmlAlloc(size_t size_in_bytes);
  void DmlFree(void* allocation);
  OrtMemoryInfo* DmlInfo() const;
  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* this_, size_t size);
  static void ORT_API_CALL FreeImpl(struct OrtAllocator* this_, void* p);
  static const struct OrtMemoryInfo* ORT_API_CALL InfoImpl(const struct OrtAllocator* this_);
  void Destroy();

 private:
  Microsoft::WRL::ComPtr<ID3D12Device> d3d12_device_;
  const OrtDmlApi* p_dml_api_{};
  OrtMemoryInfo* memory_info_{};
  DmlExecutionContext* dml_execution_context_{};
  OrtAllocator* allocator_wrapper_{};
  std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> resources_;
};