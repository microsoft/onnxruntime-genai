// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <wrl/client.h>
#include <unordered_map>
#include <d3d12.h>
#include "onnxruntime_c_api.h"

enum class DmlAllocatorState {
  AllocatingWeights,
  AllocatingPromptActivations,
  AllocatingIncrementalActivations,
  ReusingPromptActivations,
};

struct OrtDmlApi;

struct DmlAllocator : public OrtAllocator {
  DmlAllocator(const OrtDmlApi* p_dml_api, ID3D12Device* d3d12_device, OrtMemoryInfo* memory_info);

  void* DmlAlloc(size_t size_in_bytes);
  void DmlFree(void* allocation);
  OrtMemoryInfo* DmlInfo() const;
  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* this_, size_t size);
  static void ORT_API_CALL FreeImpl(struct OrtAllocator* this_, void* p);
  static const struct OrtMemoryInfo* ORT_API_CALL InfoImpl(const struct OrtAllocator* this_);
  void SetState(DmlAllocatorState new_state);

 private:
  const OrtDmlApi* p_dml_api_{};
  Microsoft::WRL::ComPtr<ID3D12Device> d3d12_device_;
  OrtMemoryInfo* memory_info_{};
  std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> resources_;
  DmlAllocatorState state_;
  uint32_t prompt_resource_index_ = 0;
};