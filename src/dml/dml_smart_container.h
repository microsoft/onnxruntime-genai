// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <wrl/client.h>
#include <wrl/implements.h>
#include "../models/onnxruntime_api.h"

// Allows objects to be added to a D3D12 object via SetPrivateDataInterface and extend its lifetime beyond the life of the model. For
// example, we can put the DML allocator on the D3D12 device (which is a unique singleton for each adapter) and be sure that the allocator won't be
// destroyed until nothing holds on to the device anymore.
class DmlSmartContainer : public Microsoft::WRL::RuntimeClass<Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IUnknown> {
 public:
  DmlSmartContainer(std::unique_ptr<OrtMemoryInfo>&& memory_info, std::unique_ptr<OrtAllocator>&& allocator, std::unique_ptr<OrtAllocator>&& dml_allocation_decoder)
      : memory_info_(std::move(memory_info)), allocator_(std::move(allocator)), dml_allocation_decoder_(std::move(dml_allocation_decoder)) {}

  const OrtMemoryInfo* GetMemoryInfo() const { return memory_info_.get(); }
  OrtAllocator* GetAllocator() const { return allocator_.get(); }
  OrtAllocator* GetAllocationDecoder() const { return dml_allocation_decoder_.get(); }

 private:
  std::unique_ptr<OrtMemoryInfo> memory_info_;
  std::unique_ptr<OrtAllocator> allocator_;
  std::unique_ptr<OrtAllocator> dml_allocation_decoder_;
};