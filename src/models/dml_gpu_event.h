// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include <wrl/client.h>
#include <d3d12.h>

using Microsoft::WRL::ComPtr;

// Represents a fence which will be signaled at some point (usually by the GPU).
struct DmlGpuEvent {
  uint64_t fenceValue;
  ComPtr<ID3D12Fence> fence;

  bool IsSignaled() const {
    return fence->GetCompletedValue() >= fenceValue;
  }

  // Blocks until IsSignaled returns true.
  void WaitForSignal() const {
    if (IsSignaled())
      return;  // early-out

    // wil::unique_handle h(CreateEvent(nullptr, TRUE, FALSE, nullptr));
    // ORT_THROW_LAST_ERROR_IF(!h);

    // THROW_IF_FAILED(fence->SetEventOnCompletion(fenceValue, h.get()));

    while (!IsSignaled()) {
      // DO nothing
    }

    // WaitForSingleObject(h.get(), INFINITE);
  }
};
