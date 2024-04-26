// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include <wrl/client.h>
#include <d3d12.h>

using Microsoft::WRL::ComPtr;

// Represents a fence which will be signaled at some point (usually by the GPU).
struct DmlGpuEvent {
  uint64_t fence_value;
  ComPtr<ID3D12Fence> fence;

  bool IsSignaled() const {
    return fence->GetCompletedValue() >= fence_value;
  }

  // Blocks until IsSignaled returns true.
  void WaitForSignal() const {
    if (IsSignaled()) {
      return;  // early-out
    }

    while (!IsSignaled()) {
#if defined(_M_AMD64) || defined(__x86_64__)
      _mm_pause();
#endif
    }
  }
};