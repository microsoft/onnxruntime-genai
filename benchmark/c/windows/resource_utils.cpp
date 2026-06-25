// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "resource_utils.h"

#include <Windows.h>
#include <psapi.h>
#include <dxgi1_4.h>

#include <stdexcept>
#include <string>

namespace benchmark::utils {

size_t GetPeakWorkingSetSizeInBytes() {
  PROCESS_MEMORY_COUNTERS pmc;
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    throw std::runtime_error("GetProcessMemoryInfo failed with error code " + std::to_string(GetLastError()));
  }

  return pmc.PeakWorkingSetSize;
}

GpuMemoryInfo GetGpuMemoryUsage() {
  GpuMemoryInfo result = {};
  IDXGIFactory4* factory = nullptr;
  if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
    return result;
  }

  for (UINT i = 0;; ++i) {
    IDXGIAdapter1* adapter1 = nullptr;
    const HRESULT enum_hr = factory->EnumAdapters1(i, &adapter1);
    if (enum_hr == DXGI_ERROR_NOT_FOUND) break;
    if (FAILED(enum_hr) || adapter1 == nullptr) {
      if (adapter1 != nullptr) {
        adapter1->Release();
      }
      continue;
    }

    DXGI_ADAPTER_DESC1 desc;
    if (FAILED(adapter1->GetDesc1(&desc))) {
      adapter1->Release();
      continue;
    }
    // Skip software/remote adapters
    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
      adapter1->Release();
      continue;
    }

    IDXGIAdapter3* adapter3 = nullptr;
    if (SUCCEEDED(adapter1->QueryInterface(IID_PPV_ARGS(&adapter3)))) {
      DXGI_QUERY_VIDEO_MEMORY_INFO local_info = {};
      if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local_info))) {
        if (local_info.CurrentUsage > result.dedicated) {
          result.dedicated = static_cast<size_t>(local_info.CurrentUsage);
        }
      }
      DXGI_QUERY_VIDEO_MEMORY_INFO nonlocal_info = {};
      if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &nonlocal_info))) {
        if (nonlocal_info.CurrentUsage > result.shared) {
          result.shared = static_cast<size_t>(nonlocal_info.CurrentUsage);
        }
      }
      adapter3->Release();
    }
    adapter1->Release();
  }

  factory->Release();
  return result;
}

}  // namespace benchmark::utils
