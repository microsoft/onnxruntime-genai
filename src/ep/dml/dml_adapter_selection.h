// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <wil/result.h>
#include <dxcore_interface.h>
#include <dxgi1_4.h>
#include <vector>
#include <wil/wrl.h>

// Retrieves information from a DXCore or DXGI adapter.
namespace AdapterSelection {
// Holds a strong reference to a ComPtr and an HMODULE. The HMODULE is freed *after* the pointer is. This is used to
// keep a DLL loaded while we have a pointer to something in that DLL.
template <typename T>
struct ComPtrAndDll {
  wil::unique_hmodule dll;
  Microsoft::WRL::ComPtr<T> ptr;

  explicit operator bool() { return ptr != nullptr; }

  void Reset() {
    ptr.Reset();
    dll = {};
  }
};

HRESULT CreateDXCoreFactory(_Out_ ComPtrAndDll<IDXCoreAdapterFactory>& factory);
ComPtrAndDll<IDXCoreAdapterFactory> TryCreateDXCoreFactory();

HRESULT CreateDXGIFactory(_Out_ ComPtrAndDll<IDXGIFactory4>& factory);
ComPtrAndDll<IDXGIFactory4> TryCreateDXGIFactory();

}  // namespace AdapterSelection
