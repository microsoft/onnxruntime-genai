// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <dxcore_interface.h>
#include <dxgi1_4.h>
#include <vector>
#include <winrt/base.h>

struct hmodule_traits {
  using type = HMODULE;

  static void close(type value) noexcept {
    WINRT_VERIFY_(1, ::FreeLibrary(value));
  }

  static type invalid() noexcept {
    return reinterpret_cast<type>(-1);
  }
};

using hmodule_handle = winrt::handle_type<hmodule_traits>;

// Retrieves information from a DXCore or DXGI adapter.
namespace AdapterSelection {
// Holds a strong reference to a ComPtr and an HMODULE. The HMODULE is freed *after* the pointer is. This is used to
// keep a DLL loaded while we have a pointer to something in that DLL.
template <typename T>
struct ComPtrAndDll {
  hmodule_handle dll;
  winrt::com_ptr<T> ptr;

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
