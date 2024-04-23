// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <dxcore_interface.h>
#include <dxcore.h>
#include "dml_adapter_selection.h"

using Microsoft::WRL::ComPtr;

namespace AdapterSelection {
HRESULT CreateDXCoreFactory(_Out_ ComPtrAndDll<IDXCoreAdapterFactory>& factory_and_dll) {
  // Failure is expected when running on older versions of Windows that don't have DXCore.dll.
  wil::unique_hmodule dxcore_dll(LoadLibrary("DXCore.dll"));
  RETURN_LAST_ERROR_IF_NULL_EXPECTED(dxcore_dll);

  // All versions of DXCore have this symbol (failure is unexpected).
  auto dxcore_create_adapter_factory = reinterpret_cast<HRESULT(WINAPI*)(REFIID, void**)>(
      GetProcAddress(dxcore_dll.get(), "DXCoreCreateAdapterFactory"));
  RETURN_LAST_ERROR_IF_NULL(dxcore_create_adapter_factory);

  // DXCore.dll exists in Windows 19H1/19H2, and it exports DXCoreCreateAdapterFactory, but it instantiates a different
  // version of IDXCoreAdapterFactory (same name, different IID) than the one we expect. In other words, it's possible
  // and expected to get E_NOINTERFACE here if running DirectML on Windows 19H1/19H2.
  ComPtr<IDXCoreAdapterFactory> factory;
  RETURN_IF_FAILED_WITH_EXPECTED(dxcore_create_adapter_factory(IID_PPV_ARGS(&factory)), E_NOINTERFACE);

  factory_and_dll.dll = std::move(dxcore_dll);
  factory_and_dll.ptr = std::move(factory);

  return S_OK;
}

ComPtrAndDll<IDXCoreAdapterFactory> TryCreateDXCoreFactory() {
  ComPtrAndDll<IDXCoreAdapterFactory> factory_and_dll;
  CreateDXCoreFactory(/*out*/ factory_and_dll);
  return factory_and_dll;
}

HRESULT CreateDXGIFactory(_Out_ ComPtrAndDll<IDXGIFactory4>& factory_and_dll) {
  wil::unique_hmodule dxgi_dll(LoadLibrary("dxgi.dll"));
  RETURN_LAST_ERROR_IF_NULL(dxgi_dll);

  auto create_dxgi_factory = reinterpret_cast<decltype(&::CreateDXGIFactory)>(
      GetProcAddress(dxgi_dll.get(), "CreateDXGIFactory"));
  RETURN_LAST_ERROR_IF(!create_dxgi_factory);

  ComPtr<IDXGIFactory4> factory;
  RETURN_IF_FAILED(create_dxgi_factory(IID_PPV_ARGS(&factory)));

  factory_and_dll = {std::move(dxgi_dll), std::move(factory)};
  return S_OK;
}

ComPtrAndDll<IDXGIFactory4> TryCreateDXGIFactory() {
  ComPtrAndDll<IDXGIFactory4> factory_and_dll;
  CreateDXGIFactory(/*out*/ factory_and_dll);
  return factory_and_dll;
}

}  // namespace AdapterSelection
