// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <wil/result.h>
#include "dml_adapter_info.h"
#include "dml_adapter_selection.h"

using AdapterSelection::ComPtrAndDll;
using Microsoft::WRL::ComPtr;

AdapterInfo::AdapterInfo(ID3D12Device* device)
    : AdapterInfo(device->GetAdapterLuid()) {
}

AdapterInfo::AdapterInfo(LUID adapter_luid) {
  HRESULT dxcore_result = S_OK;
  ComPtrAndDll<IDXCoreAdapterFactory> dxcore_factory = AdapterSelection::TryCreateDXCoreFactory();

  if (dxcore_factory) {
    // Try DXCore first; this is important because MCDM devices aren't enumerable through DXGI
    ComPtr<IDXCoreAdapter> adapter;
    dxcore_result = dxcore_factory.ptr->GetAdapterByLuid(adapter_luid, IID_PPV_ARGS(&adapter));

    if (SUCCEEDED(dxcore_result)) {
      Initialize(adapter.Get());
    } else if (dxcore_result != E_INVALIDARG) {
      // E_INVALIDARG can happen when the adapter LUID is not available through DXCore, so only fail for other
      // errors
      THROW_HR(dxcore_result);
    }
  }

  if (!dxcore_factory || dxcore_result == E_INVALIDARG) {
    // DXCore not available; fall back to DXGI
    if (ComPtrAndDll<IDXGIFactory4> dxgi_factory = AdapterSelection::TryCreateDXGIFactory()) {
      ComPtr<IDXGIAdapter> adapter;
      THROW_IF_FAILED(dxgi_factory.ptr->EnumAdapterByLuid(adapter_luid, IID_PPV_ARGS(&adapter)));

      Initialize(adapter.Get());
    } else {
      THROW_HR(E_FAIL);  // Neither DXCore nor DXGI were available
    }
  }
}

void AdapterInfo::Initialize(IDXCoreAdapter* adapter) {
  DXCoreHardwareID hardware_id = {};
  THROW_IF_FAILED(adapter->GetProperty(DXCoreAdapterProperty::HardwareID, &hardware_id));

  vendor_id_ = static_cast<::VendorID>(hardware_id.vendorID);
}

void AdapterInfo::Initialize(IDXGIAdapter* adapter) {
  DXGI_ADAPTER_DESC desc = {};
  THROW_IF_FAILED(adapter->GetDesc(&desc));

  vendor_id_ = static_cast<::VendorID>(desc.VendorId);
}

VendorID AdapterInfo::VendorID() const {
  return vendor_id_;
}

bool AdapterInfo::IsIntel() const {
  return (vendor_id_ == VendorID::Intel);
}
