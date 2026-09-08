// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <dxgi.h>
#include <dxcore_interface.h>
#include <d3d12.h>

enum class VendorID {
  Undefined = 0,
  Intel = 0x8086,
  Microsoft = 0x1414,
};

// Retrieves information from a DXCore or DXGI adapter.
class AdapterInfo {
 public:
  AdapterInfo(LUID adapter_luid);
  AdapterInfo(ID3D12Device* device);

  VendorID VendorID() const;
  bool IsIntel() const;

 private:
  void Initialize(IDXGIAdapter* adapter);
  void Initialize(IDXCoreAdapter* adapter);

  ::VendorID vendor_id_;
  uint32_t device_id_;
};
