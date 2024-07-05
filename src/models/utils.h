// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "ortx_utils.h"

namespace Generators {

template <typename T>
struct OrtxPtr {
  ~OrtxPtr() { OrtxDispose(&p_); }
  T** Address() {
    assert(!p_);
    return &p_;
  }
  operator T*() { return p_; }
  operator const T*() const { return p_; }

  T* p_{};
};

size_t SizeOf(ONNXTensorElementDataType type);

// Slower fp16 to fp32 conversion that handles NaN and Inf (useful for debugging vs runtime conversion)
float Float16ToFloat32(uint16_t v);

// Fast fp16<->fp32 conversions that do not handle NaN and Inf but are fast (as these are not typical values)
float FastFloat16ToFloat32(const uint16_t x);
uint16_t FastFloat32ToFloat16(float v);

// Creates a copy of OrtValue on the device that was specified in the model
// returns a shared_ptr with OrtValue so it can be shared.
std::shared_ptr<OrtValue> CopyToDevice(const OrtValue& source, const Model& model);

/// <summary>
/// Copies the source OrtValue to ort_device according to the settings
/// </summary>
/// <param name="source"></param>
/// <param name="ort_device"></param>
/// <param name="device_type"></param>
/// <param name="cuda_stream"></param>
void CopyToDevice(const OrtValue& source, OrtValue& ort_device, DeviceType device_type, cuda_stream_holder cuda_stream);

/// <summary>
/// Creates an OrtValue over the same buffer as the source
/// </summary>
/// <param name="source"></param>
/// <returns>a shared ptr</returns>
std::shared_ptr<OrtValue> DuplicateOrtValue(OrtValue& source);

}  // namespace Generators