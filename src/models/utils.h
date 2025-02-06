// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "ortx_utils.h"

#include "../span.h"

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

int64_t ElementCountFromShape(std::span<const int64_t> shape);

// Slower fp16 to fp32 conversion that handles NaN and Inf (useful for debugging vs runtime conversion)
float Float16ToFloat32(uint16_t v);

// Fast fp16<->fp32 conversions that do not handle NaN and Inf but are fast (as these are not typical values)
float FastFloat16ToFloat32(const uint16_t x);
uint16_t FastFloat32ToFloat16(float v);

}  // namespace Generators