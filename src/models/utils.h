// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace Generators {

size_t SizeOf(ONNXTensorElementDataType type);

// Slower fp16 to fp32 conversion that handles NaN and Inf (useful for debugging vs runtime conversion)
float Float16ToFloat32(uint16_t v);

// Fast fp16<->fp32 conversions that do not handle NaN and Inf but are fast (as these are not typical values)
float FastFloat16ToFloat32(const uint16_t x);
uint16_t FastFloat32ToFloat16(float v);

}  // namespace Generators