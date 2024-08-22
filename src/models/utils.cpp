// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"

namespace Generators {

size_t SizeOf(ONNXTensorElementDataType type) {
  switch (type) {
    case Ort::TypeToTensorType<uint8_t>:
      return sizeof(uint8_t);
    case Ort::TypeToTensorType<int8_t>:
      return sizeof(int8_t);
    case Ort::TypeToTensorType<uint16_t>:
      return sizeof(uint16_t);
    case Ort::TypeToTensorType<int16_t>:
      return sizeof(int16_t);
    case Ort::TypeToTensorType<uint32_t>:
      return sizeof(uint32_t);
    case Ort::TypeToTensorType<int32_t>:
      return sizeof(int32_t);
    case Ort::TypeToTensorType<uint64_t>:
      return sizeof(int64_t);
    case Ort::TypeToTensorType<int64_t>:
      return sizeof(int64_t);
    case Ort::TypeToTensorType<bool>:
      return sizeof(bool);
    case Ort::TypeToTensorType<float>:
      return sizeof(float);
    case Ort::TypeToTensorType<double>:
      return sizeof(double);
    case Ort::TypeToTensorType<Ort::Float16_t>:
      return sizeof(Ort::Float16_t);
    case Ort::TypeToTensorType<Ort::BFloat16_t>:
      return sizeof(Ort::BFloat16_t);
    default:
      throw std::runtime_error("Unsupported ONNXTensorElementDataType in GetTypeSize");
  }
}

// IEEE 752-2008 binary16 format, 1 sign bit, 5 bit exponent, 10 bit fraction
float Float16ToFloat32(uint16_t v) {
  // Extract sign, exponent, and fraction from numpy.float16
  int const sign = (v & 0x8000) >> 15;
  int const exponent = (v & 0x7C00) >> 10;
  int const fraction = v & 0x03FF;

  // Handle special cases
  if (exponent == 0) {
    if (fraction == 0) {
      // Zero
      return sign != 0 ? -0.0f : 0.0f;
    }  // Subnormal number
    return std::ldexp((sign != 0 ? -1.0f : 1.0f) * static_cast<float>(fraction) / 1024.0f, -14);
  }
  if (exponent == 31) {
    if (fraction == 0) {
      // Infinity
      return sign != 0 ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
    }  // NaN
    return std::numeric_limits<float>::quiet_NaN();
  }

  // Normalized number
  return std::ldexp((sign != 0 ? -1.0f : 1.0f) * (1.0f + static_cast<float>(fraction) / 1024.0f), exponent - 15);
}

// C++17 compatible version of bit_cast for the code below
template <typename TTo, typename TFrom>
TTo bit_cast(TFrom x) {
  return *reinterpret_cast<TTo*>(&x);
}

// IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
// IEEE 752-2008 binary16 format, 1 sign bit, 5 bit exponent, 10 bit fraction
float FastFloat16ToFloat32(const uint16_t x) {
  const uint32_t e = (x & 0x7C00) >> 10;  // exponent
  const uint32_t m = (x & 0x03FF) << 13;  // mantissa

  const uint32_t v = bit_cast<uint32_t>((float)m) >> 23;                                                                                                       // log2 bit hack to count leading zeros in denormalized format
  return bit_cast<float>((x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) | ((e == 0) & (m != 0)) * ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000)));  // sign : normalized : denormalized
}

uint16_t FastFloat32ToFloat16(float v) {
  const uint32_t b = bit_cast<uint32_t>(v) + 0x00001000;  // round-to-nearest-even: add last bit after truncated mantissa

  const uint32_t e = (b & 0x7F800000) >> 23;                                                                                                                                                                  // exponent
  const uint32_t m = b & 0x007FFFFF;                                                                                                                                                                          // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
  return static_cast<uint16_t>((b & 0x80000000) >> 16 | (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) | ((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) | (e > 143) * 0x7FFF);  // sign : normalized : denormalized : saturate
}

}  // namespace Generators