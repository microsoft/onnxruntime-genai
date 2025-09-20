// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "utils.h"

namespace Generators {

DeviceSpan<uint8_t> ByteWrapTensor(DeviceInterface& device, OrtValue& value) {
  auto info = value.GetTensorTypeAndShapeInfo();
  return device.WrapMemory(std::span<uint8_t>{value.GetTensorMutableData<uint8_t>(), info->GetElementCount() * Ort::SizeOf(info->GetElementType())});
}

const char* TypeToString(ONNXTensorElementDataType type) {
  switch (type) {
    case Ort::TypeToTensorType<uint8_t>:
      return "uint8";
    case Ort::TypeToTensorType<int8_t>:
      return "int8";
    case Ort::TypeToTensorType<uint16_t>:
      return "uint16";
    case Ort::TypeToTensorType<int16_t>:
      return "int16";
    case Ort::TypeToTensorType<uint32_t>:
      return "uint32";
    case Ort::TypeToTensorType<int32_t>:
      return "int32";
    case Ort::TypeToTensorType<uint64_t>:
      return "uint64";
    case Ort::TypeToTensorType<int64_t>:
      return "int64";
    case Ort::TypeToTensorType<bool>:
      return "bool";
    case Ort::TypeToTensorType<float>:
      return "float32";
    case Ort::TypeToTensorType<double>:
      return "float64";
    case Ort::TypeToTensorType<Ort::Float16_t>:
      return "float16";
    case Ort::TypeToTensorType<Ort::BFloat16_t>:
      return "bfloat16";
    default:
      return "(unsupported type, please add)";
  }
}

int64_t ElementCountFromShape(std::span<const int64_t> shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}

template <int exponent_bits, int fraction_bits>
float TFloatToFloat32(uint16_t v) {
  constexpr int exponent_bias = (1 << (exponent_bits - 1)) - 1;
  constexpr int fraction_mask = (1 << fraction_bits) - 1;
  constexpr int exponent_mask = ((1 << exponent_bits) - 1) << fraction_bits;

  int sign = v >> (exponent_bits + fraction_bits);
  int exponent = (v & exponent_mask) >> fraction_bits;
  int fraction = v & fraction_mask;

  // Handle special cases
  if (exponent == 0) {
    if (fraction == 0)  // Zero
      return sign != 0 ? -0.0f : 0.0f;
    // Subnormal number
    return std::ldexp((sign != 0 ? -1.0f : 1.0f) * static_cast<float>(fraction) / (1 << fraction_bits), 1 - exponent_bias);
  }
  if (exponent == (1 << exponent_bits) - 1) {
    if (fraction == 0)  // Infinity
      return sign != 0 ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
    // NaN
    return std::numeric_limits<float>::quiet_NaN();
  }

  // Normalized number
  return std::ldexp((sign != 0 ? -1.0f : 1.0f) * (1.0f + static_cast<float>(fraction) / (1 << fraction_bits)), exponent - exponent_bias);
}

// IEEE 752-2008 binary16 format, 1 sign bit, 5 bit exponent, 10 bit fraction
float Float16ToFloat32(uint16_t v) {
  return TFloatToFloat32<5, 10>(v);
}

// BFloat16 binary16 format, 1 sign bit, 8 bit exponent, 7 bit fraction
float BFloat16ToFloat32(uint16_t v) {
  return TFloatToFloat32<8, 7>(v);
}

// Get most significant 16 bits
uint16_t Float32ToBFloat16(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
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