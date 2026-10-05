// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "utils.h"

#include <cstring>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#define OGA_ARCH_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#if defined(__clang__) || defined(__GNUC__)
#define OGA_NOINLINE __attribute__((noinline))
#define OGA_TARGET_F16C __attribute__((target("avx,f16c")))
#elif defined(_MSC_VER)
#define OGA_NOINLINE __declspec(noinline)
#define OGA_TARGET_F16C
#else
#define OGA_NOINLINE
#define OGA_TARGET_F16C
#endif

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

namespace {

float LoadBFloat16AsFloat(uint16_t v) {
  const uint32_t bits = static_cast<uint32_t>(v) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

#if defined(OGA_ARCH_X86)

void CpuId(int out[4], int leaf) {
#if defined(_MSC_VER)
  __cpuid(out, leaf);
#else
  unsigned int a = 0, b = 0, c = 0, d = 0;
  __cpuid(leaf, a, b, c, d);
  out[0] = static_cast<int>(a);
  out[1] = static_cast<int>(b);
  out[2] = static_cast<int>(c);
  out[3] = static_cast<int>(d);
#endif
}

unsigned long long Xcr0() {
#if defined(_MSC_VER)
  return _xgetbv(0);
#else
  unsigned int eax = 0, edx = 0;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  return (static_cast<unsigned long long>(edx) << 32) | eax;
#endif
}

bool CpuHasF16C() {
  constexpr int kCpuidBasicLeaf = 0;
  constexpr int kCpuidFeatureLeaf = 1;
  constexpr int kEax = 0;
  constexpr int kEcx = 2;
  // CPUID leaf 1, ECX feature bits.
  constexpr int kOsxsaveBit = 1 << 27;
  constexpr int kAvxBit = 1 << 28;
  constexpr int kF16cBit = 1 << 29;
  // XCR0 bit 1 enables XMM state, bit 2 enables YMM state.
  constexpr unsigned long long kXcr0Xmm = 1ull << 1;
  constexpr unsigned long long kXcr0Ymm = 1ull << 2;
  constexpr unsigned long long kXcr0XmmYmm = kXcr0Xmm | kXcr0Ymm;

  int info[4] = {};
  CpuId(info, kCpuidBasicLeaf);
  if (info[kEax] < kCpuidFeatureLeaf)
    return false;

  CpuId(info, kCpuidFeatureLeaf);
  const int ecx = info[kEcx];
  const bool osxsave = (ecx & kOsxsaveBit) != 0;
  const bool avx = (ecx & kAvxBit) != 0;
  const bool f16c = (ecx & kF16cBit) != 0;
  if (!osxsave || !avx || !f16c)
    return false;

  return (Xcr0() & kXcr0XmmYmm) == kXcr0XmmYmm;
}

// Not inlined into the dispatcher: vcvtph2ps and vzeroupper must not run when F16C is absent.
OGA_NOINLINE OGA_TARGET_F16C void ConvertFloat16ToFloat32F16C(const uint16_t* src, float* dst, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m128i halves = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
    const __m256 values = _mm256_cvtph_ps(halves);
    _mm256_storeu_ps(dst + i, values);
  }
  for (; i < count; ++i) {
    const __m128i half = _mm_cvtsi32_si128(static_cast<int>(src[i]));
    dst[i] = _mm_cvtss_f32(_mm_cvtph_ps(half));
  }
}

// SSE2 is the baseline ISA, so this can be inlined into the caller.
void ConvertBFloat16ToFloat32Sse(const uint16_t* src, float* dst, size_t count) {
  size_t i = 0;
  const __m128i zero = _mm_setzero_si128();
  for (; i + 4 <= count; i += 4) {
    __m128i halves = _mm_setzero_si128();
    std::memcpy(&halves, src + i, sizeof(uint64_t));
    const __m128i extended = _mm_unpacklo_epi16(zero, halves);
    _mm_storeu_ps(dst + i, _mm_castsi128_ps(extended));
  }
  for (; i < count; ++i)
    dst[i] = LoadBFloat16AsFloat(src[i]);
}

#endif  // OGA_ARCH_X86

}  // namespace

void ConvertFloat16ToFloat32(const uint16_t* src, float* dst, size_t count) {
#if defined(OGA_ARCH_X86)
  static const bool use_f16c = CpuHasF16C();
  if (use_f16c) {
    ConvertFloat16ToFloat32F16C(src, dst, count);
    return;
  }
#endif
  for (size_t i = 0; i < count; ++i)
    dst[i] = FastFloat16ToFloat32(src[i]);
}

void ConvertFloat32ToFloat16(const float* src, uint16_t* dst, size_t count) {
  for (size_t i = 0; i < count; ++i)
    dst[i] = FastFloat32ToFloat16(src[i]);
}

void ConvertBFloat16ToFloat32(const uint16_t* src, float* dst, size_t count) {
#if defined(OGA_ARCH_X86)
  ConvertBFloat16ToFloat32Sse(src, dst, count);
#else
  for (size_t i = 0; i < count; ++i)
    dst[i] = LoadBFloat16AsFloat(src[i]);
#endif
}

void ConvertFloat32ToBFloat16(const float* src, uint16_t* dst, size_t count) {
  for (size_t i = 0; i < count; ++i)
    dst[i] = Float32ToBFloat16(src[i]);
}

}  // namespace Generators