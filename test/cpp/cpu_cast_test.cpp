// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "generator/generators.h"
#include "ep/cpu/interface.h"
#include "models/utils.h"
#include "telemetry_test_environment.h"

namespace {

using Generators::DeviceInterface;

DeviceInterface& Cpu() {
  static auto cpu = Generators::CreateCpuInterface();
  return *cpu;
}

uint32_t FloatBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float FloatFromBits(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// IEEE-754 binary16 to binary32. CpuInterface::Cast must match this for zeros and
// normal numbers on every backend (scalar and F16C). Inf and NaN differ between
// the scalar fast path and the hardware path; Float16ToFloat32SpecialValues covers those.
float Float16ToFloat32Reference(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  const uint32_t exponent = (value & 0x7C00u) >> 10;
  const uint32_t mantissa = value & 0x03FFu;

  uint32_t bits = 0;
  if (exponent == 0) {
    bits = sign;
  } else {
    bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
  }
  return FloatFromBits(bits);
}

bool IsFloat16ZeroOrNormal(uint16_t value) {
  const uint32_t exponent = (value & 0x7C00u) >> 10;
  if (exponent == 0)
    return (value & 0x03FFu) == 0;
  return exponent != 31;
}

uint16_t Float16Pattern(size_t index) {
  // Constants first, then a stride through the remaining encodings.
  static constexpr uint16_t kSpecial[] = {
      0x0000, 0x8000, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x3800, 0x7BFF, 0x0400, 0x3555};
  if (index < std::size(kSpecial))
    return kSpecial[index];
  return static_cast<uint16_t>(index * 97u);
}

void ExpectFloat16ToFloat32(size_t count, size_t source_pad, size_t dest_pad) {
  std::vector<uint16_t> source(count + source_pad);
  std::vector<float> dest(count + dest_pad, 42.0f);
  for (size_t i = 0; i < count; ++i)
    source[source_pad + i] = Float16Pattern(i);

  ASSERT_TRUE(Cpu().Cast(source.data() + source_pad, dest.data() + dest_pad,
                         Ort::TypeToTensorType<Ort::Float16_t>,
                         Ort::TypeToTensorType<float>, count));

  for (size_t i = 0; i < count; ++i) {
    const uint16_t encoded = source[source_pad + i];
    if (!IsFloat16ZeroOrNormal(encoded))
      continue;
    const uint32_t actual = FloatBits(dest[dest_pad + i]);
    const uint32_t expected = FloatBits(Float16ToFloat32Reference(encoded));
    EXPECT_EQ(actual, expected) << "index " << i << " fp16 0x" << std::hex << encoded;
    if (actual != expected)
      return;
  }
}

void ExpectBFloat16ToFloat32(size_t count, size_t source_pad) {
  std::vector<uint16_t> source(count + source_pad);
  std::vector<float> dest(count, 42.0f);
  for (size_t i = 0; i < count; ++i)
    source[source_pad + i] = static_cast<uint16_t>(i);

  ASSERT_TRUE(Cpu().Cast(source.data() + source_pad, dest.data(),
                         Ort::TypeToTensorType<Ort::BFloat16_t>,
                         Ort::TypeToTensorType<float>, count));

  for (size_t i = 0; i < count; ++i) {
    const uint16_t encoded = source[source_pad + i];
    const uint32_t expected = static_cast<uint32_t>(encoded) << 16;
    const uint32_t actual = FloatBits(dest[i]);
    EXPECT_EQ(actual, expected) << "index " << i << " bf16 0x" << std::hex << encoded;
    if (actual != expected)
      return;
  }
}

// The two fp16 paths agree on every encoding except Inf and NaN.
// F16C follows vcvtph2ps: ±Inf stays ±Inf, and a NaN becomes a quiet NaN with the
// sign kept and the 10-bit payload shifted into the top of the fp32 significand.
// FastFloat16ToFloat32 treats the all-ones exponent as a normal number, so ±Inf
// becomes ±65536 and a NaN becomes a large finite value.
struct Float16Special {
  uint16_t encoded;
  uint32_t f16c_bits;
  uint32_t scalar_bits;
};

constexpr Float16Special kFloat16Specials[] = {
    {0x7C00, 0x7F800000u, 0x47800000u},  // +Inf -> +Inf, or 65536
    {0xFC00, 0xFF800000u, 0xC7800000u},  // -Inf -> -Inf, or -65536
    {0x7C01, 0x7FC02000u, 0x47802000u},  // signaling NaN
    {0x7E00, 0x7FC00000u, 0x47C00000u},  // quiet NaN
    {0x7FFF, 0x7FFFE000u, 0x47FFE000u},  // NaN, full payload
    {0xFC01, 0xFFC02000u, 0xC7802000u},  // negative signaling NaN
    {0xFE00, 0xFFC00000u, 0xC7C00000u},  // negative quiet NaN
    {0xFFFF, 0xFFFFE000u, 0xC7FFE000u},  // negative NaN, full payload
};

void ExpectFloat16SpecialValues(size_t count, size_t source_pad, size_t dest_pad, bool f16c_path) {
  std::vector<uint16_t> source(count + source_pad);
  std::vector<float> dest(count + dest_pad, 42.0f);
  for (size_t i = 0; i < count; ++i)
    source[source_pad + i] = kFloat16Specials[i % std::size(kFloat16Specials)].encoded;

  ASSERT_TRUE(Cpu().Cast(source.data() + source_pad, dest.data() + dest_pad,
                         Ort::TypeToTensorType<Ort::Float16_t>,
                         Ort::TypeToTensorType<float>, count));

  for (size_t i = 0; i < count; ++i) {
    const Float16Special& spec = kFloat16Specials[i % std::size(kFloat16Specials)];
    const uint32_t expected = f16c_path ? spec.f16c_bits : spec.scalar_bits;
    const uint32_t actual = FloatBits(dest[dest_pad + i]);
    EXPECT_EQ(actual, expected) << "index " << i << " fp16 0x" << std::hex << spec.encoded
                                << (f16c_path ? " f16c" : " scalar");
    if (actual != expected)
      return;
  }
}

}  // namespace

TEST(CpuCastTest, Float16ToFloat32) {
  // Lengths cover the F16C body (8-wide), its tail, and a vocabulary-sized buffer.
  for (size_t count : {size_t{1}, size_t{3}, size_t{4}, size_t{7}, size_t{8}, size_t{9},
                       size_t{15}, size_t{16}, size_t{17}, size_t{31}, size_t{32}, size_t{128256}}) {
    ExpectFloat16ToFloat32(count, 0, 0);
  }
  // Source and destination are intentionally not 16-byte aligned.
  ExpectFloat16ToFloat32(17, 1, 1);
}

TEST(CpuCastTest, Float16ToFloat32SpecialValues) {
  for (const Float16Special& spec : kFloat16Specials) {
    const uint32_t scalar = FloatBits(Generators::FastFloat16ToFloat32(spec.encoded));
    EXPECT_EQ(scalar, spec.scalar_bits) << "fp16 0x" << std::hex << spec.encoded;
  }

  uint16_t positive_inf = 0x7C00;
  float converted = 0.0f;
  ASSERT_TRUE(Cpu().Cast(&positive_inf, &converted,
                         Ort::TypeToTensorType<Ort::Float16_t>,
                         Ort::TypeToTensorType<float>, 1));
  const uint32_t inf_bits = FloatBits(converted);
  bool f16c_path = false;
  if (inf_bits == 0x7F800000u) {
    f16c_path = true;
  } else {
    ASSERT_EQ(inf_bits, 0x47800000u) << "fp16 +Inf must be IEEE Inf or the scalar 65536";
  }

  // Lengths cover the 8-wide F16C body, its tail, and an unaligned source and destination.
  for (size_t count : {size_t{1}, size_t{3}, size_t{7}, size_t{8}, size_t{9}, size_t{17}})
    ExpectFloat16SpecialValues(count, 0, 0, f16c_path);
  ExpectFloat16SpecialValues(9, 1, 1, f16c_path);
}

TEST(CpuCastTest, BFloat16ToFloat32) {
  // Lengths cover the 4-wide SSE body, its tail, and a full sweep of encodings.
  for (size_t count : {size_t{1}, size_t{3}, size_t{4}, size_t{5}, size_t{8}, size_t{65535}, size_t{65536}}) {
    ExpectBFloat16ToFloat32(count, 0);
  }
  ExpectBFloat16ToFloat32(5, 1);
}

TEST(CpuCastTest, Float32ToFloat16) {
  const float values[] = {0.0f, 1.0f, -1.0f, 2.0f, 0.5f, 65504.0f, 1.0e10f, -1.0e10f};
  const uint16_t expected[] = {0x0000, 0x3C00, 0xBC00, 0x4000, 0x3800, 0x7BFF, 0x7FFF, 0xFFFF};
  static_assert(std::size(values) == std::size(expected));

  std::vector<uint16_t> dest(std::size(values), 0x1234);
  ASSERT_TRUE(Cpu().Cast(const_cast<float*>(values), dest.data(),
                         Ort::TypeToTensorType<float>,
                         Ort::TypeToTensorType<Ort::Float16_t>, std::size(values)));
  for (size_t i = 0; i < std::size(values); ++i)
    EXPECT_EQ(dest[i], expected[i]) << "value " << values[i];
}

TEST(CpuCastTest, Float32ToBFloat16Truncates) {
  // 1.0f is an exact bf16. The extra mantissa bit must be dropped, not rounded up.
  const float values[] = {0.0f, 1.0f, -1.0f, FloatFromBits(0x3F800001u)};
  const uint16_t expected[] = {0x0000, 0x3F80, 0xBF80, 0x3F80};

  std::vector<uint16_t> dest(std::size(values));
  ASSERT_TRUE(Cpu().Cast(const_cast<float*>(values), dest.data(),
                         Ort::TypeToTensorType<float>,
                         Ort::TypeToTensorType<Ort::BFloat16_t>, std::size(values)));
  for (size_t i = 0; i < std::size(values); ++i)
    EXPECT_EQ(dest[i], expected[i]);
}

TEST(CpuCastTest, Int32ToInt64) {
  const int32_t values[] = {0, 1, -1, std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::min()};
  std::vector<int64_t> dest(std::size(values));
  ASSERT_TRUE(Cpu().Cast(const_cast<int32_t*>(values), dest.data(),
                         Ort::TypeToTensorType<int32_t>,
                         Ort::TypeToTensorType<int64_t>, std::size(values)));
  for (size_t i = 0; i < std::size(values); ++i)
    EXPECT_EQ(dest[i], static_cast<int64_t>(values[i]));
}

TEST(CpuCastTest, EmptyCountLeavesOutputUntouched) {
  uint16_t input = 0x3C00;
  float output = 42.0f;
  ASSERT_TRUE(Cpu().Cast(&input, &output,
                         Ort::TypeToTensorType<Ort::Float16_t>,
                         Ort::TypeToTensorType<float>, 0));
  EXPECT_EQ(output, 42.0f);
}

TEST(CpuCastTest, RejectsSameType) {
  float input = 1.0f;
  float output = 0.0f;
  EXPECT_THROW(Cpu().Cast(&input, &output,
                          Ort::TypeToTensorType<float>,
                          Ort::TypeToTensorType<float>, 1),
               std::runtime_error);
}

TEST(CpuCastTest, RejectsUnimplementedType) {
  double input = 1.0;
  float output = 0.0f;
  EXPECT_THROW(Cpu().Cast(&input, &output,
                          Ort::TypeToTensorType<double>,
                          Ort::TypeToTensorType<float>, 1),
               std::runtime_error);
}

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  Generators::Shutdown();
  return result;
}
