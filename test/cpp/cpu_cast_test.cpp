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
// the scalar fast path and the hardware path, so those patterns are not checked here.
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
