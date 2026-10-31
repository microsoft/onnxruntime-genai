// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ep/cuda/session_options.h"
#include "models/session_options.h"

namespace Generators::test {

namespace {

constexpr size_t kOrtDefault = static_cast<size_t>(-1);

bool Contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

std::string SelectionError(const std::vector<std::string>& ids, const std::optional<std::string>& requested,
                           bool hardware_filtered) {
  try {
    SelectCudaDeviceIndex(ids, requested, hardware_filtered);
  } catch (const std::runtime_error& error) {
    return error.what();
  }
  return {};
}

}  // namespace

TEST(CudaArenaOptionTest, StrategyAcceptsTheSymbolicNamesAndNumbers) {
  using CUDAExecutionProvider::ParseArenaOptionValue;
  EXPECT_EQ(ParseArenaOptionValue("arena_extend_strategy", "kNextPowerOfTwo"), 0u);
  EXPECT_EQ(ParseArenaOptionValue("arena_extend_strategy", "kSameAsRequested"), 1u);
  EXPECT_EQ(ParseArenaOptionValue("arena_extend_strategy", "0"), 0u);
  EXPECT_EQ(ParseArenaOptionValue("arena_extend_strategy", "1"), 1u);
  EXPECT_EQ(ParseArenaOptionValue("arena_extend_strategy", "-1"), kOrtDefault);
}

TEST(CudaArenaOptionTest, StrategyRejectsUnknownNames) {
  using CUDAExecutionProvider::ParseArenaOptionValue;
  EXPECT_THROW(ParseArenaOptionValue("arena_extend_strategy", "kSomethingElse"), std::invalid_argument);
  EXPECT_THROW(ParseArenaOptionValue("arena_extend_strategy", "ksameasrequested"), std::invalid_argument);
  EXPECT_THROW(ParseArenaOptionValue("arena_extend_strategy", ""), std::invalid_argument);
}

TEST(CudaArenaOptionTest, EveryByteFieldRejectsTheStrategyNames) {
  using CUDAExecutionProvider::ParseArenaOptionValue;
  for (const char* key : {"max_mem", "initial_chunk_size_bytes", "max_dead_bytes_per_chunk",
                          "initial_growth_chunk_size_bytes"}) {
    for (const char* name : {"kNextPowerOfTwo", "kSameAsRequested"}) {
      try {
        ParseArenaOptionValue(key, name);
        ADD_FAILURE() << key << "=" << name << " must be rejected";
      } catch (const std::invalid_argument& error) {
        EXPECT_TRUE(Contains(error.what(), key)) << error.what();
      }
    }
  }
}

TEST(CudaArenaOptionTest, ByteFieldsKeepValidatedNumericParsing) {
  using CUDAExecutionProvider::ParseArenaOptionValue;
  for (const char* key : {"max_mem", "initial_chunk_size_bytes", "max_dead_bytes_per_chunk",
                          "initial_growth_chunk_size_bytes"}) {
    EXPECT_EQ(ParseArenaOptionValue(key, "0"), 0u) << key;
    EXPECT_EQ(ParseArenaOptionValue(key, "1073741824"), 1073741824u) << key;
    EXPECT_EQ(ParseArenaOptionValue(key, "-1"), kOrtDefault) << key;
    EXPECT_THROW(ParseArenaOptionValue(key, "-2"), std::out_of_range) << key;
    EXPECT_THROW(ParseArenaOptionValue(key, "99999999999999999999"), std::out_of_range) << key;
    EXPECT_THROW(ParseArenaOptionValue(key, "lots"), std::invalid_argument) << key;
  }
}

TEST(CudaDeviceSelectionTest, MatchesTheRequestedIdWhateverTheEnumerationOrder) {
  const std::vector<std::string> ids = {"2", "0", "3", "1"};
  EXPECT_EQ(SelectCudaDeviceIndex(ids, std::string("0"), false), 1u);
  EXPECT_EQ(SelectCudaDeviceIndex(ids, std::string("1"), false), 3u);
  EXPECT_EQ(SelectCudaDeviceIndex(ids, std::string("2"), false), 0u);
  EXPECT_EQ(SelectCudaDeviceIndex(ids, std::string("3"), false), 2u);
}

TEST(CudaDeviceSelectionTest, WithoutAnExplicitIdUsesCudaDeviceZeroEvenWhenEnumeratedLater) {
  EXPECT_EQ(SelectCudaDeviceIndex({"1", "0"}, std::nullopt, false), 1u);
  EXPECT_EQ(SelectCudaDeviceIndex({"0", "1"}, std::nullopt, false), 0u);
}

TEST(CudaDeviceSelectionTest, AnAbsentExplicitIdIsRejectedInsteadOfFallingBackToTheFirstDevice) {
  const std::string error = SelectionError({"0", "1"}, std::string("7"), false);
  ASSERT_FALSE(error.empty());
  EXPECT_TRUE(Contains(error, "device_id=7")) << error;
  EXPECT_TRUE(Contains(error, "[0, 1]")) << error;
  EXPECT_FALSE(Contains(error, "device_filtering_options")) << error;
}

TEST(CudaDeviceSelectionTest, HardwareFilteringTakesPrecedenceOverAnExplicitId) {
  // Filtering left only the GPU with CUDA id 1, but the session explicitly asks for GPU 0.
  const std::string conflict = SelectionError({"1"}, std::string("0"), true);
  ASSERT_FALSE(conflict.empty());
  EXPECT_TRUE(Contains(conflict, "device_id=0")) << conflict;
  EXPECT_TRUE(Contains(conflict, "[1]")) << conflict;
  EXPECT_TRUE(Contains(conflict, "device_filtering_options")) << conflict;

  EXPECT_EQ(SelectCudaDeviceIndex({"1"}, std::string("1"), true), 0u);
  EXPECT_EQ(SelectCudaDeviceIndex({"3", "1"}, std::string("1"), true), 1u);
}

TEST(CudaDeviceSelectionTest, HardwareFilteringPicksTheSurvivorWhenNoIdIsRequested) {
  EXPECT_EQ(SelectCudaDeviceIndex({"3", "1"}, std::nullopt, true), 0u);
}

TEST(CudaDeviceSelectionTest, DevicesThatReportNoIdNeverSatisfyAnExplicitId) {
  const std::string error = SelectionError({""}, std::string("0"), false);
  ASSERT_FALSE(error.empty());
  EXPECT_TRUE(Contains(error, "<not reported>")) << error;
  EXPECT_EQ(SelectCudaDeviceIndex({""}, std::nullopt, false), 0u);
}

TEST(CudaDeviceSelectionTest, NoCandidatesIsAnError) {
  EXPECT_THROW(SelectCudaDeviceIndex({}, std::nullopt, false), std::runtime_error);
  EXPECT_THROW(SelectCudaDeviceIndex({}, std::string("0"), false), std::runtime_error);
}

}  // namespace Generators::test
