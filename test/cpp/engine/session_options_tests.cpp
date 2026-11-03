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

TEST(CudaArenaOptionTest, ByteFieldsRejectTextAfterTheNumber) {
  using CUDAExecutionProvider::ParseArenaOptionValue;
  for (const char* key : {"max_mem", "initial_chunk_size_bytes", "max_dead_bytes_per_chunk",
                          "initial_growth_chunk_size_bytes", "arena_extend_strategy"}) {
    for (const char* value : {"1GB", "1073741824 ", "12k", "1.5", "0x10", "1e3", "5-", "1 2"}) {
      try {
        ParseArenaOptionValue(key, value);
        ADD_FAILURE() << key << "='" << value << "' must be rejected";
      } catch (const std::invalid_argument& error) {
        EXPECT_TRUE(Contains(error.what(), key)) << error.what();
        EXPECT_TRUE(Contains(error.what(), value)) << error.what();
      }
    }
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

namespace {

ProviderOptions CudaOptions(std::vector<NamedString> entries) {
  ProviderOptions options;
  options.name = "cuda";
  options.options = std::move(entries);
  return options;
}

// The device_id values of the options in order, which is what the stream binding reads (the first non-empty one wins).
std::vector<std::string> DeviceIds(const ProviderOptions& options) {
  std::vector<std::string> ids;
  for (const auto& [key, value] : options.options)
    if (key == "device_id") ids.push_back(value);
  return ids;
}

}  // namespace

TEST(CudaStreamBindingOptionsTest, WithoutAPluginDeviceAnOmittedIdMeansGpuZero) {
  // The provider-bridge EP puts a session that names no device on GPU 0 whatever the calling thread's device is, so
  // the shared stream must be bound to GPU 0 too instead of to the device that happens to be current.
  const auto resolved = CUDAExecutionProvider::ResolveStreamBindingOptions(CudaOptions({}), std::nullopt);
  EXPECT_EQ(DeviceIds(resolved), std::vector<std::string>{"0"});

  const auto unrelated = CUDAExecutionProvider::ResolveStreamBindingOptions(CudaOptions({{"enable_cuda_graph", "1"}}), std::nullopt);
  EXPECT_EQ(DeviceIds(unrelated), std::vector<std::string>{"0"});
  EXPECT_EQ(unrelated.options.size(), 2u) << "the other options are kept";
  EXPECT_EQ(unrelated.options.front(), (NamedString{"enable_cuda_graph", "1"}));
}

TEST(CudaStreamBindingOptionsTest, AnExplicitIdIsNeverReplaced) {
  for (const auto& plugin_id : {std::optional<std::string>{}, std::optional<std::string>{"0"}, std::optional<std::string>{"3"}, std::optional<std::string>{""}}) {
    const auto resolved = CUDAExecutionProvider::ResolveStreamBindingOptions(CudaOptions({{"device_id", "1"}}), plugin_id);
    EXPECT_EQ(DeviceIds(resolved), std::vector<std::string>{"1"}) << "plugin id " << plugin_id.value_or("<none>");
  }
}

TEST(CudaStreamBindingOptionsTest, AnEmptyIdCountsAsOmitted) {
  const auto resolved = CUDAExecutionProvider::ResolveStreamBindingOptions(CudaOptions({{"device_id", ""}}), std::nullopt);
  // The first non-empty value is the one that applies.
  const auto ids = DeviceIds(resolved);
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_EQ(ids.back(), "0");

  const auto plugin = CUDAExecutionProvider::ResolveStreamBindingOptions(CudaOptions({{"device_id", ""}}), std::string("2"));
  EXPECT_EQ(DeviceIds(plugin).back(), "2");
}

TEST(CudaStreamBindingOptionsTest, ThePluginDeviceFillsAnOmittedId) {
  const auto resolved = CUDAExecutionProvider::ResolveStreamBindingOptions(CudaOptions({}), std::string("2"));
  EXPECT_EQ(DeviceIds(resolved), std::vector<std::string>{"2"});
}

TEST(CudaStreamBindingOptionsTest, APluginDeviceThatReportsNoIdAddsNothing) {
  // There is nothing to bind to then; the stream follows the current device as before.
  const auto resolved = CUDAExecutionProvider::ResolveStreamBindingOptions(CudaOptions({{"enable_cuda_graph", "1"}}), std::string());
  EXPECT_TRUE(DeviceIds(resolved).empty());
  EXPECT_EQ(resolved.options.size(), 1u);
}

}  // namespace Generators::test
