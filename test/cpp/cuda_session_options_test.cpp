// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "ep/cuda/session_options.h"

namespace Generators::test {
namespace {

TEST(CudaProviderBridgeOptionsTest, TranslatesSupportedArenaOptions) {
  Config::ProviderOptions options;
  options.options = {
      {"max_mem", "1073741824"},
      {"arena_extend_strategy", "1"},
      {"enable_cuda_graph", "1"}};

  const auto normalized =
      CUDAExecutionProvider::NormalizeProviderBridgeOptions(options);

  EXPECT_EQ(normalized.options,
            (std::vector<Config::NamedString>{
                {"gpu_mem_limit", "1073741824"},
                {"arena_extend_strategy", "1"},
                {"enable_cuda_graph", "1"}}));
}

TEST(CudaProviderBridgeOptionsTest, RejectsPluginOnlyArenaOptionsBeforeAppend) {
  Config::ProviderOptions options;
  options.options = {{"initial_chunk_size_bytes", "1048576"}};
  EXPECT_THROW(
      CUDAExecutionProvider::NormalizeProviderBridgeOptions(options),
      std::runtime_error);

  options.options = {{"arena.max_mem", "1073741824"}};
  EXPECT_THROW(
      CUDAExecutionProvider::NormalizeProviderBridgeOptions(options),
      std::runtime_error);
}

TEST(CudaProviderBridgeOptionsTest, RejectsConflictingMemoryLimits) {
  Config::ProviderOptions options;
  options.options = {
      {"max_mem", "1073741824"},
      {"gpu_mem_limit", "536870912"}};

  EXPECT_THROW(
      CUDAExecutionProvider::NormalizeProviderBridgeOptions(options),
      std::invalid_argument);
}

}  // namespace
}  // namespace Generators::test
