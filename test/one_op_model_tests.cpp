// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <vector>

#include "../src/models/one_op_model_executor.h"

using namespace Generators;

class OneOpModelTests : public ::testing::Test {
 protected:
  void SetUp() override {
    // Clear the cache before each test to ensure clean state
    OneOpModelExecutor::ClearCache();
  }

  void TearDown() override {
    // Clear the cache after each test
    OneOpModelExecutor::ClearCache();
  }
};

// Test 1: ExecuteCastOp - Float to Int32 on CPU
TEST_F(OneOpModelTests, ExecuteCastOpFloatToInt32) {
  std::vector<float> input = {1.7f, 2.3f, 3.9f, 4.1f, -5.5f};
  std::vector<int32_t> output(input.size());
  std::vector<int32_t> expected = {1, 2, 3, 4, -5};  // Truncation behavior

  // Create CPU memory info using C API directly
  OrtMemoryInfo* cpu_mem_info_ptr = nullptr;
  OrtGetApiBase()->GetApi(ORT_API_VERSION)->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &cpu_mem_info_ptr);

  ExecuteCastOp(
      input.data(),
      output.data(),
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
      input.size(),
      "",  // Empty EP name = CPU
      cpu_mem_info_ptr);

  OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseMemoryInfo(cpu_mem_info_ptr);

  // Verify results
  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_EQ(output[i], expected[i]) << "Mismatch at index " << i;
  }
}

// Test 2: ExecuteCastOp - Int32 to Float on CPU
TEST_F(OneOpModelTests, ExecuteCastOpInt32ToFloat) {
  std::vector<int32_t> input = {1, 2, 3, 4, 5};
  std::vector<float> output(input.size());

  OrtMemoryInfo* cpu_mem_info_ptr = nullptr;
  OrtGetApiBase()->GetApi(ORT_API_VERSION)->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &cpu_mem_info_ptr);

  ExecuteCastOp(
      input.data(),
      output.data(),
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      input.size(),
      "",
      cpu_mem_info_ptr);

  OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseMemoryInfo(cpu_mem_info_ptr);

  // Verify results
  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(input[i])) << "Mismatch at index " << i;
  }
}

// Test 3: ExecuteCastOp - Different sizes (cache reuse test)
TEST_F(OneOpModelTests, ExecuteCastOpCacheReuse) {
  OrtMemoryInfo* cpu_mem_info_ptr = nullptr;
  OrtGetApiBase()->GetApi(ORT_API_VERSION)->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &cpu_mem_info_ptr);

  // First execution with size 3
  {
    std::vector<float> input = {1.5f, 2.5f, 3.5f};
    std::vector<int32_t> output(3);

    ExecuteCastOp(input.data(), output.data(),
                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                  3, "", cpu_mem_info_ptr);

    EXPECT_EQ(output[0], 1);
    EXPECT_EQ(output[1], 2);
    EXPECT_EQ(output[2], 3);
  }

  // Second execution with size 5 (should reuse cached session due to dynamic shape)
  {
    std::vector<float> input = {10.1f, 20.2f, 30.3f, 40.4f, 50.5f};
    std::vector<int32_t> output(5);

    ExecuteCastOp(input.data(), output.data(),
                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                  5, "", cpu_mem_info_ptr);

    EXPECT_EQ(output[0], 10);
    EXPECT_EQ(output[1], 20);
    EXPECT_EQ(output[2], 30);
    EXPECT_EQ(output[3], 40);
    EXPECT_EQ(output[4], 50);
  }

  OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseMemoryInfo(cpu_mem_info_ptr);
}

// Test 4: Cache clear functionality
TEST_F(OneOpModelTests, CacheClear) {
  OrtMemoryInfo* cpu_mem_info_ptr = nullptr;
  OrtGetApiBase()->GetApi(ORT_API_VERSION)->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &cpu_mem_info_ptr);

  std::vector<float> input = {1.5f};
  std::vector<int32_t> output(1);

  // Execute to populate cache
  ExecuteCastOp(input.data(), output.data(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                1, "", cpu_mem_info_ptr);

  EXPECT_EQ(output[0], 1);

  // Clear cache
  OneOpModelExecutor::ClearCache();

  // Execute again (should recreate session)
  ExecuteCastOp(input.data(), output.data(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                1, "", cpu_mem_info_ptr);

  EXPECT_EQ(output[0], 1);

  OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseMemoryInfo(cpu_mem_info_ptr);
}
