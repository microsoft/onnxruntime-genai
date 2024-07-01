// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <generators.h>
#include <models/model.h>
#include <models/lora_adapter.h>
#include <iostream>

namespace Generators {
namespace tests {

TEST(GeneratorsTests, LoraAdapterManagementTests) {
  const std::string adapter_name_1 = "adapter_1";
  const std::string adapter_name_2 = "adapter_2";

  LoraAdapterManagement lora_adapter_management;
  lora_adapter_management.CreateAdapter(adapter_name_1);
  // Try creating again should throw
  ASSERT_THROW(lora_adapter_management.CreateAdapter(adapter_name_1), std::runtime_error);

  lora_adapter_management.CreateAdapter(adapter_name_2);

  const std::array<int64_t, 2> lora_param_shape = {4, 2};
  // Generate random data for lora param according to the shape above
  std::array<float, 8> lora_param = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  auto mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  auto ort_value_param =
      OrtValue::CreateTensor(*mem_info, lora_param.data(), lora_param.size() * sizeof(float), lora_param_shape,
                             ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  auto param_tensor = std::make_shared<Generators::Tensor>();
  param_tensor->ort_tensor_ = std::move(ort_value_param);

  lora_adapter_management.AddParameter(adapter_name_1, "lora_param_1", param_tensor);
  lora_adapter_management.AddParameter(adapter_name_2, "lora_param_1", param_tensor);

  const std::string activate[] = {adapter_name_1};
  lora_adapter_management.ActivateAdapters(activate);

    // List active adapters
  auto active_names = lora_adapter_management.GetActiveAdapterNames();
  ASSERT_EQ(active_names.size(), 1U);
  ASSERT_EQ(active_names[0], adapter_name_1);

  // Can not remove active adapter
  ASSERT_THROW(lora_adapter_management.RemoveAdapter(adapter_name_1), std::runtime_error);


  // Deactivate two even though only one is active, no error.
  const std::string deactivate[] = {adapter_name_1, adapter_name_1};
  ASSERT_NO_THROW(lora_adapter_management.DeactiveAdapters(deactivate));
  active_names = lora_adapter_management.GetActiveAdapterNames();
  ASSERT_TRUE(active_names.empty());

  // No active adapters, no error reported.
  ASSERT_NO_THROW(lora_adapter_management.DeactiveAllAdapters());

  lora_adapter_management.RemoveAdapter(adapter_name_1);
  lora_adapter_management.RemoveAdapter(adapter_name_2);
}

}  // namespace tests
}  // namespace Generators