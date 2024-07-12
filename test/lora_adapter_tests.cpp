// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <generators.h>
#include <models/model.h>
#include <models/lora_adapter.h>

#include <algorithm>
#include <iostream>

namespace Generators {
namespace tests {

TEST(GeneratorsTests, LoraAdapterManagementTests) {
  const std::string adapter_name_1 = "adapter_1";
  const std::string adapter_name_2 = "adapter_2";

  LoraAdapterManagement lora_adapter_management(nullptr);
  lora_adapter_management.CreateAdapter(adapter_name_1);
  // Try creating again should throw
  ASSERT_THROW(lora_adapter_management.CreateAdapter(adapter_name_1), std::runtime_error);

  lora_adapter_management.CreateAdapter(adapter_name_2);

  // Two shapes with different lora_r placements
  const std::array<int64_t, 2> lora_param_shape_1 = {4, 2};
  const std::array<int64_t, 2> lora_param_shape_2 = {2, 4};

  // Lora parameter data
  std::array<float, 8> lora_param = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  auto mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  auto ort_value_param_1 =
      OrtValue::CreateTensor(*mem_info, lora_param.data(), lora_param.size() * sizeof(float), lora_param_shape_1,
                             ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  auto param_tensor_1 = std::make_shared<Generators::Tensor>();
  param_tensor_1->ort_tensor_ = std::move(ort_value_param_1);

  lora_adapter_management.AddParameter(adapter_name_1, "lora_param_1", param_tensor_1);

  auto ort_value_param_2 =
      OrtValue::CreateTensor(*mem_info, lora_param.data(), lora_param.size() * sizeof(float), lora_param_shape_2,
                             ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  auto param_tensor_2 = std::make_shared<Generators::Tensor>();
  param_tensor_2->ort_tensor_ = std::move(ort_value_param_2);

  lora_adapter_management.AddParameter(adapter_name_2, "lora_param_2", param_tensor_2);

  {
    // No adapters are active at this point
    std::set<std::string> active_adapters;
    // Fetch parameters and names, and make sure that all of the parameters returned are empty.
    std::vector<std::string> param_names;
    std::vector<std::shared_ptr<OrtValue>> params;
    lora_adapter_management.OutputAdaptersParameters(active_adapters, std::back_inserter(param_names),
                                                     std::back_inserter(params));

    ASSERT_EQ(param_names.size(), 2U);
    ASSERT_EQ(params.size(), 2U);

    for (auto& ort_val : params) {
      auto val_type_shape = ort_val->GetTensorTypeAndShapeInfo();
      ASSERT_EQ(val_type_shape->GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      auto shape = val_type_shape->GetShape();
      ASSERT_EQ(shape.size(), 2U);
      if (shape[0] == 4) {
        ASSERT_EQ(shape[1], 0);
      } else {
        ASSERT_EQ(shape[0], 0);
      }
    }
  }

  {
    // One parameter would be empty, another is not
    const std::set<std::string> active_adapters = {adapter_name_1};
    std::vector<std::string> param_names;
    std::vector<std::shared_ptr<OrtValue>> params;
    lora_adapter_management.OutputAdaptersParameters(active_adapters, std::back_inserter(param_names),
                                                     std::back_inserter(params));

    ASSERT_EQ(param_names.size(), 2U);
    ASSERT_EQ(params.size(), 2U);

    for (size_t i = 0; i < params.size(); ++i) {
      auto& ort_val = params[i];
      auto val_type_shape = ort_val->GetTensorTypeAndShapeInfo();
      ASSERT_EQ(val_type_shape->GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      auto shape = val_type_shape->GetShape();
      ASSERT_EQ(shape.size(), 2U);

      if (param_names[i] == "lora_param_1") {
        ASSERT_EQ(shape[0], 4);
        ASSERT_EQ(shape[1], 2);
      } else {
        // For inactive params shape must contain 1 zero
        if (shape[0] == 4) {
          ASSERT_EQ(shape[1], 0);
        } else {
          ASSERT_EQ(shape[0], 0);
        }
      }
    }
  }

  lora_adapter_management.RemoveAdapter(adapter_name_1);
  lora_adapter_management.RemoveAdapter(adapter_name_2);
}

}  // namespace tests
}  // namespace Generators