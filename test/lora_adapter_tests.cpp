// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <generators.h>
#include <models/model.h>
#include <models/lora_adapter.h>

#include <algorithm>
#include <cmath>
#include <iostream>

namespace Generators {
namespace tests {

// We do not initialize device specifics in this test.
 #if !defined(USE_DML)

TEST(GeneratorsTests, LoraAdapterContainerTests) {
  const std::string model_folder = MODEL_PATH "tiny-random-llama-lora";
  auto model = CreateModel(GetOrtEnv(), model_folder.c_str());

  LoraAdapterContainer& lora_adapter_container = model->GetLoraAdapterContainer();
  lora_adapter_container.LoadAdaptersFromConfig(model_folder, *model->config_);

  {
    // No adapters are active at this point
    std::set<std::string> active_adapters;
    // Fetch parameters and names, and make sure that all of the parameters returned are empty.
    std::vector<std::string> param_names;
    std::vector<std::shared_ptr<OrtValue>> params;
    OutputAdaptersParameters(*model, lora_adapter_container,
                             active_adapters, std::back_inserter(param_names),
                             std::back_inserter(params));

    ASSERT_EQ(param_names.size(), 28U);
    ASSERT_EQ(params.size(), 28U);

    // No active adapters, all params are empty
    for (auto& ort_val : params) {
      auto val_type_shape = ort_val->GetTensorTypeAndShapeInfo();
      ASSERT_EQ(val_type_shape->GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      auto shape = val_type_shape->GetShape();
      ASSERT_EQ(shape.size(), 2U);
      ASSERT_TRUE(shape[0] != 0 || shape[1] != 0);
      ASSERT_TRUE(shape[0] == 0 || shape[1] == 0);
    }
  }

  {
    const std::set<std::string> active_adapters = {"guanaco"};
    std::vector<std::string> param_names;
    std::vector<std::shared_ptr<OrtValue>> params;
    OutputAdaptersParameters(*model, lora_adapter_container, active_adapters,
                             std::back_inserter(param_names),
                             std::back_inserter(params));

    ASSERT_EQ(param_names.size(), 28U);
    ASSERT_EQ(params.size(), 28U);

    for (size_t i = 0, lim = params.size(); i < lim; ++i) {
      auto& ort_val = params[i];
      auto val_type_shape = ort_val->GetTensorTypeAndShapeInfo();
      ASSERT_EQ(val_type_shape->GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      auto shape = val_type_shape->GetShape();
      ASSERT_EQ(shape.size(), 2U);

      // Let's make sure we can read all the data
      const auto element_num = val_type_shape->GetElementCount();
      const auto* data = ort_val->GetTensorData<float>();
      for (size_t j = 0; j < element_num; ++j) {
        // Do some silly op
        ASSERT_TRUE(std::isfinite(data[j]));
      }

    }
  }
}

#endif

}  // namespace tests
}  // namespace Generators
