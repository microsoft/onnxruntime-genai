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
#if !defined(USE_CUDA) && !defined(USE_DML)

class TestState : public State {
 public:
  TestState(const GeneratorParams& params, const Model& model) : State(params, model) {}

  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override {
    return RoamingArray<float>();
  }
};

class TestModel : public Model {
 public:
  TestModel(std::unique_ptr<Config> config) : Model(std::move(config)) {
    this->allocator_device_ = &this->allocator_cpu_;
  }

  std::unique_ptr<State> CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) const override {
    return std::make_unique<TestState>(params, *this);
  }
};

TEST(GeneratorsTests, LoraAdapterContainerTests) {
  const std::string model_folder = MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32-lora";
  auto model = std::make_unique<TestModel>(std::make_unique<Config>(model_folder));

  LoraAdapterContainer lora_adapter_container;
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

    ASSERT_EQ(param_names.size(), 2U);
    ASSERT_EQ(params.size(), 2U);

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

    ASSERT_EQ(param_names.size(), 2U);
    ASSERT_EQ(params.size(), 2U);

    for (size_t i = 0; i < params.size(); ++i) {
      auto& ort_val = params[i];
      auto val_type_shape = ort_val->GetTensorTypeAndShapeInfo();
      ASSERT_EQ(val_type_shape->GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      auto shape = val_type_shape->GetShape();
      ASSERT_EQ(shape.size(), 2U);

      if (param_names[i] == "model.layers.0.attn.qkv_proj.lora_A.weight") {
        ASSERT_EQ(shape[0], 3072);
        ASSERT_EQ(shape[1], 64);
      } else {
        ASSERT_EQ(shape[0], 64);
        ASSERT_EQ(shape[1], 9216);
      }

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
