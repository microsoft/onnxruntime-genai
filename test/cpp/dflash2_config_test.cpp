// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "dflash2_drafter.h"

namespace Generators::test {
namespace {

Config MakeDflash2Config() {
  Config config;
  auto& decoder = config.model.decoder;
  decoder.filename = "target.onnx";
  decoder.sliding_window = Config::Model::Decoder::SlidingWindow{4096};
  decoder.state_groups.emplace();
  decoder.state_groups->push_back(Config::Model::Decoder::StateGroup{
      Config::Model::Decoder::StateGroupKind::Fixed});

  auto& dflash2 = config.model.dflash2;
  dflash2.filename = "dflash2.onnx";
  dflash2.num_hidden_layers = 3;
  dflash2.num_key_value_heads = 2;
  dflash2.head_size = 8;
  dflash2.block_size = 4;
  dflash2.num_draft_tokens = 3;
  dflash2.selector_top_k = 2;
  dflash2.sliding_window = 17;
  return config;
}

struct TensorMetadata {
  ONNXTensorElementDataType data_type;
  std::vector<int64_t> shape;
};

class FakeModelStateMetadata final : public ModelStateMetadata {
 public:
  void AddInput(std::string name, ONNXTensorElementDataType data_type,
                std::vector<int64_t> shape) {
    inputs_.insert_or_assign(
        std::move(name), TensorMetadata{data_type, std::move(shape)});
  }

  void AddOutput(std::string name, ONNXTensorElementDataType data_type,
                 std::vector<int64_t> shape) {
    outputs_.insert_or_assign(
        std::move(name), TensorMetadata{data_type, std::move(shape)});
  }

  bool HasInput(const std::string& name) const override { return inputs_.contains(name); }
  bool HasOutput(const std::string& name) const override { return outputs_.contains(name); }

  ONNXTensorElementDataType GetInputDataType(const std::string& name) const override {
    return inputs_.at(name).data_type;
  }

  ONNXTensorElementDataType GetOutputDataType(const std::string& name) const override {
    return outputs_.at(name).data_type;
  }

  std::vector<int64_t> GetInputShape(const std::string& name) const override {
    return inputs_.at(name).shape;
  }

  std::vector<int64_t> GetOutputShape(const std::string& name) const override {
    return outputs_.at(name).shape;
  }

 private:
  std::unordered_map<std::string, TensorMetadata> inputs_;
  std::unordered_map<std::string, TensorMetadata> outputs_;
};

std::pair<FakeModelStateMetadata, FakeModelStateMetadata> MakeCompatibleMetadata() {
  FakeModelStateMetadata target;
  target.AddOutput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 64});
  FakeModelStateMetadata drafter;
  drafter.AddInput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 64});
  return {std::move(target), std::move(drafter)};
}

}  // namespace

TEST(Dflash2ConfigTest, RequiresDrafterFilename) {
  auto config = MakeDflash2Config();
  config.model.dflash2.filename.clear();
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresCompleteGeometry) {
  auto config = MakeDflash2Config();
  config.model.dflash2.num_key_value_heads = 0;
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresPositiveSelectorTopK) {
  auto config = MakeDflash2Config();
  config.model.dflash2.selector_top_k = 0;
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);
}

TEST(Dflash2ConfigTest, RejectsAsynchronousExecution) {
  auto config = MakeDflash2Config();
  config.model.dflash2.run_options = Config::RunOptions{
      {"disable_synchronize_execution_providers", "1"}};
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);
}

TEST(Dflash2ConfigTest, AcceptsCompatibleAuxiliaryHiddenStates) {
  const auto config = MakeDflash2Config();
  const auto [target, drafter] = MakeCompatibleMetadata();
  EXPECT_NO_THROW(ValidateDflash2ModelCompatibility(config, target, drafter));
}

TEST(Dflash2ConfigTest, UsesConfiguredTargetOutput) {
  auto config = MakeDflash2Config();
  config.model.dflash2.main_aux_hidden_states = "custom_aux_hidden_states";
  FakeModelStateMetadata target;
  target.AddOutput("custom_aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 64});
  FakeModelStateMetadata drafter;
  drafter.AddInput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 64});
  EXPECT_NO_THROW(ValidateDflash2ModelCompatibility(config, target, drafter));
}

TEST(Dflash2ConfigTest, RequiresConfiguredTargetOutput) {
  const auto config = MakeDflash2Config();
  FakeModelStateMetadata target;
  FakeModelStateMetadata drafter;
  drafter.AddInput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 64});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresConfiguredDrafterInput) {
  const auto config = MakeDflash2Config();
  FakeModelStateMetadata target;
  target.AddOutput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 64});
  const FakeModelStateMetadata drafter;
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresTwoDimensionalAuxiliaryTensors) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  target.AddOutput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 4, 16});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter), std::runtime_error);

  std::tie(target, drafter) = MakeCompatibleMetadata();
  drafter.AddInput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 4, 16});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresMatchingAuxiliaryWidth) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.AddInput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 32});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresMatchingAuxiliaryType) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.AddInput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 64});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresOneDraftPerNonAnchorBlockRow) {
  auto config = MakeDflash2Config();
  config.model.dflash2.num_draft_tokens = 2;
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);
}

TEST(Dflash2ConfigTest, ProjectsDrafterWithoutTargetState) {
  const auto projected = CreateDflash2Config(MakeDflash2Config());
  const auto& decoder = projected->model.decoder;
  EXPECT_EQ(decoder.filename, "dflash2.onnx");
  EXPECT_EQ(decoder.num_hidden_layers, 3);
  EXPECT_EQ(decoder.num_key_value_heads, 2);
  EXPECT_EQ(decoder.head_size, 8);
  EXPECT_FALSE(decoder.sliding_window.has_value());
  EXPECT_FALSE(decoder.state_groups.has_value());
}

TEST(Dflash2ConfigTest, AccountsForWindowedPagedCache) {
  const auto config = MakeDflash2Config();
  EXPECT_EQ(Dflash2Drafter::BytesPerBlock(config, 16), 3072u);
  EXPECT_EQ(Dflash2Drafter::PoolBlocks(config, 8, 3), 15u);
}

TEST(Dflash2ConfigTest, RejectsUnboundedCachePool) {
  auto config = MakeDflash2Config();
  config.model.dflash2.sliding_window = 0;
  EXPECT_THROW(Dflash2Drafter::PoolBlocks(config, 8, 3), std::runtime_error);
}

}  // namespace Generators::test
