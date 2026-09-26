// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine_test_helpers.h"
#include "models/model_state_manifest.h"

namespace Generators::test {
namespace {

struct TensorMetadata {
  ONNXTensorElementDataType data_type;
  std::vector<int64_t> shape;
};

class FakeModelStateMetadata final : public ModelStateMetadata {
 public:
  void AddInput(std::string name,
                ONNXTensorElementDataType data_type,
                std::vector<int64_t> shape) {
    inputs_.insert_or_assign(
        std::move(name),
        TensorMetadata{data_type, std::move(shape)});
  }

  void AddOutput(std::string name,
                 ONNXTensorElementDataType data_type,
                 std::vector<int64_t> shape) {
    outputs_.insert_or_assign(
        std::move(name),
        TensorMetadata{data_type, std::move(shape)});
  }

  bool HasInput(const std::string& name) const override {
    return inputs_.contains(name);
  }

  bool HasOutput(const std::string& name) const override {
    return outputs_.contains(name);
  }

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

Config::Model::Decoder MakeSparseDecoder() {
  using Decoder = Config::Model::Decoder;

  Decoder decoder;
  decoder.num_hidden_layers = 4;
  decoder.inputs.past_key_names = "past.%d.key";
  decoder.inputs.past_value_names = "past.%d.value";
  decoder.inputs.past_conv_names = "past.%d.conv";
  decoder.inputs.past_recurrent_names = "past.%d.recurrent";
  decoder.outputs.present_key_names = "present.%d.key";
  decoder.outputs.present_value_names = "present.%d.value";
  decoder.outputs.present_conv_names = "present.%d.conv";
  decoder.outputs.present_recurrent_names = "present.%d.recurrent";
  decoder.state_groups = std::vector<Decoder::StateGroup>{
      Decoder::StateGroup{
          Decoder::StateGroupKind::PagedKeyValue,
          {1, 3}},
      Decoder::StateGroup{
          Decoder::StateGroupKind::FixedConv,
          {0, 2}},
      Decoder::StateGroup{
          Decoder::StateGroupKind::FixedRecurrent,
          {0, 2}}};
  return decoder;
}

FakeModelStateMetadata MakeValidMetadata() {
  FakeModelStateMetadata metadata;
  for (const int layer_id : {1, 3}) {
    for (const auto* semantic : {"key", "value"}) {
      metadata.AddInput(
          "past." + std::to_string(layer_id) + "." + semantic,
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
          {-1, 256, 4, 128});
      metadata.AddOutput(
          "present." + std::to_string(layer_id) + "." + semantic,
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
          {-1, 256, 4, 128});
    }
  }
  for (const int layer_id : {0, 2}) {
    metadata.AddInput(
        "past." + std::to_string(layer_id) + ".conv",
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
        {-1, 8192, 3});
    metadata.AddOutput(
        "present." + std::to_string(layer_id) + ".conv",
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
        {-1, 8192, 3});
    metadata.AddInput(
        "past." + std::to_string(layer_id) + ".recurrent",
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        {-1, 16, 128, 128});
    metadata.AddOutput(
        "present." + std::to_string(layer_id) + ".recurrent",
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        {-1, 16, 128, 128});
  }
  return metadata;
}

Config::Model::Decoder MakePleAndIndexerDecoder() {
  using Decoder = Config::Model::Decoder;
  Decoder decoder;
  decoder.num_hidden_layers = 2;
  decoder.inputs.past_ple_token_names = "past.%d.ple_tokens";
  decoder.inputs.past_ple_conv_names = "past.%d.ple_conv";
  decoder.inputs.past_indexer_names = "past.%d.indexer_key";
  decoder.inputs.past_indexer_kv_buffer_names = "past.%d.indexer_kv_buffer";
  decoder.inputs.past_indexer_state_lengths_names = "past.%d.indexer_state_lengths";
  decoder.outputs.present_ple_token_names = "present.%d.ple_tokens";
  decoder.outputs.present_ple_conv_names = "present.%d.ple_conv";
  decoder.outputs.present_indexer_names = "present.%d.indexer_key";
  decoder.outputs.present_indexer_kv_buffer_names = "present.%d.indexer_kv_buffer";
  decoder.outputs.present_indexer_state_lengths_names = "present.%d.indexer_state_lengths";
  decoder.state_groups = std::vector<Decoder::StateGroup>{
      Decoder::StateGroup{Decoder::StateGroupKind::FixedPle, {0}},
      Decoder::StateGroup{Decoder::StateGroupKind::FixedIndexer, {1}}};
  return decoder;
}

FakeModelStateMetadata MakePleAndIndexerMetadata() {
  FakeModelStateMetadata metadata;
  const auto add_pair = [&](const std::string& input, const std::string& output,
                            ONNXTensorElementDataType type, std::vector<int64_t> shape) {
    metadata.AddInput(input, type, shape);
    metadata.AddOutput(output, type, std::move(shape));
  };
  add_pair("past.0.ple_tokens", "present.0.ple_tokens",
           ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {-1, 2});
  add_pair("past.0.ple_conv", "present.0.ple_conv",
           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 16, 9});
  add_pair("past.1.indexer_key", "present.1.indexer_key",
           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 1024, 128});
  add_pair("past.1.indexer_kv_buffer", "present.1.indexer_kv_buffer",
           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 7, 128});
  add_pair("past.1.indexer_state_lengths", "present.1.indexer_state_lengths",
           ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1, 2});
  return metadata;
}

std::string CaptureValidationError(const ModelStateManifest& manifest,
                                   const ModelStateMetadata& metadata) {
  try {
    manifest.ValidateSession(metadata);
  } catch (const std::runtime_error& error) {
    return error.what();
  }
  return {};
}

TEST(ModelStateManifestTest, ReportsStateGroupCapabilities) {
  const ModelStateManifest hybrid{MakeSparseDecoder()};
  EXPECT_TRUE(hybrid.HasStateGroupKind(
      Config::Model::Decoder::StateGroupKind::PagedKeyValue));
  EXPECT_TRUE(hybrid.HasStateGroupKind(
      Config::Model::Decoder::StateGroupKind::FixedConv));
  EXPECT_TRUE(hybrid.HasStateGroupKind(
      Config::Model::Decoder::StateGroupKind::FixedRecurrent));
  EXPECT_TRUE(hybrid.HasFixedStateGroups());

  Config::Model::Decoder legacy;
  legacy.num_hidden_layers = 4;
  const ModelStateManifest dense{legacy};
  EXPECT_FALSE(dense.HasStateGroupKind(
      Config::Model::Decoder::StateGroupKind::PagedKeyValue));
  EXPECT_FALSE(dense.HasFixedStateGroups());
}

TEST(ModelStateManifestTest, ValidatesEveryExpandedBinding) {
  const ModelStateManifest manifest{MakeSparseDecoder()};
  const auto metadata = MakeValidMetadata();

  EXPECT_NO_THROW(manifest.ValidateSession(metadata));
}

TEST(ModelStateManifestTest, ValidatesPleAndPackedIndexerBindings) {
  const ModelStateManifest manifest{MakePleAndIndexerDecoder()};
  EXPECT_TRUE(manifest.HasFixedStateGroups());
  EXPECT_TRUE(manifest.HasStateGroupKind(
      Config::Model::Decoder::StateGroupKind::FixedPle));
  EXPECT_TRUE(manifest.HasStateGroupKind(
      Config::Model::Decoder::StateGroupKind::FixedIndexer));
  EXPECT_NO_THROW(manifest.ValidateSession(MakePleAndIndexerMetadata()));
}

TEST(ModelStateManifestTest, RejectsMissingPackedIndexerComponent) {
  const ModelStateManifest manifest{MakePleAndIndexerDecoder()};
  FakeModelStateMetadata metadata;
  const auto message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("past.0.ple_tokens"), std::string::npos) << message;
}

TEST(ModelStateManifestTest, RejectsStateUpdatesForPleAndIndexerGroups) {
  auto decoder = MakePleAndIndexerDecoder();
  decoder.state_groups->front().state_update = Config::Model::Decoder::StateUpdate{3};
  EXPECT_THROW(ModelStateManifest{decoder}, std::runtime_error);
}

TEST(ModelStateManifestTest, RejectsMissingBinding) {
  const ModelStateManifest manifest{MakeSparseDecoder()};
  const FakeModelStateMetadata metadata;

  const auto message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("input was not found: past.1.key"), std::string::npos) << message;
}

TEST(ModelStateManifestTest, RejectsInputOutputDtypeMismatch) {
  const ModelStateManifest manifest{MakeSparseDecoder()};
  auto metadata = MakeValidMetadata();
  metadata.AddOutput(
      "present.1.key",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      {-1, 256, 4, 128});

  const auto message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("incompatible dtypes"), std::string::npos) << message;
}

TEST(ModelStateManifestTest, RejectsInputOutputShapeMismatch) {
  const ModelStateManifest manifest{MakeSparseDecoder()};
  auto metadata = MakeValidMetadata();
  metadata.AddOutput(
      "present.1.key",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
      {-1, 256, 4, 64});

  const auto message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("incompatible shapes"), std::string::npos) << message;
}

TEST(ModelStateManifestTest, AllowsDynamicDimensions) {
  const ModelStateManifest manifest{MakeSparseDecoder()};
  auto metadata = MakeValidMetadata();
  metadata.AddInput(
      "past.1.key",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
      {-1, -1, 4, 128});
  metadata.AddOutput(
      "present.1.key",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
      {-1, -1, 4, 128});

  EXPECT_NO_THROW(manifest.ValidateSession(metadata));
}

TEST(ModelStateManifestTest, RejectsIncompatiblePagedGeometry) {
  const ModelStateManifest manifest{MakeSparseDecoder()};
  auto metadata = MakeValidMetadata();
  metadata.AddInput(
      "past.1.value",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
      {-1, 256, 4, 64});
  metadata.AddOutput(
      "present.1.value",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
      {-1, 256, 4, 64});

  const auto message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("incompatible paged geometry"), std::string::npos) << message;
}

TEST(ModelStateManifestTest, RejectsMissingFixedConvDecoderBinding) {
  auto decoder = MakeSparseDecoder();
  decoder.inputs.past_conv_names = "missing.%d.conv";
  const ModelStateManifest manifest{decoder};
  const auto metadata = MakeValidMetadata();

  const auto message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("input was not found: missing.0.conv"), std::string::npos) << message;
}

TEST(ModelStateManifestTest, DecoderModelLoadValidatesDecoderBindings) {
  const auto model_path = fs::path{std::string{MODEL_PATH "engine/dummy-decoder"}};
  const auto invalid_overlay = R"({
    "model": {"decoder": {
      "outputs": {"present_key_names": "missing.%d.key"},
      "state_groups": [{
        "kind": "paged_kv", "layer_ids": [0]
      }]
    }}
  })";
  auto invalid_config = std::make_unique<Config>(model_path, invalid_overlay);
  EXPECT_THROW(CreateModel(GetOrtEnv(), std::move(invalid_config)), std::runtime_error);
}

TEST(ModelStateManifestTest, DecoderModelLoadsWithValidDecoderBindings) {
  const auto model_path = fs::path{std::string{MODEL_PATH "engine/dummy-decoder"}};
  const auto valid_overlay = R"({
    "model": {"decoder": {
      "state_groups": [{
        "kind": "paged_kv", "layer_ids": [0]
      }]
    }}
  })";
  auto valid_config = std::make_unique<Config>(model_path, valid_overlay);
  EXPECT_NO_THROW(CreateModel(GetOrtEnv(), std::move(valid_config)));
}

TEST(ModelStateManifestTest, AcceptsFixedDynamicEngineContract) {
  // Packed hybrid IO binds fixed tensors at the same boundary where compatibility is enabled.
  auto decoder = MakeSparseDecoder();
  EXPECT_NO_THROW(
      ModelStateManifest::ValidateDynamicEngineCompatibility(decoder));
}

TEST(ModelStateManifestTest, RejectsFixedGroupsWithoutPagedGroup) {
  // Enabling fixed groups must not weaken the "exactly one paged_kv group" rule: a decoder that
  // declares only fixed groups still has no paged pool to run through and is rejected.
  auto decoder = MakeSparseDecoder();
  decoder.state_groups->erase(
      std::remove_if(decoder.state_groups->begin(), decoder.state_groups->end(),
                     [](const auto& group) {
                       return group.kind ==
                              Config::Model::Decoder::StateGroupKind::PagedKeyValue;
                     }),
      decoder.state_groups->end());
  try {
    ModelStateManifest::ValidateDynamicEngineCompatibility(decoder);
    FAIL() << "Expected a decoder without a paged_kv group to be rejected";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string{error.what()}.find("paged_kv"), std::string::npos) << error.what();
  }
}

TEST(ModelStateManifestTest, AcceptsSparsePagedDynamicEngineContract) {
  auto decoder = MakeSparseDecoder();
  decoder.state_groups->erase(
      std::remove_if(decoder.state_groups->begin(), decoder.state_groups->end(),
                     [](const auto& group) {
                       return group.kind == Config::Model::Decoder::StateGroupKind::FixedConv ||
                              group.kind == Config::Model::Decoder::StateGroupKind::FixedRecurrent;
                     }),
      decoder.state_groups->end());

  EXPECT_NO_THROW(ModelStateManifest::ValidateDynamicEngineCompatibility(decoder));
}

TEST(ModelStateManifestTest, AcceptsLegacyDynamicEngineContract) {
  Config::Model::Decoder decoder;
  decoder.num_hidden_layers = 4;

  EXPECT_NO_THROW(ModelStateManifest::ValidateDynamicEngineCompatibility(decoder));
}

}  // namespace
}  // namespace Generators::test
