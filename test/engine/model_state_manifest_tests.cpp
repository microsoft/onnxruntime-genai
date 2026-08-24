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

  void RemoveOutput(const std::string& name) {
    outputs_.erase(name);
  }

  void RemoveInput(const std::string& name) {
    inputs_.erase(name);
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
  decoder.state_groups = std::vector<Decoder::StateGroup>{
      Decoder::StateGroup{
          Decoder::StateGroupKind::PagedKeyValue,
          {1, 3},
          Decoder::StateBinding{"past.%d.key", "present.%d.key"},
          Decoder::StateBinding{"past.%d.value", "present.%d.value"},
          std::nullopt},
      Decoder::StateGroup{
          Decoder::StateGroupKind::Fixed,
          {0, 2},
          std::nullopt,
          std::nullopt,
          Decoder::StateBinding{"past.%d.conv", "present.%d.conv"}}};
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
  }
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

// The speculative-rollback variant of the fixed group: the committed state keeps its shape and
// the per-token series arrives through a separate output with one leading slot axis.
Config::Model::Decoder MakeCheckpointDecoder(int checkpoint_count = 4) {
  auto decoder = MakeSparseDecoder();
  auto& fixed = decoder.state_groups->back();
  fixed.state->checkpoints = "checkpoints.%d.conv";
  fixed.checkpoint_count = checkpoint_count;
  fixed.checkpoint_alignment = Config::Model::Decoder::CheckpointAlignment::Left;
  return decoder;
}

FakeModelStateMetadata MakeCheckpointMetadata(int checkpoint_count = 4) {
  auto metadata = MakeValidMetadata();
  for (const int layer_id : {0, 2}) {
    metadata.AddOutput(
        "checkpoints." + std::to_string(layer_id) + ".conv",
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
        {checkpoint_count, -1, 8192, 3});
  }
  return metadata;
}

Config::Model::Decoder MakeCompactConvDecoder() {
  using Decoder = Config::Model::Decoder;
  auto decoder = MakeCheckpointDecoder();
  auto& fixed = decoder.state_groups->back();
  fixed.state_update = Decoder::StateUpdate{
      Decoder::StateUpdateKind::CausalConv,
      3,
      "state_update_capture_count",
      "state_update.%d.conv_value"};
  return decoder;
}

FakeModelStateMetadata MakeCompactConvMetadata() {
  auto metadata = MakeCheckpointMetadata();
  metadata.AddInput(
      "state_update_capture_count",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
      {-1});
  for (const int layer_id : {0, 2}) {
    metadata.AddOutput(
        "state_update." + std::to_string(layer_id) + ".conv_value",
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
        {-1, 3, 8192});
  }
  return metadata;
}

Config::Model::Decoder MakeCompactGdnDecoder() {
  using Decoder = Config::Model::Decoder;
  Decoder decoder;
  decoder.num_hidden_layers = 1;
  Decoder::StateGroup group;
  group.kind = Decoder::StateGroupKind::Fixed;
  group.layer_ids = {0};
  group.state = Decoder::StateBinding{"past.%d.recurrent", "present.%d.recurrent"};
  group.state_update = Decoder::StateUpdate{
      Decoder::StateUpdateKind::GatedDeltaNet,
      3,
      "state_update_capture_count",
      "",
      "state_update.%d.decay",
      "state_update.%d.key",
      "state_update.%d.delta"};
  decoder.state_groups = std::vector<Decoder::StateGroup>{std::move(group)};
  return decoder;
}

FakeModelStateMetadata MakeCompactGdnMetadata() {
  FakeModelStateMetadata metadata;
  metadata.AddInput(
      "past.0.recurrent",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      {-1, 48, 128, 128});
  metadata.AddOutput(
      "present.0.recurrent",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      {-1, 48, 128, 128});
  metadata.AddInput(
      "state_update_capture_count",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
      {-1});
  metadata.AddOutput(
      "state_update.0.decay",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      {-1, 3, 48});
  metadata.AddOutput(
      "state_update.0.key",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      {-1, 3, 16, 128});
  metadata.AddOutput(
      "state_update.0.delta",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      {-1, 3, 48, 128});
  return metadata;
}

TEST(ModelStateManifestTest, ValidatesCheckpointOutputs) {
  const ModelStateManifest manifest{MakeCheckpointDecoder()};

  EXPECT_NO_THROW(manifest.ValidateSession(MakeCheckpointMetadata()));
}

TEST(ModelStateManifestTest, RejectsMissingCheckpointOutput) {
  const ModelStateManifest manifest{MakeCheckpointDecoder()};

  const auto message = CaptureValidationError(manifest, MakeValidMetadata());
  EXPECT_NE(message.find("checkpoints output was not found: checkpoints.0.conv"),
            std::string::npos)
      << message;
}

TEST(ModelStateManifestTest, RejectsCheckpointCountMismatch) {
  const ModelStateManifest manifest{MakeCheckpointDecoder(4)};

  const auto message = CaptureValidationError(manifest, MakeCheckpointMetadata(3));
  EXPECT_NE(message.find("must be [4, ...]"), std::string::npos) << message;
}

TEST(ModelStateManifestTest, RejectsCheckpointRankMismatch) {
  const ModelStateManifest manifest{MakeCheckpointDecoder()};
  auto metadata = MakeCheckpointMetadata();
  metadata.AddOutput(
      "checkpoints.0.conv",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
      {-1, 8192, 3});

  const auto message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("must be [4, ...]"), std::string::npos) << message;
}

TEST(ModelStateManifestTest, RejectsCheckpointCountWithoutTemplate) {
  auto decoder = MakeSparseDecoder();
  decoder.state_groups->back().checkpoint_count = 4;

  EXPECT_THROW(ModelStateManifest{decoder}, std::runtime_error);
}

TEST(ModelStateManifestTest, RejectsCheckpointTemplateWithoutCount) {
  auto decoder = MakeSparseDecoder();
  decoder.state_groups->back().state->checkpoints = "checkpoints.%d.conv";

  EXPECT_THROW(ModelStateManifest{decoder}, std::runtime_error);
}

TEST(ModelStateManifestTest, RejectsCheckpointsOnPagedGroup) {
  auto decoder = MakeSparseDecoder();
  auto& paged = decoder.state_groups->front();
  paged.checkpoint_count = 4;
  paged.key->checkpoints = "checkpoints.%d.key";

  EXPECT_THROW(ModelStateManifest{decoder}, std::runtime_error);
}

TEST(ModelStateManifestTest, RejectsCheckpointTemplateColliding) {
  auto decoder = MakeCheckpointDecoder();
  decoder.state_groups->back().state->checkpoints = "present.%d.conv";

  EXPECT_THROW(ModelStateManifest{decoder}, std::runtime_error);
}

TEST(ModelStateManifestTest, ValidatesCompactConvUpdatesAlongsideCheckpoints) {
  const ModelStateManifest manifest{MakeCompactConvDecoder()};

  EXPECT_NO_THROW(manifest.ValidateSession(MakeCompactConvMetadata()));
}

TEST(ModelStateManifestTest, RejectsPartialCompactStateUpdateCoverage) {
  auto decoder = MakeCompactConvDecoder();
  auto fixed_without_update = decoder.state_groups->back();
  fixed_without_update.layer_ids = {3};
  fixed_without_update.state_update.reset();
  decoder.state_groups->push_back(std::move(fixed_without_update));

  EXPECT_THROW(ModelStateManifest{decoder}, std::runtime_error);
}

TEST(ModelStateManifestTest, ValidatesCompactGdnUpdatesWithDynamicBatch) {
  const ModelStateManifest manifest{MakeCompactGdnDecoder()};

  EXPECT_NO_THROW(manifest.ValidateSession(MakeCompactGdnMetadata()));
}

TEST(ModelStateManifestTest, ValidatesCompactStateUpdateActivityInput) {
  auto decoder = MakeCompactGdnDecoder();
  decoder.state_groups->front().state_update->active = "state_update_active";
  const ModelStateManifest manifest{decoder};
  auto metadata = MakeCompactGdnMetadata();
  metadata.AddInput(
      "state_update_active",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
      {1});
  EXPECT_NO_THROW(manifest.ValidateSession(metadata));

  metadata.AddInput(
      "state_update_active",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
      {1});
  auto message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("active input 'state_update_active' must have int32 dtype"),
            std::string::npos)
      << message;

  metadata.AddInput(
      "state_update_active",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
      {2});
  message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("must have shape [1]"), std::string::npos) << message;
}

TEST(ModelStateManifestTest, RejectsInvalidCompactCaptureCountMetadata) {
  const ModelStateManifest manifest{MakeCompactConvDecoder()};
  auto metadata = MakeCompactConvMetadata();
  metadata.RemoveInput("state_update_capture_count");
  auto message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("capture_count input was not found"), std::string::npos) << message;

  metadata = MakeCompactConvMetadata();
  metadata.AddInput(
      "state_update_capture_count",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
      {-1});
  message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("must have int32 dtype"), std::string::npos) << message;

  metadata.AddInput(
      "state_update_capture_count",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
      {-1, 1});
  message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("must have rank 1"), std::string::npos) << message;

  metadata.AddInput(
      "state_update_capture_count",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
      {2});
  metadata.AddOutput(
      "present.0.conv",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
      {3, 8192, 3});
  message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("batch dimension incompatible"), std::string::npos) << message;
}

TEST(ModelStateManifestTest, RejectsMissingCompactStateUpdateOutput) {
  const ModelStateManifest manifest{MakeCompactGdnDecoder()};
  auto metadata = MakeCompactGdnMetadata();
  metadata.RemoveOutput("state_update.0.key");

  const auto message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("key output was not found: state_update.0.key"),
            std::string::npos)
      << message;
}

TEST(ModelStateManifestTest, RejectsInvalidCompactConvUpdateMetadata) {
  const ModelStateManifest manifest{MakeCompactConvDecoder()};
  auto metadata = MakeCompactConvMetadata();
  metadata.AddOutput(
      "state_update.0.conv_value",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      {-1, 3, 8192});
  auto message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("same dtype as state output"), std::string::npos) << message;

  metadata.AddOutput(
      "state_update.0.conv_value",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
      {-1, 2, 8192});
  message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("capacity dimension must be 3"), std::string::npos) << message;

  metadata.AddOutput(
      "state_update.0.conv_value",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
      {-1, 3, 4096});
  message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("channel dimensions are incompatible"), std::string::npos) << message;
}

TEST(ModelStateManifestTest, RejectsInvalidCompactGdnDtypeAndShapeMetadata) {
  const ModelStateManifest manifest{MakeCompactGdnDecoder()};
  auto metadata = MakeCompactGdnMetadata();
  metadata.AddOutput(
      "state_update.0.decay",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
      {-1, 3, 48});
  auto message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("must have float dtype"), std::string::npos) << message;

  metadata.AddOutput(
      "state_update.0.decay",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      {-1, 3, 48});
  metadata.AddOutput(
      "state_update.0.key",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      {-1, 3, 0, 128});
  message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("positive key-head dimension"), std::string::npos) << message;

  metadata.AddOutput(
      "state_update.0.key",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      {-1, 3, 16, 64});
  message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("key dimensions are incompatible"), std::string::npos) << message;

  metadata.AddOutput(
      "state_update.0.key",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      {-1, 3, 16, 128});
  metadata.AddOutput(
      "state_update.0.delta",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      {-1, 3, 24, 128});
  message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("value-head dimensions are incompatible"), std::string::npos) << message;

  metadata.AddOutput(
      "state_update.0.delta",
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      {-1, 3, 48, 64});
  message = CaptureValidationError(manifest, metadata);
  EXPECT_NE(message.find("value dimensions are incompatible"), std::string::npos) << message;
}

TEST(ModelStateManifestTest, ReportsStateGroupCapabilities) {
  const ModelStateManifest hybrid{MakeSparseDecoder()};
  EXPECT_TRUE(hybrid.HasStateGroupKind(
      Config::Model::Decoder::StateGroupKind::PagedKeyValue));
  EXPECT_TRUE(hybrid.HasStateGroupKind(
      Config::Model::Decoder::StateGroupKind::Fixed));
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

TEST(ModelStateManifestTest, DecoderModelLoadValidatesExplicitBindings) {
  const auto model_path = fs::path{std::string{MODEL_PATH "engine/dummy-decoder"}};
  const auto invalid_overlay = R"({
    "model": {"decoder": {"state_groups": [{
      "kind": "paged_kv",
      "layer_ids": [0],
      "bindings": {
        "key": {"input": "past_key_values.%d.key", "output": "missing.%d.key"},
        "value": {"input": "past_key_values.%d.value", "output": "present.%d.value"}
      }
    }]}}
  })";
  auto invalid_config = std::make_unique<Config>(model_path, invalid_overlay);
  EXPECT_THROW(CreateModel(GetOrtEnv(), std::move(invalid_config)), std::runtime_error);
}

TEST(ModelStateManifestTest, DecoderModelLoadsWithValidExplicitBindings) {
  const auto model_path = fs::path{std::string{MODEL_PATH "engine/dummy-decoder"}};
  const auto valid_overlay = R"({
    "model": {"decoder": {"state_groups": [{
      "kind": "paged_kv",
      "layer_ids": [0],
      "bindings": {
        "key": {"input": "past_key_values.%d.key", "output": "present.%d.key"},
        "value": {"input": "past_key_values.%d.value", "output": "present.%d.value"}
      }
    }]}}
  })";
  auto valid_config = std::make_unique<Config>(model_path, valid_overlay);
  EXPECT_NO_THROW(CreateModel(GetOrtEnv(), std::move(valid_config)));
}

TEST(ModelStateManifestTest, ParsesCheckpointStateGroupFields) {
  const auto model_path = fs::path{std::string{MODEL_PATH "engine/dummy-decoder"}};
  const auto overlay = R"({
    "model": {"decoder": {"state_groups": [{
      "kind": "fixed",
      "layer_ids": [0],
      "checkpoint_count": 4,
      "checkpoint_alignment": "left",
      "bindings": {
        "state": {
          "input": "past_key_values.%d.conv",
          "output": "present.%d.conv",
          "checkpoints": "checkpoints.%d.conv"
        }
      }
    }]}}
  })";
  const Config config{model_path, overlay};

  ASSERT_TRUE(config.model.decoder.state_groups.has_value());
  ASSERT_EQ(config.model.decoder.state_groups->size(), 1u);
  const auto& group = config.model.decoder.state_groups->front();
  EXPECT_EQ(group.checkpoint_count, 4);
  EXPECT_EQ(group.checkpoint_alignment, Config::Model::Decoder::CheckpointAlignment::Left);
  ASSERT_TRUE(group.state.has_value());
  EXPECT_EQ(group.state->checkpoints, "checkpoints.%d.conv");
}

TEST(ModelStateManifestTest, RejectsOutOfRangeCheckpointCount) {
  const auto model_path = fs::path{std::string{MODEL_PATH "engine/dummy-decoder"}};
  const auto overlay = R"({
    "model": {"decoder": {"state_groups": [{
      "kind": "fixed",
      "layer_ids": [0],
      "checkpoint_count": 9,
      "bindings": {
        "state": {
          "input": "past_key_values.%d.conv",
          "output": "present.%d.conv",
          "checkpoints": "checkpoints.%d.conv"
        }
      }
    }]}}
  })";
  EXPECT_THROW((Config{model_path, overlay}), std::runtime_error);
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
                       return group.kind == Config::Model::Decoder::StateGroupKind::Fixed;
                     }),
      decoder.state_groups->end());

  EXPECT_NO_THROW(
      ModelStateManifest::ValidateDynamicEngineCompatibility(decoder));
}

TEST(ModelStateManifestTest, AcceptsDynamicEngineBindingsThatDifferFromLegacyNames) {
  using Decoder = Config::Model::Decoder;
  Decoder decoder;
  decoder.num_hidden_layers = 1;
  decoder.state_groups = std::vector<Decoder::StateGroup>{
      Decoder::StateGroup{
          Decoder::StateGroupKind::PagedKeyValue,
          {0},
          Decoder::StateBinding{"custom.%d.key", "present.%d.key"},
          Decoder::StateBinding{"custom.%d.value", "present.%d.value"},
          std::nullopt}};

  EXPECT_NO_THROW(
      ModelStateManifest::ValidateDynamicEngineCompatibility(decoder));
}

TEST(ModelStateManifestTest, AcceptsLegacyDynamicEngineContract) {
  Config::Model::Decoder decoder;
  decoder.num_hidden_layers = 4;

  EXPECT_NO_THROW(ModelStateManifest::ValidateDynamicEngineCompatibility(decoder));
}

}  // namespace
}  // namespace Generators::test
