// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "dflash2_drafter.h"
#include "ort_genai.h"

namespace Generators::test {
namespace {

namespace fs_std = std::filesystem;

Config MakeDflash2Config() {
  Config config;
  config.model.vocab_size = 128;
  auto& decoder = config.model.decoder;
  decoder.filename = "target.onnx";
  decoder.sliding_window = Config::Model::Decoder::SlidingWindow{4096};
  decoder.state_groups.emplace();
  decoder.state_groups->push_back(Config::Model::Decoder::StateGroup{
      Config::Model::Decoder::StateGroupKind::FixedConv});

  auto& dflash2 = config.model.dflash2;
  dflash2.filename = "dflash2.onnx";
  dflash2.num_hidden_layers = 3;
  dflash2.num_key_value_heads = 2;
  dflash2.head_size = 8;
  dflash2.block_size = 4;
  dflash2.num_draft_tokens = 3;
  dflash2.selector_top_k = 2;
  dflash2.mask_token_id = 31;
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
  drafter.AddInput("input_ids", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {-1});
  for (const auto* name : {"q_row_map", "qkv_row_map", "block_row_index",
                           "cumulative_sequence_lengths", "past_sequence_lengths"}) {
    drafter.AddInput(name, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1});
  }
  drafter.AddInput("block_table", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1, -1});
  drafter.AddInput("attention_metadata", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {3});
  for (int layer = 0; layer < 3; ++layer) {
    for (const auto* kind : {"key", "value"}) {
      drafter.AddInput("past_key_values." + std::to_string(layer) + "." + kind,
                       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 8, 2, 8});
      drafter.AddOutput("present." + std::to_string(layer) + "." + kind,
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 8, 2, 8});
    }
  }
  drafter.AddOutput("draft_candidate_ids", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1, 3, 2});
  drafter.AddOutput("draft_scores", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 3, 2, 2});
  return {std::move(target), std::move(drafter)};
}

fs_std::path WriteDsparkConfig(std::string_view section_name) {
  const auto root = fs_std::temp_directory_path() /
                    ("ortgenai_dspark_config_" + std::string{section_name});
  std::error_code error;
  fs_std::remove_all(root, error);
  fs_std::create_directories(root);

  std::ofstream out(root / "genai_config.json", std::ios::binary);
  out << "{ \"model\": { \"type\": \"tiny-test-model\","
         " \"vocab_size\": 16, \"context_length\": 32,"
         " \"decoder\": { \"filename\": \"model.onnx\" }, \""
      << section_name
      << "\": { \"filename\": \"dspark.onnx\", \"num_hidden_layers\": 1,"
         " \"num_key_value_heads\": 2, \"head_size\": 8, \"block_size\": 4,"
         " \"num_draft_tokens\": 4, \"selector_top_k\": 2 } }, \"search\": {} }";
  return root;
}

fs_std::path WriteDuplicateBlockDrafterConfig() {
  const auto root = fs_std::temp_directory_path() / "ortgenai_duplicate_block_drafter_config";
  std::error_code error;
  fs_std::remove_all(root, error);
  fs_std::create_directories(root);

  std::ofstream out(root / "genai_config.json", std::ios::binary);
  out << R"({"model":{"type":"tiny-test-model","vocab_size":16,"context_length":32,)"
         R"("decoder":{"filename":"model.onnx"},)"
         R"("dflash2":{"filename":"dflash2.onnx"},)"
         R"("dspark":{"filename":"dspark.onnx"}},"search":{}})";
  return root;
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

TEST(Dflash2ConfigTest, RejectsSimultaneousMtpDrafter) {
  auto config = MakeDflash2Config();
  config.model.mtp.filename = "mtp.onnx";
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);
}

TEST(Dflash2ConfigTest, PreservesTargetProviderOptions) {
  auto config = MakeDflash2Config();
  config.model.decoder.session_options.providers = {"cuda"};
  config.model.decoder.session_options.provider_options.push_back(
      {"cuda", {{"device_id", "1"}, {"arena_extend_strategy", "kSameAsRequested"}}});
  config.model.dflash2.session_options.emplace();
  config.model.dflash2.session_options->provider_options.push_back(
      {"cuda", {{"arena_extend_strategy", "kNextPowerOfTwo"}}});

  const auto projected = CreateDflash2Config(config);
  const auto& session_options = projected->model.decoder.session_options;
  ASSERT_EQ(session_options.providers.size(), 1u);
  EXPECT_EQ(session_options.providers[0], "cuda");
  ASSERT_EQ(session_options.provider_options.size(), 1u);
  ASSERT_EQ(session_options.provider_options[0].options.size(), 2u);
  EXPECT_EQ(session_options.provider_options[0].options[0],
            Config::NamedString("arena_extend_strategy", "kNextPowerOfTwo"));
  EXPECT_EQ(session_options.provider_options[0].options[1],
            Config::NamedString("device_id", "1"));
}

TEST(Dflash2ConfigTest, AcceptsCompatibleAuxiliaryHiddenStates) {
  const auto config = MakeDflash2Config();
  const auto [target, drafter] = MakeCompatibleMetadata();
  EXPECT_NO_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8));
}

TEST(Dflash2ConfigTest, UsesConfiguredTargetOutput) {
  auto config = MakeDflash2Config();
  config.model.dflash2.main_aux_hidden_states = "custom_aux_hidden_states";
  auto [target, drafter] = MakeCompatibleMetadata();
  target.AddOutput("custom_aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 64});
  EXPECT_NO_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8));
}

TEST(Dflash2ConfigTest, RequiresConfiguredTargetOutput) {
  const auto config = MakeDflash2Config();
  FakeModelStateMetadata target;
  FakeModelStateMetadata drafter;
  drafter.AddInput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 64});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresConfiguredDrafterInput) {
  const auto config = MakeDflash2Config();
  FakeModelStateMetadata target;
  target.AddOutput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 64});
  const FakeModelStateMetadata drafter;
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresTwoDimensionalAuxiliaryTensors) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  target.AddOutput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 4, 16});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);

  std::tie(target, drafter) = MakeCompatibleMetadata();
  drafter.AddInput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 4, 16});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresMatchingAuxiliaryWidth) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.AddInput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 32});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresMatchingAuxiliaryType) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.AddInput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 64});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresCompleteDrafterContract) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.AddOutput("draft_scores", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 3, 2});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);

  std::tie(target, drafter) = MakeCompatibleMetadata();
  drafter.AddInput("past_key_values.1.key", ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
                   {-1, -1, 4, 8});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresOneDraftPerNonAnchorBlockRow) {
  auto config = MakeDflash2Config();
  config.model.dflash2.num_draft_tokens = 1;
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresMaskTokenInsideVocabulary) {
  auto config = MakeDflash2Config();
  config.model.vocab_size = 0;
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);

  config.model.vocab_size = 128;
  config.model.dflash2.mask_token_id = -1;
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);

  config.model.dflash2.mask_token_id = config.model.vocab_size;
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);

  config.model.dflash2.mask_token_id = 0;
  EXPECT_NO_THROW(CreateDflash2Config(config));

  config.model.dflash2.mask_token_id = config.model.vocab_size - 1;
  EXPECT_NO_THROW(CreateDflash2Config(config));
}

TEST(Dflash2ConfigTest, AcceptsOneDraftPerDsparkBlockRow) {
  auto config = MakeDflash2Config();
  config.model.dflash2.is_dspark = true;
  config.model.dflash2.num_draft_tokens = config.model.dflash2.block_size;
  EXPECT_NO_THROW(CreateDflash2Config(config));
}

TEST(Dflash2ConfigTest, RejectsDsparkGeometryForDflash2) {
  auto config = MakeDflash2Config();
  config.model.dflash2.num_draft_tokens = config.model.dflash2.block_size;
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);
}

TEST(Dflash2ConfigTest, RejectsDflash2GeometryForDspark) {
  auto config = MakeDflash2Config();
  config.model.dflash2.is_dspark = true;
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);
}

TEST(Dflash2ConfigTest, ParsesDsparkAlias) {
  EXPECT_NO_THROW(OgaConfig::Create(WriteDsparkConfig("dspark").string().c_str()));
  EXPECT_THROW(OgaConfig::Create(WriteDsparkConfig("dspark2").string().c_str()), std::exception);
}

TEST(Dflash2ConfigTest, RejectsBothBlockDrafterAliases) {
  EXPECT_THROW(OgaConfig::Create(WriteDuplicateBlockDrafterConfig().string().c_str()),
               std::runtime_error);
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
  EXPECT_EQ(Dflash2Drafter::BytesPerBlock(
                config, 16, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
            3072u);
  EXPECT_EQ(Dflash2Drafter::PoolBlocks(config, 8, 3), 15u);
}

TEST(Dflash2ConfigTest, BillsFullAttentionCachePerTargetBlock) {
  auto config = MakeDflash2Config();
  config.model.dflash2.sliding_window = 0;
  EXPECT_EQ(Dflash2Drafter::PoolBlocks(config, 8, 3), 0u);
  EXPECT_EQ(Dflash2Drafter::BytesPerBlock(
                config, 8, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
            3072u);
}

TEST(Dflash2ConfigTest, ReservesEveryFullAttentionQuerySpillBlock) {
  EXPECT_EQ(Dflash2Drafter::FullAttentionPoolBlocks(100, 4, 4, 3), 103u);
  EXPECT_EQ(Dflash2Drafter::FullAttentionPoolBlocks(100, 4, 5, 3), 106u);
  EXPECT_EQ(Dflash2Drafter::FullAttentionPoolBlocks(100, 4, 9, 3), 109u);
  EXPECT_EQ(Dflash2Drafter::FullAttentionReservedBytes(4, 9, 3, 128), 1152u);
}

TEST(Dflash2ConfigTest, RejectsFullAttentionPoolOverflow) {
  EXPECT_THROW(
      Dflash2Drafter::FullAttentionPoolBlocks(
          std::numeric_limits<size_t>::max(), 4, 4, 1),
      std::runtime_error);
  EXPECT_THROW(
      Dflash2Drafter::FullAttentionReservedBytes(
          4, 4, 2, std::numeric_limits<size_t>::max()),
      std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresMatchingCacheTypes) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.AddInput("past_key_values.1.value", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                   {-1, 8, 2, 8});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresDynamicCachePoolDimension) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.AddInput("past_key_values.1.value", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                   {32, 8, 2, 8});
  drafter.AddOutput("present.1.value", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                    {32, 8, 2, 8});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RejectsUnsupportedCacheType) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  for (int layer = 0; layer < 3; ++layer) {
    for (const auto* kind : {"key", "value"}) {
      drafter.AddInput("past_key_values." + std::to_string(layer) + "." + kind,
                       ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1, 8, 2, 8});
      drafter.AddOutput("present." + std::to_string(layer) + "." + kind,
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1, 8, 2, 8});
    }
  }
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresUniqueCacheBindings) {
  auto config = MakeDflash2Config();
  config.model.dflash2.inputs.past_value_names = config.model.dflash2.inputs.past_key_names;
  const auto [target, drafter] = MakeCompatibleMetadata();
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresMatchingPagedBlockSize) {
  const auto config = MakeDflash2Config();
  const auto [target, drafter] = MakeCompatibleMetadata();
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 16),
               std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresMatchingLatticeGeometry) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.AddOutput("draft_candidate_ids", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1, 4, 2});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresDynamicMatchingLatticeBatchDimension) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.AddOutput("draft_candidate_ids", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {4, 3, 2});
  drafter.AddOutput("draft_scores", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4, 3, 2, 2});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);

  std::tie(target, drafter) = MakeCompatibleMetadata();
  drafter.AddOutput("draft_scores", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-2, 3, 2, 2});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RejectsInvalidCachePoolGeometry) {
  const auto config = MakeDflash2Config();
  EXPECT_THROW(Dflash2Drafter::PoolBlocks(config, 0, 3), std::runtime_error);
  EXPECT_THROW(Dflash2Drafter::PoolBlocks(config, 8, std::numeric_limits<size_t>::max()),
               std::runtime_error);
}

TEST(Dflash2ConfigTest, CapsDraftWidthBySessionAndTurnLimits) {
  EXPECT_EQ(Dflash2DraftWidth(7, 5, 8, 10, 9), 1u);
  EXPECT_EQ(Dflash2DraftWidth(7, 5, 8, 20, 3), 2u);
  EXPECT_EQ(Dflash2DraftWidth(7, 5, 8, 20, 1), 0u);
  EXPECT_EQ(Dflash2DraftWidth(7, 5, 9, 10, 9), 0u);
}

TEST(Dflash2ConfigTest, ReusesProposalBufferUntilAStepOutgrowsIt) {
  auto* device = GetDeviceInterface(DeviceType::CPU);
  constexpr auto type = Ort::TypeToTensorType<int32_t>;
  std::unique_ptr<Tensor> slot;

  Dflash2StepTensor(slot, device, type, {2, 4});
  EXPECT_EQ(slot->GetElementCount(), 8u);
  const void* buffer = slot->buffer_;
  ASSERT_NE(buffer, nullptr);

  // A narrower step reshapes a view over the same buffer instead of allocating.
  Dflash2StepTensor(slot, device, type, {1, 3});
  EXPECT_EQ(slot->GetElementCount(), 3u);
  EXPECT_EQ(slot->GetShape(), (std::vector<int64_t>{1, 3}));
  EXPECT_EQ(slot->buffer_, buffer);

  // Tensor rejects a static shape larger than its buffer, so growth must start a new one.
  EXPECT_NO_THROW(Dflash2StepTensor(slot, device, type, {4, 8}));
  EXPECT_EQ(slot->GetElementCount(), 32u);
  const void* grown = slot->buffer_;

  Dflash2StepTensor(slot, device, type, {2, 4});
  EXPECT_EQ(slot->buffer_, grown);
}

TEST(Dflash2ConfigTest, ReusesProposalBufferForAnEmptyStep) {
  auto* device = GetDeviceInterface(DeviceType::CPU);
  constexpr auto type = Ort::TypeToTensorType<int32_t>;
  std::unique_ptr<Tensor> slot;

  // A step can serve feeds that carry a query block but no context rows.
  EXPECT_NO_THROW(Dflash2StepTensor(slot, device, type, {0, 64}));
  EXPECT_EQ(slot->GetElementCount(), 0u);
  EXPECT_NO_THROW(Dflash2StepTensor(slot, device, type, {2, 64}));
  EXPECT_EQ(slot->GetElementCount(), 128u);
}

TEST(Dflash2ConfigTest, RejectsProposalShapesThatOverflow) {
  auto* device = GetDeviceInterface(DeviceType::CPU);
  std::unique_ptr<Tensor> slot;
  const int64_t huge = std::numeric_limits<int64_t>::max();
  EXPECT_THROW(Dflash2StepTensor(slot, device, Ort::TypeToTensorType<int32_t>, {huge, huge}),
               std::runtime_error);
}

TEST(Dflash2ConfigTest, ReplacesProposalBufferWhenTheElementTypeChanges) {
  auto* device = GetDeviceInterface(DeviceType::CPU);
  std::unique_ptr<Tensor> slot;

  Dflash2StepTensor(slot, device, Ort::TypeToTensorType<int32_t>, {8});
  // A static Tensor keeps the type it was constructed with, so reusing the buffer for another type
  // would silently hand the session the wrong element type.
  Dflash2StepTensor(slot, device, Ort::TypeToTensorType<float>, {4});
  EXPECT_EQ(slot->GetType(), Ort::TypeToTensorType<float>);
  EXPECT_EQ(slot->GetElementCount(), 4u);
}

TEST(Dflash2ConfigTest, DraftsOnlyForGreedyRequests) {
  Config::Search search;
  search.do_sample = false;
  EXPECT_TRUE(Dflash2CanDraft(search));

  search.do_sample = true;
  search.top_k = 50;
  search.temperature = 1.0f;
  EXPECT_FALSE(Dflash2CanDraft(search));

  // Sampling that can only ever pick the top logit is still greedy.
  search.top_k = 1;
  EXPECT_TRUE(Dflash2CanDraft(search));

  search.top_k = 50;
  search.temperature = 0.0f;
  EXPECT_TRUE(Dflash2CanDraft(search));
}

}  // namespace Generators::test
