// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "dflash2_drafter.h"
#include "engine/paged_key_value_cache.h"
#include "engine/step_plan.h"
#include "models/io/kv_cache.h"
#include "ort_genai.h"
#include "search.h"

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

class CheckpointAllocationDevice final : public DeviceInterface {
 public:
  CheckpointAllocationDevice(DeviceInterface& inner, Ort::Allocator& allocator)
      : inner_{inner}, allocator_{allocator} {}

  DeviceType GetType() const override { return inner_.GetType(); }
  void InitOrt(const OrtApi& api, Ort::Allocator& allocator) override {
    inner_.InitOrt(api, allocator);
  }
  Ort::Allocator& GetAllocator() override { return allocator_; }
  std::unique_ptr<OrtMemoryInfo> GetMemoryInfo() const override {
    return inner_.GetMemoryInfo();
  }
  std::string GetExecutionProviderName() const override {
    return inner_.GetExecutionProviderName();
  }
  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return inner_.AllocateBase(size);
  }
  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* memory, size_t size) override {
    return inner_.WrapMemoryBase(memory, size);
  }
  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override {
    return inner_.CreateGreedy(params);
  }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override {
    return inner_.CreateBeam(params);
  }
  std::unique_ptr<KeyValueCache> CreateKeyValueCache(State& state) override {
    return inner_.CreateKeyValueCache(state);
  }
  void Synchronize() override { inner_.Synchronize(); }

 private:
  DeviceInterface& inner_;
  Ort::Allocator& allocator_;
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

fs_std::path WriteIncompleteBlockDrafterConfig() {
  const auto root = fs_std::temp_directory_path() / "ortgenai_incomplete_block_drafter_config";
  std::error_code error;
  fs_std::remove_all(root, error);
  fs_std::create_directories(root);

  std::ofstream out(root / "genai_config.json", std::ios::binary);
  out << R"({"model":{"type":"tiny-test-model","vocab_size":16,"context_length":32,)"
         R"("decoder":{"filename":"model.onnx"},"dflash2":{ }},"search":{}})";
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

TEST(Dflash2ConfigTest, ParsesIndependentSamplingOptions) {
  const auto root = fs_std::temp_directory_path() / "ortgenai_dflash_independent_sampling";
  std::error_code error;
  fs_std::remove_all(root, error);
  fs_std::create_directories(root);
  std::ofstream out(root / "genai_config.json", std::ios::binary);
  out << R"({"model":{"type":"tiny-test-model","vocab_size":128,"context_length":32,)"
         R"("decoder":{"filename":"model.onnx"},"dflash2":{"filename":"dflash2.onnx",)"
         R"("num_hidden_layers":1,"num_key_value_heads":2,"head_size":8,"block_size":4,)"
         R"("num_draft_tokens":3,"selector_top_k":4,"mask_token_id":31,"sliding_window":17,)"
         R"("independent_sampling":true,"sampling_temperature":0.1,"sampling_top_p":0.95,)"
         R"("sampling_min_p":0.3}},"search":{}})";
  out.close();

  Config config(fs::path{root.string()}, "");
  EXPECT_TRUE(config.model.dflash2.independent_sampling);
  EXPECT_FLOAT_EQ(config.model.dflash2.sampling_temperature, 0.1f);
  EXPECT_FLOAT_EQ(config.model.dflash2.sampling_top_p, 0.95f);
  EXPECT_FLOAT_EQ(config.model.dflash2.sampling_min_p, 0.3f);
}

TEST(Dflash2ConfigTest, RejectsSamplingTemperatureOutsideFloatRange) {
  const auto root = fs_std::temp_directory_path() /
                    "ortgenai_dflash_sampling_temperature_out_of_range";
  std::error_code error;
  fs_std::remove_all(root, error);
  fs_std::create_directories(root);
  std::ofstream out(root / "genai_config.json", std::ios::binary);
  out << R"({"model":{"type":"tiny-test-model","vocab_size":128,"context_length":32,)"
         R"("decoder":{"filename":"model.onnx"},"dflash2":{"filename":"dflash2.onnx",)"
         R"("num_hidden_layers":1,"num_key_value_heads":2,"head_size":8,"block_size":4,)"
         R"("num_draft_tokens":3,"selector_top_k":4,"mask_token_id":31,"sliding_window":17,)"
         R"("independent_sampling":true,"sampling_temperature":1e39}},"search":{}})";
  out.close();

  EXPECT_THROW(Config(fs::path{root.string()}, ""), std::runtime_error);
}

TEST(Dflash2ConfigTest, BuildsReferenceIndependentDistribution) {
  const std::array<int32_t, 4> candidates{10, 11, 12, 13};
  const std::array<float, 4> logits{4.0f, 3.0f, 2.0f, 1.0f};
  const auto distribution = Dflash2IndependentDraftDistribution(
      candidates.data(), logits.data(), candidates.size(),
      /*temperature=*/1.0f, /*top_p=*/0.95f, /*min_p=*/0.3f);
  ASSERT_EQ(distribution.indices, (std::vector<int32_t>{10, 11}));
  ASSERT_EQ(distribution.probs.size(), 2u);
  EXPECT_NEAR(distribution.probs[0], 0.7310586f, 1e-6f);
  EXPECT_NEAR(distribution.probs[1], 0.2689414f, 1e-6f);
}

TEST(Dflash2ConfigTest, SortsSelectorScoresBeforeApplyingProbabilityFilters) {
  const std::array<int32_t, 4> candidates{10, 11, 12, 13};
  const std::array<float, 4> logits{2.0f, 4.0f, 1.0f, 3.0f};
  const auto distribution = Dflash2IndependentDraftDistribution(
      candidates.data(), logits.data(), candidates.size(),
      /*temperature=*/1.0f, /*top_p=*/0.7f, /*min_p=*/0.2f);
  ASSERT_EQ(distribution.indices, (std::vector<int32_t>{11, 13}));
  ASSERT_EQ(distribution.probs.size(), 2u);
  EXPECT_NEAR(distribution.probs[0], 0.7310586f, 1e-6f);
  EXPECT_NEAR(distribution.probs[1], 0.2689414f, 1e-6f);
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

TEST(Dflash2ConfigTest, AllowsDisabledMtpMetadata) {
  auto config = MakeDflash2Config();
  config.model.mtp.filename = "mtp.onnx";
  config.model.mtp.enabled = false;
  EXPECT_NO_THROW(CreateDflash2Config(config));
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

TEST(Dflash2ConfigTest, DrafterSessionOverridesTargetConfigEntries) {
  auto config = MakeDflash2Config();
  config.model.decoder.session_options.config_entries.push_back(
      {"ep.cuda.fpa_intb_gemm", "1"});
  config.model.dflash2.session_options.emplace();
  config.model.dflash2.session_options->config_entries.push_back(
      {"ep.cuda.fpa_intb_gemm", "0"});

  const auto projected = CreateDflash2Config(config);
  const auto& entries = projected->model.decoder.session_options.config_entries;
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(entries[0], Config::NamedString("ep.cuda.fpa_intb_gemm", "0"));
}

TEST(Dflash2ConfigTest, AcceptsCompatibleAuxiliaryHiddenStates) {
  const auto config = MakeDflash2Config();
  const auto [target, drafter] = MakeCompatibleMetadata();
  EXPECT_NO_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8));
}

// Dflash2Drafter::AllocateCache() allocates only key and value buffers from the logical head size
// and binds only past_key_names/past_value_names, so a declared scale template would be silently
// dropped and the drafter session would run against unbound, uninitialized scales. Reject it.
TEST(Dflash2ConfigTest, RejectsScaleNameTemplates) {
  const auto metadata = MakeCompatibleMetadata();
  for (int field = 0; field < 4; ++field) {
    auto config = MakeDflash2Config();
    auto& inputs = config.model.dflash2.inputs;
    auto& outputs = config.model.dflash2.outputs;
    switch (field) {
      case 0:
        inputs.past_key_scale_names = "draft_past.%d.key_scale";
        break;
      case 1:
        inputs.past_value_scale_names = "draft_past.%d.value_scale";
        break;
      case 2:
        outputs.present_key_scale_names = "draft_present.%d.key_scale";
        break;
      default:
        outputs.present_value_scale_names = "draft_present.%d.value_scale";
        break;
    }
    try {
      static_cast<void>(ValidateDflash2ModelCompatibility(config, metadata.first, metadata.second, 8));
      FAIL() << "Expected a quantized block-drafter cache to be rejected for field " << field;
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string{error.what()}.find("not supported"), std::string::npos)
          << "field " << field << ": " << error.what();
    }
  }
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

TEST(Dflash2ConfigTest, RequiresDynamicPackedDimensions) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  target.AddOutput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {1, 64});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);

  std::tie(target, drafter) = MakeCompatibleMetadata();
  drafter.AddInput("q_row_map", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {1});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);

  std::tie(target, drafter) = MakeCompatibleMetadata();
  drafter.AddInput("block_table", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1, 8});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresUniqueRuntimeInputNames) {
  auto config = MakeDflash2Config();
  config.model.dflash2.inputs.qkv_row_map = config.model.dflash2.inputs.q_row_map;
  const auto [target, drafter] = MakeCompatibleMetadata();
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

TEST(Dflash2ConfigTest, RequiresUniqueEmbeddingInputName) {
  auto config = MakeDflash2Config();
  config.model.embedding.filename = "embedding.onnx";
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.AddInput(config.model.dflash2.inputs.embeddings, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 64});
  EXPECT_NO_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8));
  for (const auto* alias : {"aux_hidden_states", "input_ids", "past_key_values.0.key", ""}) {
    config.model.dflash2.inputs.embeddings = alias;
    EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error) << alias;
  }
}

TEST(Dflash2ConfigTest, ReservesEmbeddingGrowthPeak) {
  EXPECT_EQ(Dflash2Drafter::EmbeddingReservedBytes(2, 4, 64, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
            3u * 2 * 4 * 64 * 2);
  EXPECT_EQ(Dflash2Drafter::EmbeddingReservedBytes(3, 8, 64, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
            3u * 3 * 8 * 64 * 4);
  EXPECT_THROW(Dflash2Drafter::EmbeddingReservedBytes(std::numeric_limits<size_t>::max(), 2, 64,
                                                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
               std::runtime_error);
  EXPECT_THROW(Dflash2Drafter::EmbeddingReservedBytes(std::numeric_limits<size_t>::max() / 2, 1, 1,
                                                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
               std::runtime_error);
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

TEST(Dflash2ConfigTest, RejectsFullAttentionForDflash2) {
  auto config = MakeDflash2Config();
  config.model.dflash2.sliding_window = 0;
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

TEST(Dflash2ConfigTest, AllowsSameBlockDrafterAliasInOverlays) {
  const auto root = WriteDsparkConfig("dspark");
  EXPECT_NO_THROW(Config(fs::path{root.string()},
                         R"({"model":{"dspark":{"selector_top_k":3}}})"));

  auto config = OgaConfig::Create(root.string().c_str());
  EXPECT_NO_THROW(config->Overlay(R"({"model":{"dspark":{"selector_top_k":3}}})"));
}

TEST(Dflash2ConfigTest, RejectsDifferentBlockDrafterAliasInOverlay) {
  auto config = OgaConfig::Create(WriteDsparkConfig("dspark").string().c_str());
  EXPECT_THROW(config->Overlay(R"({"model":{"dflash2":{"selector_top_k":3}}})"),
               std::runtime_error);

  config = OgaConfig::Create(WriteIncompleteBlockDrafterConfig().string().c_str());
  EXPECT_THROW(config->Overlay(R"({"model":{"dspark":{"filename":"dspark.onnx"}}})"),
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
  EXPECT_EQ(Dflash2Drafter::PoolBytes(
                config, 8, 15, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
            23040u);
  EXPECT_EQ(Dflash2Drafter::PrefixCheckpointBytes(
                config, 8, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
            7680u);
}

TEST(Dflash2ConfigTest, OptionalPrefixCheckpointNeverConsumesTheLastTargetBlock) {
  EXPECT_TRUE(CanReserveDflash2PrefixCheckpoint(
      /*target_budget_bytes=*/100, /*reserved_bytes=*/40,
      /*snapshot_bytes=*/20, /*target_block_bytes=*/40));
  EXPECT_FALSE(CanReserveDflash2PrefixCheckpoint(99, 40, 20, 40));
  EXPECT_FALSE(CanReserveDflash2PrefixCheckpoint(100, 100, 20, 40));
  EXPECT_FALSE(CanReserveDflash2PrefixCheckpoint(100, 40, 60, 1));
  EXPECT_FALSE(CanReserveDflash2PrefixCheckpoint(100, 40, 20, 41));
  const size_t effective_budget = PagedCacheMemoryBudget(1000, 1.0f);
  EXPECT_TRUE(CanReserveDflash2PrefixCheckpoint(1000, 860, 40, 40));
  EXPECT_EQ(ComputePagedBlockCapacityFromBytes(1000, 1.0f, 860, 40), 1u);
  EXPECT_FALSE(CanReserveDflash2PrefixCheckpoint(effective_budget, 860, 40, 40));
}

TEST(Dflash2ConfigTest, RejectsWindowedPoolByteOverflow) {
  const auto config = MakeDflash2Config();
  EXPECT_THROW(
      Dflash2Drafter::PoolBytes(
          config, 8, std::numeric_limits<size_t>::max(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      std::runtime_error);
}

TEST(Dflash2ConfigTest, BillsFullAttentionCachePerTargetBlock) {
  auto config = MakeDflash2Config();
  config.model.dflash2.is_dspark = true;
  config.model.dflash2.num_draft_tokens = config.model.dflash2.block_size;
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

TEST(Dflash2ConfigTest, RunsFullAttentionDsparkAcrossRequestLifecycles) {
  auto config = MakeDflash2Config();
  config.config_path = fs::path{MODEL_PATH "engine/synthetic-dspark"};
  auto& dspark = config.model.dflash2;
  dspark.filename = "dspark.onnx";
  dspark.is_dspark = true;
  dspark.num_hidden_layers = 1;
  dspark.num_key_value_heads = 1;
  dspark.head_size = 1;
  dspark.block_size = 4;
  dspark.num_draft_tokens = 4;
  dspark.sliding_window = 0;

  auto model = std::make_shared<Dflash2Model>(CreateDflash2Config(config), GetOrtEnv());
  Dflash2Drafter drafter{model, /*paged_block_size=*/4, /*num_blocks=*/10,
                         /*max_requests=*/2};

  int request_a_id = 0;
  int request_b_id = 0;
  int request_d_id = 0;
  int request_e_id = 0;
  auto* request_a = reinterpret_cast<Request*>(&request_a_id);
  auto* request_b = reinterpret_cast<Request*>(&request_b_id);
  auto* request_d = reinterpret_cast<Request*>(&request_d_id);
  auto* request_e = reinterpret_cast<Request*>(&request_e_id);

  auto* device = GetDeviceInterface(DeviceType::CPU);
  Tensor first_aux{device, Ort::TypeToTensorType<float>};
  const std::array<int64_t, 2> first_aux_shape{18, 1};
  first_aux.CreateTensor(first_aux_shape);
  const std::array first_feeds{
      Dflash2Drafter::Feed{.request = request_a, .aux_row_begin = 0, .aux_row_count = 9, .first_position = 0, .anchor_token = 11, .draft_eligible = true, .wants_drafts = true},
      Dflash2Drafter::Feed{.request = request_b, .aux_row_begin = 9, .aux_row_count = 9, .first_position = 0, .anchor_token = 12, .draft_eligible = true, .wants_drafts = true},
  };
  std::vector<std::vector<int32_t>> drafts;
  drafter.Propose(first_aux, first_feeds, drafts);
  ASSERT_EQ(drafts.size(), 2u);
  EXPECT_EQ(drafts[0], (std::vector<int32_t>{6, 0, 13, 64}));
  EXPECT_EQ(drafts[1], (std::vector<int32_t>{22, 0, 13, 64}));

  Tensor growth_aux{device, Ort::TypeToTensorType<float>};
  const std::array<int64_t, 2> growth_aux_shape{8, 1};
  growth_aux.CreateTensor(growth_aux_shape);
  const std::array growth_feeds{
      Dflash2Drafter::Feed{.request = request_a, .aux_row_begin = 0, .aux_row_count = 4, .first_position = 9, .anchor_token = 13, .draft_eligible = true, .wants_drafts = true},
      Dflash2Drafter::Feed{.request = request_b, .aux_row_begin = 4, .aux_row_count = 4, .first_position = 9, .anchor_token = 14, .draft_eligible = true, .wants_drafts = true},
  };
  drafter.Propose(growth_aux, growth_feeds, drafts);
  ASSERT_EQ(drafts.size(), 2u);
  EXPECT_EQ(drafts[0], (std::vector<int32_t>{14, 9, 8, 44}));
  EXPECT_EQ(drafts[1], (std::vector<int32_t>{31, 9, 8, 44}));

  // Rewind releases only the selected request. It can replay from position zero while the peer
  // continues from its existing drafter state.
  drafter.Release(request_a);
  Tensor reused_aux{device, Ort::TypeToTensorType<float>};
  const std::array<int64_t, 2> reused_aux_shape{10, 1};
  reused_aux.CreateTensor(reused_aux_shape);
  const std::array reused_feeds{
      Dflash2Drafter::Feed{.request = request_a, .aux_row_begin = 0, .aux_row_count = 9, .first_position = 0, .anchor_token = 15, .draft_eligible = true, .wants_drafts = true},
      Dflash2Drafter::Feed{.request = request_b, .aux_row_begin = 9, .aux_row_count = 1, .first_position = 13, .anchor_token = 16, .draft_eligible = true, .wants_drafts = true},
  };
  drafter.Propose(reused_aux, reused_feeds, drafts);
  ASSERT_EQ(drafts.size(), 2u);
  EXPECT_EQ(drafts[0], (std::vector<int32_t>{13, 0, 13, 32}));
  EXPECT_EQ(drafts[1], (std::vector<int32_t>{31, 13, 5, 32}));
  EXPECT_EQ(drafter.AdmissionMisses(), 0u);

  Tensor failed_aux{device, Ort::TypeToTensorType<float>};
  const std::array<int64_t, 2> failed_aux_shape{1, 1};
  failed_aux.CreateTensor(failed_aux_shape);
  const std::array failed_feed{
      Dflash2Drafter::Feed{.request = request_a, .aux_row_begin = 0, .aux_row_count = 1, .first_position = 10, .anchor_token = 17, .draft_eligible = true, .wants_drafts = true},
  };
  EXPECT_THROW(drafter.Propose(failed_aux, failed_feed, drafts), std::logic_error);
  drafter.ReleaseAll();

  const std::array recovered_feeds{
      Dflash2Drafter::Feed{.request = request_d, .aux_row_begin = 0, .aux_row_count = 9, .first_position = 0, .anchor_token = 17, .draft_eligible = true, .wants_drafts = true},
      Dflash2Drafter::Feed{.request = request_e, .aux_row_begin = 9, .aux_row_count = 9, .first_position = 0, .anchor_token = 18, .draft_eligible = true, .wants_drafts = true},
  };
  drafter.Propose(first_aux, recovered_feeds, drafts);
  ASSERT_EQ(drafts.size(), 2u);
  EXPECT_EQ(drafts[0][1], 0);
  EXPECT_EQ(drafts[0][2], 13);
  EXPECT_EQ(drafts[0][3], 64);
  EXPECT_EQ(drafts[1][1], 0);
  EXPECT_EQ(drafts[1][2], 13);
  EXPECT_EQ(drafts[1][3], 64);
  EXPECT_EQ(drafter.AdmissionMisses(), 0u);
}

TEST(Dflash2ConfigTest, ExecutesWindowedRestoreAcrossRingWrap) {
  auto config = MakeDflash2Config();
  config.config_path = fs::path{MODEL_PATH "engine/synthetic-dspark"};
  auto& draft = config.model.dflash2;
  draft.filename = "dflash2.onnx";
  draft.num_hidden_layers = 1;
  draft.num_key_value_heads = 1;
  draft.head_size = 1;
  draft.block_size = 4;
  draft.num_draft_tokens = 3;
  draft.selector_top_k = 2;
  draft.sliding_window = 8;

  auto model = std::make_shared<Dflash2Model>(CreateDflash2Config(config), GetOrtEnv());
  const size_t ring_blocks = Dflash2Drafter::PoolBlocks(config, 4, 1);
  Dflash2Drafter drafter{model, /*paged_block_size=*/4, ring_blocks * 2,
                         /*max_requests=*/2};
  int source_id = 0, restored_id = 0, peer_id = 0, extra_id = 0, remap_id = 0;
  auto* source = reinterpret_cast<Request*>(&source_id);
  auto* restored = reinterpret_cast<Request*>(&restored_id);
  auto* peer = reinterpret_cast<Request*>(&peer_id);
  auto* extra = reinterpret_cast<Request*>(&extra_id);
  auto* remap = reinterpret_cast<Request*>(&remap_id);
  Tensor aux{GetDeviceInterface(DeviceType::CPU), Ort::TypeToTensorType<float>};
  aux.CreateTensor(std::array<int64_t, 2>{8, 1});
  std::vector<std::vector<int32_t>> proposals;

  for (size_t position = 0; position < 24; position += 8) {
    const std::array feeds{Dflash2Drafter::Feed{
        .request = source, .aux_row_count = 8, .first_position = position, .anchor_token = 11, .draft_eligible = true, .wants_drafts = true}};
    ASSERT_TRUE(drafter.Propose(aux, feeds, proposals));
  }
  auto* device = model->p_device_kvcache_;
  auto failing_allocator = Ort::Allocator::Create(
      *model->session_, device->GetAllocator().GetInfo());
  static_cast<OrtAllocator&>(*failing_allocator).Alloc =
      [](OrtAllocator*, size_t) -> void* { throw std::bad_alloc{}; };
  EXPECT_THROW(
      OrtValue::CreateTensor(
          *failing_allocator, std::array<int64_t, 1>{1}, Ort::TypeToTensorType<float>),
      Ort::Exception);
  CheckpointAllocationDevice failing_device{*device, *failing_allocator};
  model->p_device_kvcache_ = &failing_device;
  EXPECT_THROW(drafter.CapturePrefix(source, 24), std::bad_alloc);
  model->p_device_kvcache_ = device;

  auto checkpoint = drafter.CapturePrefix(source, 24);
  ASSERT_NE(checkpoint, nullptr);
  EXPECT_EQ(checkpoint->ring_blocks, ring_blocks);
  Tensor next_aux{GetDeviceInterface(DeviceType::CPU), Ort::TypeToTensorType<float>};
  next_aux.CreateTensor(std::array<int64_t, 2>{1, 1});
  next_aux.GetByteSpan().Zero();
  const std::array target_only{Dflash2Drafter::Feed{
      .request = peer, .aux_row_count = 1, .first_position = 24, .anchor_token = 13, .draft_eligible = true, .wants_drafts = true}};
  ASSERT_FALSE(drafter.Propose(next_aux, target_only, proposals));
  EXPECT_TRUE(proposals.front().empty());
  EXPECT_TRUE(drafter.CanCapturePrefix(source, 24));
  EXPECT_FALSE(drafter.CanCapturePrefix(source, 23));
  EXPECT_FALSE(drafter.CanCapturePrefix(peer, 25));
  std::weak_ptr<const Dflash2PrefixCheckpoint> indexed_checkpoint = checkpoint;
  if (drafter.CanCapturePrefix(peer, 25)) {
    checkpoint.reset();
  }
  ASSERT_FALSE(indexed_checkpoint.expired());

  const std::array uninterrupted{Dflash2Drafter::Feed{
      .request = source, .aux_row_count = 1, .first_position = 24, .anchor_token = 12, .draft_eligible = true, .wants_drafts = true}};
  ASSERT_TRUE(drafter.Propose(next_aux, uninterrupted, proposals));
  ASSERT_EQ(proposals.size(), 1u);
  const auto expected = proposals.front();
  ASSERT_FALSE(expected.empty());
  EXPECT_EQ(expected.front(), 24);

  drafter.Release(source);
  const std::array remap_feed{Dflash2Drafter::Feed{
      .request = remap, .aux_row_count = 1, .anchor_token = 13, .draft_eligible = true, .wants_drafts = true}};
  ASSERT_TRUE(drafter.Propose(next_aux, remap_feed, proposals));
  const std::array resumed{Dflash2Drafter::Feed{
      .request = restored, .prefix_checkpoint = checkpoint, .aux_row_count = 1, .first_position = 24, .anchor_token = 12, .draft_eligible = true, .wants_drafts = true}};
  ASSERT_TRUE(drafter.Propose(next_aux, resumed, proposals));
  ASSERT_EQ(proposals.size(), 1u);
  EXPECT_EQ(proposals.front(), expected);
  EXPECT_EQ(drafter.AdmissionMisses(), 0u);

  drafter.Release(remap);
  drafter.Release(restored);
  ASSERT_TRUE(drafter.Propose(next_aux, resumed, proposals));
  ASSERT_EQ(proposals.size(), 1u);
  EXPECT_EQ(proposals.front(), expected);
  drafter.Release(restored);

  auto wrong_position = resumed;
  wrong_position.front().first_position = 23;
  const std::array with_peer{
      wrong_position.front(),
      Dflash2Drafter::Feed{
          .request = peer, .aux_row_count = 1, .anchor_token = 13, .draft_eligible = true, .wants_drafts = true}};
  EXPECT_TRUE(drafter.Propose(next_aux, with_peer, proposals));
  EXPECT_TRUE(proposals.front().empty());
  EXPECT_FALSE(proposals.back().empty());
  const std::array second_peer{Dflash2Drafter::Feed{
      .request = extra, .aux_row_count = 1, .anchor_token = 14, .draft_eligible = true, .wants_drafts = true}};
  ASSERT_TRUE(drafter.Propose(next_aux, second_peer, proposals));
  const std::array full_pool{
      Dflash2Drafter::Feed{
          .request = peer, .aux_row_count = 1, .first_position = 1, .anchor_token = 15, .draft_eligible = true, .wants_drafts = true},
      resumed.front()};
  EXPECT_TRUE(drafter.Propose(next_aux, full_pool, proposals));
  EXPECT_FALSE(proposals.front().empty());
  EXPECT_TRUE(proposals.back().empty());
  EXPECT_EQ(drafter.AdmissionMisses(), 1u);

  drafter.ReleaseAll();
  ASSERT_TRUE(drafter.Propose(next_aux, resumed, proposals));
  ASSERT_EQ(proposals.size(), 1u);
  EXPECT_FALSE(proposals.front().empty());

  drafter.ReleaseAll();
  auto alternate = resumed;
  alternate.front().request = remap;
  alternate.front().anchor_token = 13;
  alternate.front().aux_row_count = 2;
  Tensor alternate_aux{GetDeviceInterface(DeviceType::CPU), Ort::TypeToTensorType<float>};
  alternate_aux.CreateTensor(std::array<int64_t, 2>{2, 1});
  alternate_aux.GetByteSpan().Zero();
  ASSERT_TRUE(drafter.Propose(alternate_aux, alternate, proposals));
  ASSERT_EQ(proposals.size(), 1u);
  const auto alternate_expected = proposals.front();
  ASSERT_FALSE(alternate_expected.empty());
  ASSERT_NE(alternate_expected, expected);
  drafter.ReleaseAll();

  Tensor paired_aux{GetDeviceInterface(DeviceType::CPU), Ort::TypeToTensorType<float>};
  paired_aux.CreateTensor(std::array<int64_t, 2>{3, 1});
  paired_aux.GetByteSpan().Zero();
  auto paired_alternate = alternate.front();
  paired_alternate.aux_row_begin = 1;
  const std::array paired{resumed.front(), paired_alternate};
  ASSERT_TRUE(drafter.Propose(paired_aux, paired, proposals));
  ASSERT_EQ(proposals.size(), 2u);
  EXPECT_EQ(proposals[0], expected);
  EXPECT_EQ(proposals[1], alternate_expected);
  drafter.ReleaseAll();

  auto missing_state = std::make_shared<Dflash2PrefixCheckpoint>();
  missing_state->token_count = checkpoint->token_count;
  missing_state->ring_blocks = checkpoint->ring_blocks;
  for (const auto& cache : checkpoint->caches) {
    auto copy = std::make_unique<Tensor>(GetDeviceInterface(DeviceType::CPU), cache->GetType());
    copy->CreateTensor(cache->GetShape());
    copy->GetByteSpan().CopyFrom(cache->GetByteSpan());
    missing_state->caches.push_back(std::move(copy));
  }
  missing_state->caches.front()->GetByteSpan().Zero();
  drafter.ReleaseAll();
  auto damaged = resumed;
  damaged.front().prefix_checkpoint = std::move(missing_state);
  ASSERT_TRUE(drafter.Propose(next_aux, damaged, proposals));
  ASSERT_EQ(proposals.size(), 1u);
  ASSERT_FALSE(proposals.front().empty());
  EXPECT_NE(proposals.front().front(), expected.front());

  drafter.ReleaseAll();
  auto invalid = std::make_shared<Dflash2PrefixCheckpoint>();
  invalid->token_count = checkpoint->token_count;
  invalid->ring_blocks = checkpoint->ring_blocks;
  invalid->caches.resize(checkpoint->caches.size());
  damaged.front().prefix_checkpoint = std::move(invalid);
  EXPECT_THROW(drafter.Propose(next_aux, damaged, proposals), std::logic_error);
}

TEST(Dflash2ConfigTest, TrackedDsparkIngestsSampledTurnsAndResumesDrafting) {
  auto config = MakeDflash2Config();
  config.config_path = fs::path{MODEL_PATH "engine/synthetic-dspark"};
  auto& dspark = config.model.dflash2;
  dspark.filename = "dspark.onnx";
  dspark.is_dspark = true;
  dspark.num_hidden_layers = 1;
  dspark.num_key_value_heads = 1;
  dspark.head_size = 1;
  dspark.block_size = 4;
  dspark.num_draft_tokens = 4;
  dspark.sliding_window = 0;

  auto model = std::make_shared<Dflash2Model>(CreateDflash2Config(config), GetOrtEnv());
  Dflash2Drafter drafter{model, /*paged_block_size=*/4, /*num_blocks=*/6,
                         /*max_requests=*/1};

  int tracked_id = 0;
  int untracked_id = 0;
  auto* tracked = reinterpret_cast<Request*>(&tracked_id);
  auto* untracked = reinterpret_cast<Request*>(&untracked_id);
  auto* device = GetDeviceInterface(DeviceType::CPU);
  Tensor aux{device, Ort::TypeToTensorType<float>};
  const std::array<int64_t, 2> aux_shape{1, 1};
  aux.CreateTensor(aux_shape);
  std::vector<std::vector<int32_t>> drafts;

  const std::array initial{
      Dflash2Drafter::Feed{.request = tracked, .aux_row_count = 1, .anchor_token = 11, .draft_eligible = true, .wants_drafts = true},
  };
  EXPECT_TRUE(drafter.Propose(aux, initial, drafts));
  ASSERT_EQ(drafts.size(), 1u);
  EXPECT_FALSE(drafts[0].empty());

  const std::array sampled{
      Dflash2Drafter::Feed{.request = tracked, .aux_row_count = 1, .first_position = 1, .anchor_token = 12, .draft_eligible = false, .wants_drafts = false},
  };
  EXPECT_TRUE(drafter.Propose(aux, sampled, drafts));
  ASSERT_EQ(drafts.size(), 1u);
  EXPECT_TRUE(drafts[0].empty());

  const std::array resumed{
      Dflash2Drafter::Feed{.request = tracked, .aux_row_count = 1, .first_position = 2, .anchor_token = 13, .draft_eligible = true, .wants_drafts = true},
  };
  EXPECT_TRUE(drafter.Propose(aux, resumed, drafts));
  ASSERT_EQ(drafts.size(), 1u);
  EXPECT_FALSE(drafts[0].empty());

  drafter.Release(tracked);
  const std::array ineligible{
      Dflash2Drafter::Feed{.request = untracked, .aux_row_count = 1, .anchor_token = 14, .draft_eligible = false, .wants_drafts = false},
  };
  EXPECT_FALSE(drafter.Propose(aux, ineligible, drafts));
  EXPECT_EQ(drafter.AdmissionMisses(), 0u);
  EXPECT_FALSE(drafter.Propose(aux, {}, drafts));
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

TEST(Dflash2ConfigTest, RequiresUniqueRuntimeOutputNames) {
  auto config = MakeDflash2Config();
  config.model.dflash2.block_size = 9;
  config.model.dflash2.num_draft_tokens = 8;
  config.model.dflash2.head_size = 2;
  config.model.dflash2.outputs.scores = "present.0.key";
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.AddOutput("draft_candidate_ids", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1, 8, 2});
  for (int layer = 0; layer < 3; ++layer) {
    for (const auto* kind : {"key", "value"}) {
      drafter.AddInput("past_key_values." + std::to_string(layer) + "." + kind,
                       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 8, 2, 2});
      drafter.AddOutput("present." + std::to_string(layer) + "." + kind,
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 8, 2, 2});
    }
  }
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);

  config = MakeDflash2Config();
  config.model.dflash2.outputs.scores = config.model.dflash2.outputs.candidate_ids;
  std::tie(target, drafter) = MakeCompatibleMetadata();
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

TEST(Dflash2ConfigTest, GraphBlockTableLimitIncludesWorstCaseQuerySpill) {
  EXPECT_EQ(Dflash2GraphBlockTableColumnLimit(/*context_length=*/1024,
                                              /*paged_block_size=*/128,
                                              /*query_block_size=*/8),
            9u);
  EXPECT_EQ(Dflash2GraphBlockTableColumnLimit(/*context_length=*/1024,
                                              /*paged_block_size=*/128,
                                              /*query_block_size=*/129),
            10u);
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

TEST(Dflash2ConfigTest, ReportsWhenAProposalBufferMoves) {
  auto* device = GetDeviceInterface(DeviceType::CPU);
  constexpr auto type = Ort::TypeToTensorType<int32_t>;
  std::unique_ptr<Tensor> slot;

  // A CUDA graph records the address it was captured against, so the caller has to learn about
  // every move to stop replaying a graph that now points at a freed buffer.
  bool reallocated = false;
  Dflash2StepTensor(slot, device, type, {2, 4}, &reallocated);
  EXPECT_TRUE(reallocated);
  const void* buffer = slot->buffer_;

  reallocated = false;
  Dflash2StepTensor(slot, device, type, {1, 3}, &reallocated);
  EXPECT_FALSE(reallocated);

  std::unique_ptr<Tensor> displaced;
  Dflash2StepTensor(slot, device, type, {4, 8}, &reallocated, &displaced);
  EXPECT_TRUE(reallocated);
  ASSERT_NE(displaced, nullptr);
  EXPECT_EQ(displaced->buffer_, buffer);

  reallocated = false;
  Dflash2StepTensor(slot, device, type, {2, 4}, &reallocated);
  EXPECT_FALSE(reallocated);
}

TEST(Dflash2ConfigTest, FailedReplacementPreservesTheLiveProposalBuffer) {
  auto* device = GetDeviceInterface(DeviceType::CPU);
  std::unique_ptr<Tensor> slot;
  Dflash2StepTensor(slot, device, Ort::TypeToTensorType<int32_t>, {2, 4});
  const void* buffer = slot->buffer_;

  // The type change stages a replacement, while the invalid dimension makes CreateTensor fail.
  EXPECT_ANY_THROW(
      Dflash2StepTensor(slot, device, Ort::TypeToTensorType<float>, {0, -1}));
  EXPECT_EQ(slot->GetType(), Ort::TypeToTensorType<int32_t>);
  EXPECT_EQ(slot->buffer_, buffer);
}

TEST(Dflash2ConfigTest, AmortizesProposalBufferGrowth) {
  auto* device = GetDeviceInterface(DeviceType::CPU);
  constexpr auto type = Ort::TypeToTensorType<int32_t>;
  std::unique_ptr<Tensor> slot;

  Dflash2StepTensor(slot, device, type, {2, 4});
  // A full-attention drafter's block table gains one column every `block_size` committed tokens,
  // so growth doubles the buffer instead of tracking the shape exactly.
  Dflash2StepTensor(slot, device, type, {2, 5});
  const void* grown = slot->buffer_;
  ASSERT_NE(grown, nullptr);

  Dflash2StepTensor(slot, device, type, {2, 6});
  EXPECT_EQ(slot->buffer_, grown);
  Dflash2StepTensor(slot, device, type, {2, 8});
  EXPECT_EQ(slot->buffer_, grown);
  EXPECT_EQ(slot->GetElementCount(), 16u);
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

TEST(Dflash2ConfigTest, JoinsOnlyFromAnEligibleTurnAtSequenceStart) {
  EXPECT_TRUE(Dflash2CanJoin(/*draft_eligible=*/true, /*first_position=*/0));
  EXPECT_FALSE(Dflash2CanJoin(/*draft_eligible=*/false, /*first_position=*/0));
  EXPECT_FALSE(Dflash2CanJoin(/*draft_eligible=*/true, /*first_position=*/1));
}

TEST(Dflash2ConfigTest, RingCheckpointPreservesLogicalSlotsAcrossPhysicalRemapping) {
  auto* device = GetDeviceInterface(DeviceType::CPU);
  constexpr auto type = Ort::TypeToTensorType<uint8_t>;
  Tensor source{device, type}, checkpoint{device, type}, restored{device, type};
  source.CreateTensor(std::array<int64_t, 4>{6, 4, 1, 1});
  checkpoint.CreateTensor(std::array<int64_t, 4>{3, 4, 1, 1});
  restored.CreateTensor(std::array<int64_t, 4>{5, 4, 1, 1});
  std::array<uint8_t, 24> values{};
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<uint8_t>(i + 1);
  }
  source.GetByteSpan().CopyFromCpu(values);
  restored.GetByteSpan().Zero();

  const std::array<int32_t, 3> original{4, 1, 3};
  const std::array<int32_t, 3> compact{0, 1, 2};
  const std::array<int32_t, 3> relocated{2, 0, 4};
  CopyDflash2RingBlocks(checkpoint, compact, source, original);
  CopyDflash2RingBlocks(restored, relocated, checkpoint, compact);

  auto restored_span = restored.GetByteSpan();
  const auto copied = restored_span.CopyDeviceToCpu();
  for (size_t logical_slot = 0; logical_slot < original.size(); ++logical_slot) {
    for (size_t byte = 0; byte < 4; ++byte) {
      EXPECT_EQ(copied[relocated[logical_slot] * 4 + byte],
                values[original[logical_slot] * 4 + byte]);
    }
  }
  // A later absolute block wraps back to logical slot zero, not physical block zero.
  EXPECT_EQ(copied[relocated[12 % original.size()] * 4], values[original[0] * 4]);
  const std::array<int32_t, 3> invalid{6, 1, 3};
  EXPECT_THROW(CopyDflash2RingBlocks(checkpoint, compact, source, invalid), std::logic_error);
  Tensor scalar{device, type};
  scalar.CreateTensor(std::array<int64_t, 0>{});
  EXPECT_THROW(CopyDflash2RingBlocks(checkpoint, compact, scalar, original), std::logic_error);
}

namespace {

// Captures whatever the action logs as a warning.
template <typename Action>
std::string CapturedWarnings(Action&& action) {
  const fs_std::path log_path =
      fs_std::temp_directory_path() /
      ("draft_width_warning_" + std::to_string(reinterpret_cast<uintptr_t>(&action)) + ".log");
  fs_std::remove(log_path);
  SetLogString("filename", log_path.string());
  SetLogBool("enabled", true);
  SetLogBool("warning", true);

  action();

  SetLogString("filename", "");
  SetLogBool("enabled", false);
  std::ifstream stream{log_path};
  std::stringstream contents;
  contents << stream.rdbuf();
  stream.close();
  fs_std::remove(log_path);
  return contents.str();
}

// Captures whatever WarnOnClampedDraftWidth logs for one config.
std::string CapturedDraftWidthWarnings(const Config& config) {
  return CapturedWarnings([&] { WarnOnClampedDraftWidth(config); });
}

}  // namespace

TEST(Dflash2ConfigTest, WarnsWhenTheDrafterCannotSupplyTheConfiguredDraftWidth) {
  Config config = MakeDflash2Config();
  config.speculative.max_draft_tokens = 3;  // equals num_draft_tokens, nothing is clamped
  EXPECT_EQ(CapturedDraftWidthWarnings(config), "");

  config.speculative.max_draft_tokens = 5;
  EXPECT_NE(CapturedDraftWidthWarnings(config).find("model.dflash2.num_draft_tokens"),
            std::string::npos);

  config.model.dflash2.is_dspark = true;
  EXPECT_NE(CapturedDraftWidthWarnings(config).find("model.dspark.num_draft_tokens"),
            std::string::npos);
}

TEST(Dflash2ConfigTest, WarnsWhenARuntimeProfileRaisesDraftWidthBeyondTheDrafter) {
  Config config = MakeDflash2Config();
  config.speculative.max_draft_tokens = 3;
  Config::RuntimeProfile profile;
  profile.id = "large";
  profile.eligibility.minimum_total_device_memory_bytes = 1;
  profile.overlay.speculative.max_draft_tokens = 5;
  config.runtime_profiles.push_back(profile);

  const auto warnings = CapturedWarnings([&] { ApplyRuntimeProfile(config, 1); });
  EXPECT_EQ(config.speculative.max_draft_tokens, 5);
  EXPECT_NE(warnings.find("model.dflash2.num_draft_tokens"), std::string::npos);
}

TEST(Dflash2ConfigTest, DoesNotWarnAboutHostingLimitsAtConfigLoad) {
  Config config = MakeDflash2Config();
  // These bounds depend on which speculative path the Engine hosts, so the Engine reports them.
  config.model.dflash2.num_draft_tokens = 16;
  config.model.decoder.state_update_capacity = 1;
  config.speculative.max_draft_tokens = 16;
  EXPECT_EQ(CapturedDraftWidthWarnings(config), "");
}

TEST(Dflash2ConfigTest, DoesNotWarnAboutDraftWidthWithoutABlockDrafter) {
  Config config = MakeDflash2Config();
  // MTP's ceiling is an ONNX output that no session has loaded yet, so it cannot be checked here.
  config.model.dflash2.filename.clear();
  config.model.mtp.filename = "mtp.onnx";
  config.speculative.max_draft_tokens = 16;
  EXPECT_EQ(CapturedDraftWidthWarnings(config), "");
}

}  // namespace Generators::test
