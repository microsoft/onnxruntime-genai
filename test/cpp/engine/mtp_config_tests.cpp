// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "config.h"
#include "engine/engine.h"

namespace Generators::test {
namespace {

namespace fs_std = std::filesystem;

fs_std::path WriteDisabledMtpConfig() {
  const auto root = fs_std::temp_directory_path() / "ortgenai_mtp_config_disabled";
  std::error_code ec;
  fs_std::remove_all(root, ec);
  fs_std::create_directories(root);
  std::ofstream out(root / "genai_config.json", std::ios::binary);
  out << R"({
    "model": {
      "type": "tiny-test-model",
      "vocab_size": 16,
      "context_length": 32,
      "decoder": { "filename": "model.onnx" },
      "mtp": {
        "enabled": false,
        "filename": "mtp.onnx",
        "main_hidden_states": "main_hidden",
        "outputs": { "hidden_states": "head_feedback" }
      }
    },
    "search": {}
  })";
  return root;
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

}  // namespace

TEST(MtpDecoderConfigTest, ParsesDisabledRuntimeToggle) {
  const auto root = WriteDisabledMtpConfig();
  const Config config{fs::path{root.string()}, {}};

  EXPECT_FALSE(config.model.mtp.enabled);
  EXPECT_FALSE(config.model.mtp.IsEnabled());
  EXPECT_EQ(config.model.mtp.filename, "mtp.onnx");
}

TEST(MtpDecoderConfigTest, IndexShareIsOptInAndRequiresCompleteMetadata) {
  Config config;
  config.model.decoder.hidden_size = 16;
  auto& mtp = config.model.mtp;
  mtp.filename = "mtp.onnx";
  mtp.num_key_value_heads = 1;
  mtp.head_size = 16;
  mtp.shared_initializers.push_back({"lm_head", "model.onnx.data"});
  EXPECT_FALSE(mtp.index_share.enabled);
  EXPECT_NO_THROW(CreateMtpDecoderConfig(config));
  mtp.index_share.enabled = true;
  EXPECT_THROW(CreateMtpDecoderConfig(config), std::runtime_error);
  mtp.index_share.indices_output = "indices";
  mtp.index_share.counts_output = "counts";
  mtp.index_share.base_capacity = 2051;
  mtp.index_share.max_draft_tokens = 7;
  auto projected = CreateMtpDecoderConfig(config);
  ASSERT_EQ(projected->model.decoder.shared_initializers.size(), 1u);
  EXPECT_EQ(projected->model.decoder.shared_initializers[0].name, "lm_head");
  EXPECT_EQ(projected->model.decoder.filename, "mtp.onnx");
  mtp.index_share.max_draft_tokens = 1;
  EXPECT_NO_THROW(CreateMtpDecoderConfig(config));
  mtp.index_share.max_draft_tokens = 0;
  EXPECT_THROW(CreateMtpDecoderConfig(config), std::runtime_error);
  mtp.index_share.enabled = false;
  EXPECT_THROW(CreateMtpDecoderConfig(config), std::runtime_error);
  mtp.index_share.max_draft_tokens = 7;
  projected = CreateMtpDecoderConfig(config);
  EXPECT_EQ(projected->model.decoder.shared_initializers[0].name, "lm_head");
}

TEST(MtpDecoderConfigTest, SingleModelIndexShareUsesMtpFilenameForEveryBudget) {
  Config config;
  config.model.decoder.hidden_size = 16;
  auto& mtp = config.model.mtp;
  mtp.filename = "mtp.onnx";
  mtp.num_key_value_heads = 1;
  mtp.head_size = 16;
  mtp.index_share.indices_output = "indices";
  mtp.index_share.counts_output = "counts";
  mtp.index_share.base_capacity = 2051;
  mtp.shared_initializers.push_back({"expert", "mtp.onnx.data"});
  for (int budget = 1; budget <= 7; ++budget) {
    mtp.index_share.max_draft_tokens = budget;
    const auto projected = CreateMtpDecoderConfig(config);
    EXPECT_EQ(projected->model.decoder.filename, "mtp.onnx");
    ASSERT_EQ(projected->model.decoder.shared_initializers.size(), 1u);
    EXPECT_EQ(projected->model.decoder.shared_initializers[0].name, "expert");
  }
  mtp.index_share.base_capacity = std::numeric_limits<int>::max();
  EXPECT_THROW(CreateMtpDecoderConfig(config), std::runtime_error);
}

TEST(MtpDecoderConfigTest, ProjectsPagedDecoderWithoutMainFixedState) {
  Config config;
  auto& decoder = config.model.decoder;
  decoder.filename = "text.onnx";
  decoder.num_hidden_layers = 64;
  decoder.num_key_value_heads = 8;
  decoder.head_size = 128;
  decoder.hidden_size = 2048;
  decoder.inputs.block_table = "block_table";
  decoder.inputs.cumulative_sequence_lengths = "cumulative_sequence_lengths";
  decoder.inputs.past_sequence_lengths = "past_sequence_lengths";
  decoder.inputs.attention_metadata = "attention_metadata";
  decoder.inputs.engram_embeddings = "engram_embeddings";
  config.model.engram.filename = "engram.onnx";
  decoder.session_options.providers = {"cuda"};
  decoder.session_options.intra_op_num_threads = 2;
  decoder.session_options.use_deterministic_compute = true;
  decoder.session_options.config_entries = {
      {"parent_entry", "keep"}, {"overridden_entry", "parent"}};
  decoder.session_options.provider_options.push_back(
      {"cuda", {{"device_id", "1"}, {"arena_extend_strategy", "kSameAsRequested"}}});
  decoder.run_options = Config::RunOptions{
      {"parent_run_option", "keep"}, {"overridden_run_option", "parent"}};
  decoder.shared_initializers.push_back({"weight", "weights.bin"});
  decoder.state_update_capacity = 4;
  decoder.sliding_window = Config::Model::Decoder::SlidingWindow{4096};
  decoder.state_groups.emplace();
  decoder.state_groups->push_back(Config::Model::Decoder::StateGroup{
      Config::Model::Decoder::StateGroupKind::FixedConv});
  decoder.pipeline.push_back({"preprocess.onnx"});

  auto& mtp = config.model.mtp;
  mtp.filename = "mtp.onnx";
  mtp.num_hidden_layers = 1;
  mtp.num_key_value_heads = 2;
  mtp.head_size = 64;
  mtp.inputs.hidden_states = "head_hidden";
  mtp.outputs.hidden_states = "head_hidden_out";
  mtp.session_options.emplace();
  mtp.session_options->graph_optimization_level = ORT_DISABLE_ALL;
  mtp.session_options->config_entries = {
      {"overridden_entry", "mtp"}, {"mtp_entry", "head"}};
  mtp.session_options->provider_options.push_back(
      {"cuda", {{"arena_extend_strategy", "kNextPowerOfTwo"}}});
  mtp.run_options = Config::RunOptions{
      {"overridden_run_option", "mtp"}, {"mtp_run_option", "head"}};

  auto projected = CreateMtpDecoderConfig(config);
  const auto& head = projected->model.decoder;
  EXPECT_EQ(head.filename, "mtp.onnx");
  EXPECT_EQ(head.num_hidden_layers, 1);
  EXPECT_EQ(head.num_key_value_heads, 2);
  EXPECT_EQ(head.head_size, 64);
  EXPECT_EQ(head.hidden_size, 2048);
  EXPECT_EQ(head.inputs.hidden_states, "head_hidden");
  EXPECT_EQ(head.outputs.hidden_states, "head_hidden_out");
  EXPECT_EQ(head.inputs.block_table, "block_table");
  EXPECT_EQ(head.inputs.cumulative_sequence_lengths, "cumulative_sequence_lengths");
  EXPECT_EQ(head.inputs.past_sequence_lengths, "past_sequence_lengths");
  EXPECT_EQ(head.inputs.attention_metadata, "attention_metadata");
  EXPECT_EQ(head.session_options.graph_optimization_level, ORT_DISABLE_ALL);
  EXPECT_EQ(head.session_options.intra_op_num_threads, 2);
  EXPECT_EQ(head.session_options.use_deterministic_compute, true);
  mtp.session_options->use_deterministic_compute = false;
  EXPECT_EQ(CreateMtpDecoderConfig(config)->model.decoder.session_options.use_deterministic_compute, false);
  EXPECT_EQ(head.session_options.config_entries,
            (std::vector<Config::NamedString>{{"overridden_entry", "mtp"},
                                              {"mtp_entry", "head"},
                                              {"parent_entry", "keep"}}));
  ASSERT_EQ(head.session_options.providers.size(), 1u);
  EXPECT_EQ(head.session_options.providers[0], "cuda");
  ASSERT_EQ(head.session_options.provider_options.size(), 1u);
  const auto& provider_options = head.session_options.provider_options[0];
  EXPECT_EQ(provider_options.name, "cuda");
  ASSERT_EQ(provider_options.options.size(), 2u);
  EXPECT_EQ(provider_options.options[0],
            Config::NamedString("arena_extend_strategy", "kNextPowerOfTwo"));
  EXPECT_EQ(provider_options.options[1], Config::NamedString("device_id", "1"));
  ASSERT_TRUE(head.run_options.has_value());
  EXPECT_EQ(*head.run_options,
            (Config::RunOptions{{"overridden_run_option", "mtp"},
                                {"mtp_run_option", "head"},
                                {"parent_run_option", "keep"}}));
  EXPECT_TRUE(head.shared_initializers.empty());
  EXPECT_EQ(head.state_update_capacity, 0);
  EXPECT_FALSE(head.sliding_window.has_value());
  EXPECT_FALSE(head.state_groups.has_value());
  EXPECT_TRUE(head.pipeline.empty());
  EXPECT_TRUE(projected->model.mtp.filename.empty());
  EXPECT_TRUE(projected->model.engram.filename.empty());
  EXPECT_EQ(config.model.engram.filename, "engram.onnx");
  // The projection clears model.mtp, so the head's own demand for hidden states must be recorded
  // explicitly. Without it a chained draft cannot feed the next stage.
  EXPECT_TRUE(projected->engine.hidden_states_output_required);
  ASSERT_EQ(head.layer_types.size(), 1u);
  EXPECT_EQ(head.layer_types[0], "full_attention");
  EXPECT_THROW(CreateMtpDecoderConfig(*projected), std::runtime_error);
}

TEST(MtpDecoderConfigTest, ProjectsOneLayerFixedIndexerStateForTheHead) {
  Config config;
  auto& decoder = config.model.decoder;
  decoder.num_hidden_layers = 48;
  decoder.num_key_value_heads = 2;
  decoder.head_size = 256;
  decoder.hidden_size = 2560;
  decoder.state_update_capacity = 7;
  decoder.inputs.past_indexer_kv_buffer_names = "past.%d.indexer_kv_buffer";
  decoder.inputs.past_indexer_state_lengths_names = "past.%d.indexer_state_lengths";
  decoder.outputs.present_indexer_kv_buffer_names = "present.%d.indexer_kv_buffer";
  decoder.outputs.present_indexer_state_lengths_names = "present.%d.indexer_state_lengths";
  decoder.outputs.state_update_indexer_names = "state_update.%d.indexer";
  decoder.state_groups = std::vector<Config::Model::Decoder::StateGroup>{
      {Config::Model::Decoder::StateGroupKind::PagedKeyValue,
       {3, 7},
       std::nullopt},
      {Config::Model::Decoder::StateGroupKind::FixedIndexer,
       {3, 7},
       Config::Model::Decoder::StateUpdate{7, true, 0, 4}}};

  auto& mtp = config.model.mtp;
  mtp.filename = "mtp.onnx";
  mtp.num_hidden_layers = 1;
  mtp.num_key_value_heads = 2;
  mtp.head_size = 256;
  mtp.inputs.past_indexer_names = "past.%d.indexer_key";
  mtp.outputs.present_indexer_names = "present.%d.indexer_key";

  const auto projected = CreateMtpDecoderConfig(config);
  const auto& head = projected->model.decoder;
  ASSERT_TRUE(head.state_groups.has_value());
  ASSERT_EQ(head.state_groups->size(), 2u);
  EXPECT_EQ((*head.state_groups)[0].kind,
            Config::Model::Decoder::StateGroupKind::PagedKeyValue);
  EXPECT_EQ((*head.state_groups)[0].layer_ids, std::vector<int>{0});
  const auto& indexer = (*head.state_groups)[1];
  EXPECT_EQ(indexer.kind,
            Config::Model::Decoder::StateGroupKind::FixedIndexer);
  EXPECT_EQ(indexer.layer_ids, std::vector<int>{0});
  ASSERT_TRUE(indexer.state_update.has_value());
  EXPECT_EQ(indexer.state_update->capacity, 7);
  EXPECT_FALSE(indexer.state_update->enabled);
  EXPECT_EQ(indexer.state_update->compress_ratio, 4);
  EXPECT_EQ(head.state_update_capacity, 7);
  EXPECT_EQ(head.inputs.past_indexer_kv_buffer_names,
            "past.%d.indexer_kv_buffer");
  EXPECT_EQ(head.outputs.state_update_indexer_names,
            "state_update.%d.indexer");
}

// A per-token quantized target declares scale name templates on its decoder. The MTP projection
// copies the target config wholesale, and the head is always an unquantized full-attention layer,
// so those templates must be cleared. If they survived, the engine would size and bind the head's
// cache against scale tensor names that exist only in the target session.
TEST(MtpDecoderConfigTest, ClearsInheritedScaleTemplatesForTheUnquantizedHead) {
  Config config;
  auto& decoder = config.model.decoder;
  decoder.filename = "text.onnx";
  decoder.num_hidden_layers = 64;
  decoder.num_key_value_heads = 8;
  decoder.head_size = 128;
  decoder.hidden_size = 2048;
  decoder.inputs.past_key_scale_names = "past_key_values.%d.key_scale";
  decoder.inputs.past_value_scale_names = "past_key_values.%d.value_scale";
  decoder.outputs.present_key_scale_names = "present.%d.key_scale";
  decoder.outputs.present_value_scale_names = "present.%d.value_scale";

  auto& mtp = config.model.mtp;
  mtp.filename = "mtp.onnx";
  mtp.num_hidden_layers = 1;
  mtp.num_key_value_heads = 2;
  mtp.head_size = 64;

  const auto projected = CreateMtpDecoderConfig(config);
  const auto& head = projected->model.decoder;
  EXPECT_TRUE(head.inputs.past_key_scale_names.empty());
  EXPECT_TRUE(head.inputs.past_value_scale_names.empty());
  EXPECT_TRUE(head.outputs.present_key_scale_names.empty());
  EXPECT_TRUE(head.outputs.present_value_scale_names.empty());

  // The target's own configuration is untouched: only the projected copy is unquantized.
  EXPECT_EQ(config.model.decoder.inputs.past_key_scale_names, "past_key_values.%d.key_scale");
  EXPECT_EQ(config.model.decoder.outputs.present_value_scale_names, "present.%d.value_scale");
}

TEST(MtpDecoderConfigTest, RejectsInvalidConfiguration) {
  Config config;
  auto& mtp = config.model.mtp;
  mtp.filename = "mtp.onnx";
  mtp.num_hidden_layers = 1;
  mtp.num_key_value_heads = 2;
  mtp.head_size = 64;
  config.model.decoder.hidden_size = 2048;

  const auto expect_error = [&config](std::string_view expected) {
    try {
      static_cast<void>(CreateMtpDecoderConfig(config));
      FAIL() << "Expected invalid MTP decoder configuration to be rejected";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string_view{error.what()}.find(expected), std::string_view::npos);
    }
  };

  mtp.enabled = false;
  expect_error("model.mtp.enabled");
  mtp.enabled = true;

  mtp.filename.clear();
  expect_error("filename");
  mtp.filename = "mtp.onnx";

  for (int invalid_layer_count : {0, 2}) {
    mtp.num_hidden_layers = invalid_layer_count;
    expect_error("num_hidden_layers must be 1");
  }
  mtp.num_hidden_layers = 1;

  mtp.num_key_value_heads = 0;
  expect_error("KV head count and head size must be positive");
  mtp.num_key_value_heads = 2;

  mtp.head_size = 0;
  expect_error("KV head count and head size must be positive");
  mtp.head_size = 64;

  config.model.decoder.hidden_size = 0;
  expect_error("model.decoder.hidden_size must be positive");
}

TEST(MtpDecoderConfigTest, ValidatesMainAndHeadHiddenStateContract) {
  Config config;
  config.model.decoder.hidden_size = 2048;
  config.model.mtp.main_hidden_states = "main_hidden";
  config.model.mtp.inputs.hidden_states = "head_hidden";

  FakeModelStateMetadata target;
  target.AddOutput("main_hidden", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                   {-1, 2048});
  FakeModelStateMetadata head;
  head.AddInput("head_hidden", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                {-1, 2048});
  EXPECT_NO_THROW(ValidateMtpModelCompatibility(config, target, head));

  config.model.mtp.main_hidden_states = "missing";
  EXPECT_THROW(ValidateMtpModelCompatibility(config, target, head),
               std::runtime_error);
  config.model.mtp.main_hidden_states = "main_hidden";

  FakeModelStateMetadata wrong_width;
  wrong_width.AddInput("head_hidden", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                       {-1, 1024});
  EXPECT_THROW(ValidateMtpModelCompatibility(config, target, wrong_width),
               std::runtime_error);

  FakeModelStateMetadata wrong_type;
  wrong_type.AddInput("head_hidden", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                      {-1, 2048});
  EXPECT_THROW(ValidateMtpModelCompatibility(config, target, wrong_type),
               std::runtime_error);
}

TEST(MtpDecoderConfigTest, AcceptsMatchingHiddenStateWidthIndependentOfDecoderWidth) {
  Config config;
  config.model.decoder.hidden_size = 2048;
  config.model.mtp.main_hidden_states = "main_hidden";
  config.model.mtp.inputs.hidden_states = "head_hidden";

  FakeModelStateMetadata target;
  target.AddOutput("main_hidden", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                   {-1, 8192});
  FakeModelStateMetadata head;
  head.AddInput("head_hidden", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                {-1, 8192});

  EXPECT_NO_THROW(ValidateMtpModelCompatibility(config, target, head));
}

}  // namespace Generators::test
