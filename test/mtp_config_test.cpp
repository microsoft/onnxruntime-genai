// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "ort_genai.h"
#include "config.h"

namespace Generators::test {
namespace {

namespace fs_std = std::filesystem;

fs_std::path WriteMtpConfig(const std::string& output_name) {
  const auto root = fs_std::temp_directory_path() / ("ortgenai_mtp_config_" + output_name);
  std::error_code ec;
  fs_std::remove_all(root, ec);
  fs_std::create_directories(root);

  const std::string config =
      "{ \"model\": { \"type\": \"tiny-test-model\","
      " \"vocab_size\": 16, \"context_length\": 32,"
      " \"decoder\": { \"filename\": \"model.onnx\" },"
      " \"mtp\": { \"filename\": \"mtp.onnx\","
      " \"main_hidden_states\": \"main_hidden\","
      " \"outputs\": { \"" +
      output_name +
      "\": \"head_feedback\" } } },"
      " \"search\": {} }";
  std::ofstream out(root / "genai_config.json", std::ios::binary);
  out << config;
  return root;
}

fs_std::path WriteSharedInitializerConfig(const std::string& suffix, const std::string& shape) {
  const auto root = fs_std::temp_directory_path() / ("ortgenai_shared_initializer_config_" + suffix);
  std::error_code ec;
  fs_std::remove_all(root, ec);
  fs_std::create_directories(root);

  const std::string config =
      "{ \"model\": { \"type\": \"tiny-test-model\","
      " \"vocab_size\": 16, \"context_length\": 32,"
      " \"decoder\": { \"filename\": \"model.onnx\","
      " \"shared_initializers\": [{ \"name\": \"weight\", \"data_file\": \"weights.bin\","
      " \"length\": \"1\", \"data_type\": 2, \"shape\": [" +
      shape +
      "] }] } },"
      " \"search\": {} }";
  std::ofstream out(root / "genai_config.json", std::ios::binary);
  out << config;
  return root;
}

}  // namespace

TEST(MtpConfigTest, AcceptsConfigurableFeedbackOutput) {
  const auto root = WriteMtpConfig("hidden_states");
  EXPECT_NO_THROW(OgaConfig::Create(root.string().c_str()));
}

TEST(MtpConfigTest, RejectsMisspelledFeedbackOutput) {
  const auto root = WriteMtpConfig("hidden_state");
  EXPECT_THROW(OgaConfig::Create(root.string().c_str()), std::exception);
}

TEST(MtpConfigTest, AcceptsInt64SharedInitializerDimension) {
  const auto root = WriteSharedInitializerConfig("int64", "4294967296");
  EXPECT_NO_THROW(OgaConfig::Create(root.string().c_str()));
}

TEST(MtpConfigTest, RejectsFractionalSharedInitializerDimension) {
  const auto root = WriteSharedInitializerConfig("fractional", "1.5");
  EXPECT_THROW(OgaConfig::Create(root.string().c_str()), std::exception);
}

TEST(MtpConfigTest, RejectsOutOfRangeSharedInitializerDimension) {
  const auto root = WriteSharedInitializerConfig("overflow", "9223372036854775808");
  EXPECT_THROW(OgaConfig::Create(root.string().c_str()), std::exception);
}

TEST(MtpConfigTest, ProjectsPagedMtpDecoderWithoutMainFixedState) {
  Config config;
  auto& decoder = config.model.decoder;
  decoder.filename = "text.onnx";
  decoder.num_hidden_layers = 64;
  decoder.num_key_value_heads = 8;
  decoder.head_size = 128;
  decoder.inputs.block_table = "block_table";
  decoder.inputs.cumulative_sequence_lengths = "cumulative_sequence_lengths";
  decoder.inputs.past_sequence_lengths = "past_sequence_lengths";
  decoder.inputs.attention_metadata = "attention_metadata";
  decoder.sliding_window = Config::Model::Decoder::SlidingWindow{4096};
  decoder.state_groups.emplace();
  decoder.state_groups->push_back(Config::Model::Decoder::StateGroup{
      Config::Model::Decoder::StateGroupKind::Fixed});

  auto& mtp = config.model.mtp;
  mtp.filename = "mtp.onnx";
  mtp.num_hidden_layers = 1;
  mtp.num_key_value_heads = 2;
  mtp.head_size = 64;
  mtp.inputs.hidden_states = "head_hidden";
  mtp.outputs.hidden_states = "head_hidden_out";

  auto projected = CreateMtpDecoderConfig(config);
  const auto& head = projected->model.decoder;
  EXPECT_EQ(head.filename, "mtp.onnx");
  EXPECT_EQ(head.num_hidden_layers, 1);
  EXPECT_EQ(head.num_key_value_heads, 2);
  EXPECT_EQ(head.head_size, 64);
  EXPECT_EQ(head.inputs.hidden_states, "head_hidden");
  EXPECT_EQ(head.outputs.hidden_states, "head_hidden_out");
  EXPECT_EQ(head.inputs.block_table, "block_table");
  EXPECT_EQ(head.inputs.cumulative_sequence_lengths, "cumulative_sequence_lengths");
  EXPECT_EQ(head.inputs.past_sequence_lengths, "past_sequence_lengths");
  EXPECT_EQ(head.inputs.attention_metadata, "attention_metadata");
  EXPECT_FALSE(head.sliding_window.has_value());
  EXPECT_FALSE(head.state_groups.has_value());
  ASSERT_EQ(head.layer_types.size(), 1u);
  EXPECT_EQ(head.layer_types[0], "full_attention");
}

}  // namespace Generators::test
