// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdexcept>
#include <string_view>

#include <gtest/gtest.h>

#include "config.h"

namespace Generators::test {

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
  decoder.session_options.providers = {"cuda"};
  decoder.session_options.intra_op_num_threads = 2;
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
  ASSERT_EQ(head.layer_types.size(), 1u);
  EXPECT_EQ(head.layer_types[0], "full_attention");
  EXPECT_THROW(CreateMtpDecoderConfig(*projected), std::runtime_error);
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

}  // namespace Generators::test
