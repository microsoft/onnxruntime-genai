// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

TEST(MtpDecoderConfigTest, RejectsInvalidConfiguration) {
  Config config;
  auto& mtp = config.model.mtp;
  mtp.filename = "mtp.onnx";
  mtp.num_hidden_layers = 1;
  mtp.num_key_value_heads = 2;
  mtp.head_size = 64;

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
}

}  // namespace Generators::test
