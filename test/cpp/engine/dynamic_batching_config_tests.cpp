// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include <gtest/gtest.h>

#include "config.h"

namespace Generators::test {
namespace {

Config LoadDynamicConfig(std::string_view dynamic_batching) {
  const std::string overlay =
      R"({ "engine": { "dynamic_batching": )" +
      std::string{dynamic_batching} + " } }";
  return Config{
      fs::path{std::string{MODEL_PATH "engine/dummy-decoder"}},
      overlay};
}

Config LoadDecoderConfig(std::string_view decoder) {
  const std::string overlay =
      R"({ "model": { "decoder": )" +
      std::string{decoder} + " } }";
  return Config{
      fs::path{std::string{MODEL_PATH "engine/dummy-decoder"}},
      overlay};
}

std::string CaptureStateGroupError(std::string_view decoder) {
  try {
    (void)LoadDecoderConfig(decoder);
  } catch (const std::runtime_error& error) {
    return error.what();
  }
  return {};
}

void ExpectStateGroupError(std::string_view decoder, std::string_view expected) {
  const auto message = CaptureStateGroupError(decoder);
  ASSERT_FALSE(message.empty());
  EXPECT_NE(message.find(expected), std::string::npos) << message;
}

TEST(DynamicBatchingConfigTest, ScheduledTokenBudgetDefaultsTo2048) {
  const auto config = LoadDynamicConfig(R"({ "max_batch_size": 4 })");

  ASSERT_TRUE(config.engine.dynamic_batching.has_value());
  EXPECT_EQ(config.engine.dynamic_batching->max_scheduled_tokens, 2048u);
}

TEST(DynamicBatchingConfigTest, ScheduledTokenBudgetAcceptsOverride) {
  const auto config =
      LoadDynamicConfig(R"({ "max_scheduled_tokens": 321 })");

  ASSERT_TRUE(config.engine.dynamic_batching.has_value());
  EXPECT_EQ(config.engine.dynamic_batching->max_scheduled_tokens, 321u);
}

class InvalidScheduledTokenBudgetTest
    : public ::testing::TestWithParam<const char*> {};

TEST_P(InvalidScheduledTokenBudgetTest, RejectsNonPositiveOrNonIntegralValue) {
  EXPECT_THROW(
      LoadDynamicConfig(
          std::string{R"({ "max_scheduled_tokens": )"} + GetParam() + " }"),
      std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidValues,
    InvalidScheduledTokenBudgetTest,
    ::testing::Values("0", "-1", "1.5", "4000000000"));

TEST(DecoderStateGroupsConfigTest, PreservesLegacyManifestAbsence) {
  const auto config = LoadDynamicConfig(R"({ "max_batch_size": 4 })");

  EXPECT_FALSE(config.model.decoder.state_groups.has_value());
}

TEST(DecoderStateGroupsConfigTest, UsesDefaultStateUpdateBindings) {
  const auto config = LoadDecoderConfig(R"({ "state_update_capacity": 3 })");

  EXPECT_EQ(config.model.decoder.inputs.state_update_capture_count, Config::Defaults::StateUpdateCaptureCountName);
  EXPECT_EQ(config.model.decoder.inputs.state_update_active, Config::Defaults::StateUpdateActiveName);
  EXPECT_EQ(config.model.decoder.outputs.state_update_conv_value_names, Config::Defaults::StateUpdateConvValueName);
  EXPECT_EQ(config.model.decoder.outputs.state_update_recurrent_capsule_names,
            Config::Defaults::StateUpdateRecurrentCapsuleName);
}

TEST(DecoderStateGroupsConfigTest, ParsesSparseHybridManifest) {
  const auto config = LoadDecoderConfig(R"({
    "num_hidden_layers": 4,
    "state_update_capacity": 3,
    "inputs": {
      "past_conv_names": "past_key_values.%d.conv_state",
      "past_recurrent_names": "past_key_values.%d.recurrent_state",
      "state_update_capture_count": "state_update_capture_count",
      "state_update_active": "state_update_active"
    },
    "outputs": {
      "present_conv_names": "present.%d.conv_state",
      "present_recurrent_names": "present.%d.recurrent_state",
      "state_update_conv_value_names": "state_update.%d.conv_value",
      "state_update_recurrent_capsule_names": "state_update.%d.recurrent_capsule"
    },
    "state_groups": [
      {
        "kind": "paged_kv",
        "layer_ids": [3]
      },
      {
        "kind": "fixed_conv",
        "layer_ids": [0, 1, 2]
      },
      {
        "kind": "fixed_recurrent",
        "layer_ids": [0, 1, 2]
      }
    ]
  })");

  ASSERT_TRUE(config.model.decoder.state_groups);
  const auto& state_groups = *config.model.decoder.state_groups;
  ASSERT_EQ(state_groups.size(), 3u);
  EXPECT_EQ(state_groups[0].kind, Config::Model::Decoder::StateGroupKind::PagedKeyValue);
  EXPECT_EQ(state_groups[0].layer_ids, std::vector<int>({3}));
  EXPECT_EQ(state_groups[1].kind, Config::Model::Decoder::StateGroupKind::FixedConv);
  EXPECT_EQ(state_groups[1].layer_ids, std::vector<int>({0, 1, 2}));
  EXPECT_EQ(state_groups[2].kind, Config::Model::Decoder::StateGroupKind::FixedRecurrent);
  EXPECT_EQ(state_groups[2].layer_ids, std::vector<int>({0, 1, 2}));
  EXPECT_EQ(config.model.decoder.inputs.past_conv_names, "past_key_values.%d.conv_state");
  EXPECT_EQ(config.model.decoder.inputs.past_recurrent_names, "past_key_values.%d.recurrent_state");
  EXPECT_EQ(config.model.decoder.outputs.present_conv_names, "present.%d.conv_state");
  EXPECT_EQ(config.model.decoder.outputs.present_recurrent_names, "present.%d.recurrent_state");
  EXPECT_EQ(config.model.decoder.state_update_capacity, 3);
  EXPECT_EQ(config.model.decoder.inputs.state_update_capture_count, "state_update_capture_count");
  EXPECT_EQ(config.model.decoder.inputs.state_update_active, "state_update_active");
  EXPECT_EQ(config.model.decoder.outputs.state_update_conv_value_names, "state_update.%d.conv_value");
  EXPECT_EQ(config.model.decoder.outputs.state_update_recurrent_capsule_names, "state_update.%d.recurrent_capsule");
}

TEST(DecoderStateGroupsConfigTest, OverlayValidationIsTransactional) {
  auto config = LoadDecoderConfig(R"({
    "num_hidden_layers": 1,
    "state_groups": [{
      "kind": "fixed_conv", "layer_ids": [0]
    }]
  })");

  EXPECT_THROW(
      OverlayConfig(config, R"({
        "model": {"decoder": {
          "state_groups": [{
            "kind": "fixed_conv", "layer_ids": [1]
          }]
        }}
      })"),
      std::runtime_error);

  ASSERT_TRUE(config.model.decoder.state_groups);
  ASSERT_EQ(config.model.decoder.state_groups->size(), 1u);
  EXPECT_EQ(config.model.decoder.state_groups->front().kind, Config::Model::Decoder::StateGroupKind::FixedConv);
  EXPECT_EQ(config.model.decoder.state_groups->front().layer_ids, std::vector<int>({0}));
}

TEST(DecoderStateGroupsConfigTest, RejectsMissingOrUnknownKind) {
  ExpectStateGroupError(
      R"({"num_hidden_layers": 1, "state_groups": [{
        "layer_ids": [0]
      }]})",
      "is missing kind");

  ExpectStateGroupError(
      R"({"num_hidden_layers": 1, "state_groups": [{
        "kind": "unknown", "layer_ids": [0]
      }]})",
      "Unsupported decoder state group kind 'unknown'");
}

TEST(DecoderStateGroupsConfigTest, AllowsIndependentFixedGroupsOnTheSameLayer) {
  EXPECT_NO_THROW(LoadDecoderConfig(
      R"({"num_hidden_layers": 1,
          "inputs": {"past_recurrent_names": "past_recurrent.%d"},
          "outputs": {"present_recurrent_names": "present_recurrent.%d"},
          "state_groups": [
            {"kind": "fixed_conv", "layer_ids": [0]},
            {"kind": "fixed_recurrent", "layer_ids": [0]}
          ]})"));
}

TEST(DecoderStateGroupsConfigTest, RejectsMalformedBindingTemplates) {
  ExpectStateGroupError(
      R"({"num_hidden_layers": 1,
          "inputs": {"past_recurrent_names": "past.recurrent"},
          "state_groups": [{"kind": "fixed_recurrent", "layer_ids": [0]}]})",
      "expected exactly one %d");

  ExpectStateGroupError(
      R"({"num_hidden_layers": 1,
          "inputs": {"past_recurrent_names": "past.%d.recurrent"},
          "outputs": {"present_recurrent_names": ""},
          "state_groups": [{"kind": "fixed_recurrent", "layer_ids": [0]}]})",
      "missing its output template");
}

TEST(DecoderStateGroupsConfigTest, RejectsDuplicateOrOutOfRangeLayerIds) {
  ExpectStateGroupError(
      R"({"num_hidden_layers": 2, "state_groups": [{
        "kind": "fixed_recurrent", "layer_ids": [0, 0]
      }]})",
      "duplicate layer_id 0");

  ExpectStateGroupError(
      R"({"num_hidden_layers": 2, "state_groups": [{
        "kind": "fixed_recurrent", "layer_ids": [2]
      }]})",
      "outside [0, num_hidden_layers)");
}

TEST(DecoderStateGroupsConfigTest, RejectsPagedLayerOverlap) {
  ExpectStateGroupError(
      R"({"num_hidden_layers": 1, "state_groups": [
        {"kind": "paged_kv", "layer_ids": [0]},
        {"kind": "paged_kv", "layer_ids": [0]}
      ]})",
      "overlaps another paged_kv group at layer_id 0");
}

TEST(DecoderStateGroupsConfigTest, RejectsDuplicateExpandedBindings) {
  ExpectStateGroupError(
      R"({"num_hidden_layers": 1,
          "inputs": {
            "past_conv_names": "past.%d",
            "past_recurrent_names": "past.%d"
          },
          "outputs": {
            "present_recurrent_names": "present_recurrent.%d"
          },
          "state_groups": [
            {"kind": "fixed_conv", "layer_ids": [0]},
            {"kind": "fixed_recurrent", "layer_ids": [0]}
          ]})",
      "resolves more than one binding to 'past.0'");
}

}  // namespace
}  // namespace Generators::test
