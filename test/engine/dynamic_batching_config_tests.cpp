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

TEST(DecoderStateGroupsConfigTest, ParsesSparseHybridManifest) {
  const auto config = LoadDecoderConfig(R"({
    "num_hidden_layers": 4,
    "state_groups": [
      {
        "kind": "paged_kv",
        "layer_ids": [3],
        "bindings": {
          "key": {"input": "past_key_values.%d.key", "output": "present.%d.key"},
          "value": {"input": "past_key_values.%d.value", "output": "present.%d.value"}
        }
      },
      {
        "kind": "fixed",
        "layer_ids": [0, 1, 2],
        "bindings": {
          "state": {"input": "past_key_values.%d.conv_state", "output": "present.%d.conv_state"}
        }
      },
      {
        "kind": "fixed",
        "layer_ids": [0, 1, 2],
        "bindings": {
          "state": {"input": "past_key_values.%d.recurrent_state", "output": "present.%d.recurrent_state"}
        }
      }
    ]
  })");

  ASSERT_TRUE(config.model.decoder.state_groups);
  const auto& state_groups = *config.model.decoder.state_groups;
  ASSERT_EQ(state_groups.size(), 3u);
  EXPECT_EQ(state_groups[0].kind, Config::Model::Decoder::StateGroupKind::PagedKeyValue);
  EXPECT_EQ(state_groups[0].layer_ids, std::vector<int>({3}));
  EXPECT_EQ(state_groups[1].kind, Config::Model::Decoder::StateGroupKind::Fixed);
  EXPECT_EQ(state_groups[1].layer_ids, std::vector<int>({0, 1, 2}));
  EXPECT_EQ(state_groups[2].kind, Config::Model::Decoder::StateGroupKind::Fixed);
  EXPECT_EQ(state_groups[2].layer_ids, std::vector<int>({0, 1, 2}));
}

TEST(DecoderStateGroupsConfigTest, OverlayValidationIsTransactional) {
  auto config = LoadDecoderConfig(R"({
    "num_hidden_layers": 1,
    "state_groups": [{
      "kind": "fixed",
      "layer_ids": [0],
      "bindings": {
        "state": {"input": "past_conv.%d", "output": "present_conv.%d"}
      }
    }]
  })");

  EXPECT_THROW(
      OverlayConfig(config, R"({
        "model": {"decoder": {
          "state_groups": [{
            "kind": "fixed",
            "layer_ids": [1],
            "bindings": {
              "state": {"input": "past.%d", "output": "present.%d"}
            }
          }]
        }}
      })"),
      std::runtime_error);

  ASSERT_TRUE(config.model.decoder.state_groups);
  ASSERT_EQ(config.model.decoder.state_groups->size(), 1u);
  EXPECT_EQ(config.model.decoder.state_groups->front().kind, Config::Model::Decoder::StateGroupKind::Fixed);
  EXPECT_EQ(config.model.decoder.state_groups->front().layer_ids, std::vector<int>({0}));
}

TEST(DecoderStateGroupsConfigTest, RejectsMissingOrUnknownKind) {
  ExpectStateGroupError(
      R"({"num_hidden_layers": 1, "state_groups": [{
        "layer_ids": [0],
        "bindings": {"state": {"input": "past.%d", "output": "present.%d"}}
      }]})",
      "is missing kind");

  ExpectStateGroupError(
      R"({"num_hidden_layers": 1, "state_groups": [{
        "kind": "unknown", "layer_ids": [0],
        "bindings": {"state": {"input": "past.%d", "output": "present.%d"}}
      }]})",
      "Unsupported decoder state group kind 'unknown'");
}

TEST(DecoderStateGroupsConfigTest, AllowsIndependentFixedGroupsOnTheSameLayer) {
  EXPECT_NO_THROW(LoadDecoderConfig(
      R"({"num_hidden_layers": 1, "state_groups": [
        {"kind": "fixed", "layer_ids": [0],
         "bindings": {"state": {"input": "past_conv.%d", "output": "present_conv.%d"}}},
        {"kind": "fixed", "layer_ids": [0],
         "bindings": {"state": {"input": "past_recurrent.%d", "output": "present_recurrent.%d"}}}
      ]})"));
}

TEST(DecoderStateGroupsConfigTest, RejectsMissingOrIncompatibleBindings) {
  ExpectStateGroupError(
      R"({"num_hidden_layers": 1, "state_groups": [{
        "kind": "paged_kv", "layer_ids": [0],
        "bindings": {"key": {"input": "past.%d.key", "output": "present.%d.key"}}
      }]})",
      "requires exactly key and value bindings");

  ExpectStateGroupError(
      R"({"num_hidden_layers": 1, "state_groups": [{
        "kind": "fixed", "layer_ids": [0],
        "bindings": {
          "key": {"input": "past.%d.key", "output": "present.%d.key"},
          "state": {"input": "past.%d.state", "output": "present.%d.state"}
        }
      }]})",
      "requires exactly one state binding");
}

TEST(DecoderStateGroupsConfigTest, RejectsMalformedBindingTemplates) {
  ExpectStateGroupError(
      R"({"num_hidden_layers": 1, "state_groups": [{
        "kind": "fixed", "layer_ids": [0],
        "bindings": {"state": {"input": "past.recurrent", "output": "present.%d.recurrent"}}
      }]})",
      "expected exactly one %d");

  ExpectStateGroupError(
      R"({"num_hidden_layers": 1, "state_groups": [{
        "kind": "fixed", "layer_ids": [0],
        "bindings": {"state": {"input": "past.%d.recurrent"}}
      }]})",
      "missing its output template");
}

TEST(DecoderStateGroupsConfigTest, RejectsDuplicateOrOutOfRangeLayerIds) {
  ExpectStateGroupError(
      R"({"num_hidden_layers": 2, "state_groups": [{
        "kind": "fixed", "layer_ids": [0, 0],
        "bindings": {"state": {"input": "past.%d.recurrent", "output": "present.%d.recurrent"}}
      }]})",
      "duplicate layer_id 0");

  ExpectStateGroupError(
      R"({"num_hidden_layers": 2, "state_groups": [{
        "kind": "fixed", "layer_ids": [2],
        "bindings": {"state": {"input": "past.%d.recurrent", "output": "present.%d.recurrent"}}
      }]})",
      "outside [0, num_hidden_layers)");
}

TEST(DecoderStateGroupsConfigTest, RejectsPagedLayerOverlap) {
  ExpectStateGroupError(
      R"({"num_hidden_layers": 1, "state_groups": [
        {"kind": "paged_kv", "layer_ids": [0],
         "bindings": {
           "key": {"input": "past_a.%d.key", "output": "present_a.%d.key"},
           "value": {"input": "past_a.%d.value", "output": "present_a.%d.value"}
         }},
        {"kind": "paged_kv", "layer_ids": [0],
         "bindings": {
           "key": {"input": "past_b.%d.key", "output": "present_b.%d.key"},
           "value": {"input": "past_b.%d.value", "output": "present_b.%d.value"}
         }}
      ]})",
      "overlaps another paged_kv group at layer_id 0");
}

TEST(DecoderStateGroupsConfigTest, RejectsDuplicateExpandedBindings) {
  ExpectStateGroupError(
      R"({"num_hidden_layers": 1, "state_groups": [
        {"kind": "fixed", "layer_ids": [0],
         "bindings": {"state": {"input": "past.%d", "output": "present_conv.%d"}}},
        {"kind": "fixed", "layer_ids": [0],
         "bindings": {"state": {"input": "past.%d", "output": "present_recurrent.%d"}}}
      ]})",
      "resolves more than one binding to 'past.0'");
}

TEST(DecoderStateGroupsConfigTest, ParsesCompactStateUpdatesAlongsideCheckpoints) {
  const auto config = LoadDecoderConfig(R"({
    "num_hidden_layers": 2,
    "state_groups": [
      {
        "kind": "fixed", "layer_ids": [0],
        "checkpoint_count": 4, "checkpoint_alignment": "left",
        "bindings": {"state": {
          "input": "past.%d.conv", "output": "present.%d.conv",
          "checkpoints": "checkpoints.%d.conv"
        }},
        "state_update": {
          "enabled": false,
          "kind": "causal_conv", "capacity": 3,
          "capture_count": "state_update_capture_count",
          "active": "state_update_active",
          "value": "state_update.%d.conv_value"
        }
      },
      {
        "kind": "fixed", "layer_ids": [1],
        "bindings": {"state": {
          "input": "past.%d.recurrent", "output": "present.%d.recurrent"
        }},
        "state_update": {
          "enabled": false,
          "kind": "gated_delta_net", "capacity": 3,
          "capture_count": "state_update_capture_count",
          "active": "state_update_active",
          "decay": "state_update.%d.decay",
          "key": "state_update.%d.key",
          "delta": "state_update.%d.delta"
        }
      }
    ]
  })");

  const auto& groups = *config.model.decoder.state_groups;
  ASSERT_EQ(groups.size(), 2u);
  ASSERT_TRUE(groups[0].state_update);
  EXPECT_FALSE(groups[0].state_update->enabled);
  EXPECT_EQ(groups[0].state_update->kind,
            Config::Model::Decoder::StateUpdateKind::CausalConv);
  EXPECT_EQ(groups[0].state_update->capacity, 3);
  EXPECT_EQ(groups[0].state_update->capture_count, "state_update_capture_count");
  EXPECT_EQ(groups[0].state_update->active, "state_update_active");
  EXPECT_EQ(groups[0].state_update->value, "state_update.%d.conv_value");
  EXPECT_EQ(groups[0].checkpoint_count, 4);
  ASSERT_TRUE(groups[1].state_update);
  EXPECT_FALSE(groups[1].state_update->enabled);
  EXPECT_EQ(groups[1].state_update->kind,
            Config::Model::Decoder::StateUpdateKind::GatedDeltaNet);
  EXPECT_EQ(groups[1].state_update->active, "state_update_active");
  EXPECT_EQ(groups[1].state_update->decay, "state_update.%d.decay");
  EXPECT_EQ(groups[1].state_update->key, "state_update.%d.key");
  EXPECT_EQ(groups[1].state_update->delta, "state_update.%d.delta");
}

TEST(DecoderStateGroupsConfigTest, RejectsMalformedCompactStateUpdateKindAndFields) {
  ExpectStateGroupError(
      R"({"num_hidden_layers": 1, "state_groups": [{
        "kind": "fixed", "layer_ids": [0],
        "bindings": {"state": {"input": "past.%d", "output": "present.%d"}},
        "state_update": {"kind": "unknown", "capacity": 3,
                         "capture_count": "capture", "value": "update.%d"}
      }]})",
      "Unsupported decoder state update kind 'unknown'");

  ExpectStateGroupError(
      R"({"num_hidden_layers": 1, "state_groups": [{
        "kind": "fixed", "layer_ids": [0],
        "bindings": {"state": {"input": "past.%d", "output": "present.%d"}},
        "state_update": {"capacity": 3, "capture_count": "capture", "value": "update.%d"}
      }]})",
      "state_update is missing kind");

  ExpectStateGroupError(
      R"({"num_hidden_layers": 1, "state_groups": [{
        "kind": "fixed", "layer_ids": [0],
        "bindings": {"state": {"input": "past.%d", "output": "present.%d"}},
        "state_update": {"kind": "causal_conv", "capacity": 3, "value": "update.%d"}
      }]})",
      "state_update is missing capture_count");

  ExpectStateGroupError(
      R"({"num_hidden_layers": 1, "state_groups": [{
        "kind": "fixed", "layer_ids": [0],
        "bindings": {"state": {"input": "past.%d", "output": "present.%d"}},
        "state_update": {"kind": "causal_conv", "capacity": 3,
                         "capture_count": "capture", "decay": "update.%d"}
      }]})",
      "causal_conv state_update requires only value");

  ExpectStateGroupError(
      R"({"num_hidden_layers": 1, "state_groups": [{
        "kind": "fixed", "layer_ids": [0],
        "bindings": {"state": {"input": "past.%d", "output": "present.%d"}},
        "state_update": {"kind": "gated_delta_net", "capacity": 3,
                         "capture_count": "capture", "decay": "decay.%d", "key": "key.%d"}
      }]})",
      "requires decay, key, and delta and no value");
}

TEST(DecoderStateGroupsConfigTest, RejectsMalformedCompactStateUpdateCapacity) {
  for (const auto* capacity : {"0", "9", "1.5"}) {
    ExpectStateGroupError(
        std::string{R"({"num_hidden_layers": 1, "state_groups": [{
          "kind": "fixed", "layer_ids": [0],
          "bindings": {"state": {"input": "past.%d", "output": "present.%d"}},
          "state_update": {"kind": "causal_conv", "capacity": )"} +
            capacity + R"(, "capture_count": "capture", "value": "update.%d"}
        }]})",
        "capacity");
  }
}

TEST(DecoderStateGroupsConfigTest, RejectsCompactStateUpdateOnPagedGroup) {
  ExpectStateGroupError(
      R"({"num_hidden_layers": 1, "state_groups": [{
        "kind": "paged_kv", "layer_ids": [0],
        "bindings": {
          "key": {"input": "past.%d.key", "output": "present.%d.key"},
          "value": {"input": "past.%d.value", "output": "present.%d.value"}
        },
        "state_update": {"kind": "causal_conv", "capacity": 3,
                         "capture_count": "capture", "value": "update.%d"}
      }]})",
      "only fixed state groups support state_update");
}

TEST(DecoderStateGroupsConfigTest, RejectsMalformedOrCollidingStateUpdateNames) {
  ExpectStateGroupError(
      R"({"num_hidden_layers": 1, "state_groups": [{
        "kind": "fixed", "layer_ids": [0],
        "bindings": {"state": {"input": "past.%d", "output": "present.%d"}},
        "state_update": {"kind": "causal_conv", "capacity": 3,
                         "capture_count": "capture", "value": "update"}
      }]})",
      "expected exactly one %d");

  ExpectStateGroupError(
      R"({"num_hidden_layers": 1, "state_groups": [{
        "kind": "fixed", "layer_ids": [0],
        "bindings": {"state": {"input": "past.%d", "output": "present.%d"}},
        "state_update": {"kind": "causal_conv", "capacity": 3,
                         "capture_count": "capture", "value": "present.%d"}
      }]})",
      "resolves more than one binding to 'present.0'");
}

TEST(DecoderStateGroupsConfigTest, RequiresSharedCompactStateUpdateInputAndCapacity) {
  const auto decoder_with_second_group = [](std::string_view second_update) {
    return std::string{R"({"num_hidden_layers": 1, "state_groups": [
      {"kind": "fixed", "layer_ids": [0],
       "bindings": {"state": {"input": "past_conv.%d", "output": "present_conv.%d"}},
       "state_update": {"kind": "causal_conv", "capacity": 3,
                        "capture_count": "capture", "value": "conv_update.%d"}},
      {"kind": "fixed", "layer_ids": [0],
       "bindings": {"state": {"input": "past_gdn.%d", "output": "present_gdn.%d"}},
       "state_update": )"} +
           std::string{second_update} + "]}";
  };

  ExpectStateGroupError(
      decoder_with_second_group(
          R"({"kind": "gated_delta_net", "capacity": 2, "capture_count": "capture",
              "decay": "decay.%d", "key": "key.%d", "delta": "delta.%d"}})"),
      "same capacity and capture_count input");
  ExpectStateGroupError(
      decoder_with_second_group(
          R"({"kind": "gated_delta_net", "capacity": 3, "capture_count": "other_capture",
              "decay": "decay.%d", "key": "key.%d", "delta": "delta.%d"}})"),
      "same capacity and capture_count input");
  ExpectStateGroupError(
      decoder_with_second_group(
          R"({"enabled": false, "kind": "gated_delta_net", "capacity": 3, "capture_count": "capture",
              "decay": "decay.%d", "key": "key.%d", "delta": "delta.%d"}})"),
      "same enabled setting");
}

}  // namespace
}  // namespace Generators::test
