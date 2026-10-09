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

fs_std::path WriteMtpConfig(const std::string& output_name,
                            const std::string& enabled_field = {},
                            const std::string& suffix = {},
                            const std::string& output_fields = {},
                            const std::string& speculative_fields = {}) {
  const auto root = fs_std::temp_directory_path() /
                    ("ortgenai_mtp_config_" + output_name + suffix);
  std::error_code ec;
  fs_std::remove_all(root, ec);
  fs_std::create_directories(root);

  const std::string config =
      "{ \"model\": { \"type\": \"tiny-test-model\","
      " \"vocab_size\": 16, \"context_length\": 32,"
      " \"decoder\": { \"filename\": \"model.onnx\" },"
      " \"mtp\": { " +
      enabled_field +
      "\"filename\": \"mtp.onnx\","
      " \"main_hidden_states\": \"main_hidden\","
      " \"outputs\": { \"" +
      output_name +
      "\": \"head_feedback\" " + output_fields + " } } },"
      " \"search\": {} " + speculative_fields + " }";
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

TEST(MtpConfigTest, EnablesMtpByDefault) {
  Config config;
  config.model.mtp.filename = "mtp.onnx";
  EXPECT_TRUE(config.model.mtp.enabled);
  EXPECT_TRUE(config.model.mtp.IsEnabled());
}

TEST(MtpConfigTest, ParsesIndexShareMetadata) {
  const auto root = WriteMtpConfig(
      "hidden_states",
      R"("index_share": {"enabled": false, "base_capacity": 2051,
          "max_draft_tokens": 7, "indices_output": "indices", "counts_output": "counts"}, )",
      "_indexshare");
  EXPECT_THROW(OgaConfig::Create(root.string().c_str()), std::exception);
}

TEST(MtpConfigTest, ParsesSingleModelIndexShareMetadata) {
  const auto root = WriteMtpConfig(
      "hidden_states",
      R"("index_share": {"enabled": true, "base_capacity": 2051,
          "max_draft_tokens": 1, "indices_output": "indices", "counts_output": "counts"}, )",
      "_single_indexshare");
  EXPECT_THROW(OgaConfig::Create(root.string().c_str()), std::exception);
}

TEST(MtpConfigTest, RejectsNonpositiveIndexShareCapacity) {
  const auto root = WriteMtpConfig("hidden_states", R"("index_share": {"base_capacity": 0}, )", "_bad_indexshare");
  EXPECT_THROW(OgaConfig::Create(root.string().c_str()), std::exception);
}

TEST(MtpConfigTest, EnablesFlatIndexShareMetadataByDefault) {
  const auto root = WriteMtpConfig(
      "hidden_states", R"("base_capacity": 2051, )", "_flat_indexshare",
      R"(, "present_indices": "indexshare.present_indices", "present_counts": "indexshare.present_counts")");
  const Config config{fs::path{root.string()}, {}};
  EXPECT_TRUE(config.model.mtp.index_share.enabled);
  EXPECT_EQ(config.model.mtp.index_share.base_capacity, 2051);
  EXPECT_EQ(config.model.mtp.index_share.max_draft_tokens, 7);
  EXPECT_EQ(config.model.mtp.index_share.indices_output, "indexshare.present_indices");
  EXPECT_EQ(config.model.mtp.index_share.counts_output, "indexshare.present_counts");
}

TEST(MtpConfigTest, ParsesNamedIndexShareInputsAndOutputsWithOneDraftLimit) {
  const auto root = WriteMtpConfig(
      "hidden_states",
      R"("base_capacity": 2051, "inputs": {"past_indices": "cached.indices", "past_counts": "cached.counts"}, )",
      "_named_indexshare",
      R"(, "present_indices": "selected.indices", "present_counts": "selected.counts")",
      R"(, "speculative": {"max_draft_tokens": 3})");
  const Config config{fs::path{root.string()}, {}};
  EXPECT_TRUE(config.model.mtp.index_share.enabled);
  EXPECT_EQ(config.model.mtp.index_share.indices_input, "cached.indices");
  EXPECT_EQ(config.model.mtp.index_share.counts_input, "cached.counts");
  EXPECT_EQ(config.model.mtp.index_share.indices_output, "selected.indices");
  EXPECT_EQ(config.model.mtp.index_share.counts_output, "selected.counts");
  EXPECT_EQ(config.speculative.max_draft_tokens, 3);
  EXPECT_EQ(config.model.mtp.index_share.max_draft_tokens, 7);
}

TEST(MtpConfigTest, ParsesFlatIndexShareDraftWidth) {
  const auto root = WriteMtpConfig(
      "hidden_states", R"("max_draft_tokens": 1, "base_capacity": 2051, )", "_flat_single_indexshare",
      R"(, "indices": "indices", "counts": "counts")");
  EXPECT_THROW(OgaConfig::Create(root.string().c_str()), std::exception);
}

TEST(MtpConfigTest, RejectsLegacyIndexShareOutputAliases) {
  const auto root = WriteMtpConfig(
      "hidden_states", R"("base_capacity": 2051, )", "_legacy_outputs",
      R"(, "indices": "indexshare.present_indices", "counts": "indexshare.present_counts")");
  EXPECT_THROW(OgaConfig::Create(root.string().c_str()), std::exception);
}

TEST(MtpConfigTest, RejectsNonpositiveFlatIndexShareCapacity) {
  const auto root = WriteMtpConfig("hidden_states", R"("base_capacity": 0, )", "_bad_flat_indexshare");
  EXPECT_THROW(OgaConfig::Create(root.string().c_str()), std::exception);
}

TEST(MtpConfigTest, RejectsInvalidFlatIndexShareDraftWidth) {
  const auto root = WriteMtpConfig("hidden_states", R"("base_capacity": 2051, "max_draft_tokens": 8, )", "_bad_flat_width");
  EXPECT_THROW(OgaConfig::Create(root.string().c_str()), std::exception);
}

TEST(MtpConfigTest, AcceptsDisabledMtpToggleThroughPublicApi) {
  const auto root =
      WriteMtpConfig("hidden_states", "\"enabled\": false, ", "_disabled");
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

}  // namespace Generators::test
