// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "ort_genai.h"

namespace Generators::test {
namespace {

namespace fs_std = std::filesystem;

fs_std::path WriteConfig(const std::string& name, std::string_view model_body) {
  auto root = fs_std::temp_directory_path() / ("ortgenai_mtp_config_" + name);
  std::error_code ec;
  fs_std::remove_all(root, ec);
  fs_std::create_directories(root);

  std::ofstream out(root / "genai_config.json", std::ios::binary);
  out << R"({ "model": { "type": "tiny-test-model", "vocab_size": 16, "context_length": 32,)"
      << model_body << R"( }, "search": {} })";
  return root;
}

fs_std::path WriteMtpConfig(const std::string& output_name) {
  return WriteConfig(output_name,
                     R"( "decoder": { "filename": "model.onnx" },)"
                     R"( "mtp": { "filename": "mtp.onnx",)"
                     R"( "main_hidden_states": "main_hidden",)"
                     R"( "outputs": { ")" +
                         output_name + R"(": "head_feedback" } })");
}

// The Gemma4 assistant contract as emitted by the export tooling.
constexpr std::string_view kAssistantMtpBody =
    R"( "decoder": { "filename": "model.onnx", "hidden_size": 8,)"
    R"( "max_logits_sequence_length": 6,)"
    R"( "outputs": { "hidden_states": "final_hidden_state" } },)"
    R"( "mtp": { "filename": "assistant.onnx",)"
    R"( "main_hidden_states": "final_hidden_state",)"
    R"( "main_inputs_embeds": "inputs_embeds",)"
    R"( "shared_kv_layers": [22, 23],)"
    R"( "inputs": { "hidden_states": "inputs_embeds", "attention_mask": "attention_mask",)"
    R"( "shared_key_names": ["shared_kv.sliding_attention.key", "shared_kv.full_attention.key"],)"
    R"( "shared_value_names": ["shared_kv.sliding_attention.value", "shared_kv.full_attention.value"] },)"
    R"( "outputs": { "logits": "logits", "hidden_states": "projected_state" } })";

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

TEST(MtpConfigTest, AcceptsAssistantHeadContract) {
  const auto root = WriteConfig("assistant", kAssistantMtpBody);
  EXPECT_NO_THROW(OgaConfig::Create(root.string().c_str()));
}

TEST(MtpConfigTest, RejectsNegativeMaxLogitsSequenceLength) {
  const auto root = WriteConfig("negative_max_logits",
                                " \"decoder\": { \"filename\": \"model.onnx\","
                                " \"max_logits_sequence_length\": -1 }");
  EXPECT_THROW(OgaConfig::Create(root.string().c_str()), std::exception);
}

TEST(MtpConfigTest, RejectsUnknownSharedKvKey) {
  const auto root = WriteConfig("unknown_shared_kv",
                                " \"decoder\": { \"filename\": \"model.onnx\" },"
                                " \"mtp\": { \"shared_kv_layer\": [22] }");
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
