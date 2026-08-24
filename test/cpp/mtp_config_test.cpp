// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "ort_genai.h"

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

}  // namespace Generators::test
