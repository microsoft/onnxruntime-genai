// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#define OGA_USE_SPAN 1
#include "ort_genai.h"
#include "test_utils.h"

namespace {

namespace fs_std = std::filesystem;

fs_std::path GetLfm2ModelPath() {
  return fs_std::path(MODEL_PATH) / "hf-internal-testing" / "tiny-random-lfm2-fp32";
}

void SkipIfModelUnavailable() {
  const auto model_dir = GetLfm2ModelPath();
  if (!fs_std::exists(model_dir / "genai_config.json") || !fs_std::exists(model_dir / "decoder.onnx")) {
    GTEST_SKIP() << "tiny-random-lfm2-fp32 test model not found at " << model_dir.string();
  }
}

fs_std::path MakeTempDir(const std::string& suffix) {
  static int counter = 0;
  ++counter;
  const auto dir = fs_std::temp_directory_path() /
                   ("ortgenai_lfm2_layer_types_" + suffix + "_" + std::to_string(counter));
  std::error_code ec;
  fs_std::remove_all(dir, ec);
  fs_std::create_directories(dir);
  return dir;
}

std::string ReadFile(const fs_std::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void WriteFile(const fs_std::path& path, const std::string& contents) {
  std::ofstream out(path, std::ios::binary);
  out << contents;
}

void ReplaceFirst(std::string& text, const std::string& from, const std::string& to) {
  const auto pos = text.find(from);
  if (pos == std::string::npos) {
    throw std::runtime_error("Could not find expected config fragment: " + from);
  }
  text.replace(pos, from.size(), to);
}

fs_std::path WriteModelWithLayerTypes(const std::string& suffix, const std::string& layer_types_json) {
  const auto src_dir = GetLfm2ModelPath();
  const auto dst_dir = MakeTempDir(suffix);
  fs_std::copy(src_dir, dst_dir, fs_std::copy_options::recursive | fs_std::copy_options::overwrite_existing);

  std::string config = ReadFile(dst_dir / "genai_config.json");
  ReplaceFirst(config,
               "\"layer_types\": [\"conv\", \"full_attention\", \"conv\", \"full_attention\"]",
               "\"layer_types\": " + layer_types_json);
  WriteFile(dst_dir / "genai_config.json", config);
  return dst_dir;
}

void CreateGeneratorForModel(const fs_std::path& model_dir) {
  auto model = OgaModel::Create(model_dir.string().c_str());
  auto params = OgaGeneratorParams::Create(*model);
  auto generator = OgaGenerator::Create(*model, *params);
  (void)generator;
}

std::string CaptureRuntimeErrorMessage(const fs_std::path& model_dir) {
  try {
    CreateGeneratorForModel(model_dir);
  } catch (const std::runtime_error& e) {
    return e.what();
  }

  throw std::runtime_error("Expected std::runtime_error was not thrown.");
}

void ExpectLayerTypesValidationMessage(const std::string& message, size_t actual_size, int expected_layers) {
  EXPECT_NE(message.find("LFM2Cache"), std::string::npos) << message;
  EXPECT_NE(message.find("layer_types"), std::string::npos) << message;
  EXPECT_NE(message.find("num_hidden_layers"), std::string::npos) << message;
  EXPECT_NE(message.find("layer_types array size (" + std::to_string(actual_size) + ")"), std::string::npos) << message;
  EXPECT_NE(message.find("num_hidden_layers (" + std::to_string(expected_layers) + ")"), std::string::npos) << message;
}

}  // namespace

TEST(LFM2CacheLayerTypesValidationTest, UndersizedLayerTypesArray) {
  SkipIfModelUnavailable();

  const auto model_dir = WriteModelWithLayerTypes("undersized", "[\"conv\"]");
  const std::string message = CaptureRuntimeErrorMessage(model_dir);
  ExpectLayerTypesValidationMessage(message, 1, 4);
}

TEST(LFM2CacheLayerTypesValidationTest, OversizedLayerTypesArray) {
  SkipIfModelUnavailable();

  const auto model_dir = WriteModelWithLayerTypes(
      "oversized",
      "[\"conv\", \"full_attention\", \"conv\", \"full_attention\", \"conv\"]");
  const std::string message = CaptureRuntimeErrorMessage(model_dir);
  ExpectLayerTypesValidationMessage(message, 5, 4);
}

TEST(LFM2CacheLayerTypesValidationTest, EmptyLayerTypesArray) {
  SkipIfModelUnavailable();

  const auto model_dir = WriteModelWithLayerTypes("empty", "[]");
  const std::string message = CaptureRuntimeErrorMessage(model_dir);
  ExpectLayerTypesValidationMessage(message, 0, 4);
}

TEST(LFM2CacheLayerTypesValidationTest, ValidMatchingLayers) {
  SkipIfModelUnavailable();

  const auto model_dir = WriteModelWithLayerTypes(
      "valid",
      "[\"full_attention\", \"conv\", \"full_attention\", \"conv\"]");

  EXPECT_NO_THROW({
    CreateGeneratorForModel(model_dir);
  });
}
