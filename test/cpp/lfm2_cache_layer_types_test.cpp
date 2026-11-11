// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

// LFM2 variants that the exporter labels with their own model type still have to run through the
// LFM2 runtime: the same conv state cache, and no rewind.
void ExpectLfm2Runtime(const std::string& model_type, const std::string& temp_suffix,
                       const std::string& layer_types_json = "") {
  const auto src_dir = GetLfm2ModelPath();
  const auto model_dir = MakeTempDir(temp_suffix);
  fs_std::copy(src_dir, model_dir, fs_std::copy_options::recursive | fs_std::copy_options::overwrite_existing);
  std::string config = ReadFile(model_dir / "genai_config.json");
  ReplaceFirst(config, "\"type\": \"lfm2\"", "\"type\": \"" + model_type + "\"");
  if (!layer_types_json.empty()) {
    ReplaceFirst(config,
                 "\"layer_types\": [\"conv\", \"full_attention\", \"conv\", \"full_attention\"]",
                 "\"layer_types\": " + layer_types_json);
  }
  WriteFile(model_dir / "genai_config.json", config);

  auto model = OgaModel::Create(model_dir.string().c_str());
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 8);
  auto generator = OgaGenerator::Create(*model, *params);

  const std::vector<int32_t> input_ids{0, 0, 195, 731};
  generator->AppendTokens(input_ids.data(), input_ids.size());
  generator->GenerateNextToken();
  const size_t sequence_length = generator->GetSequenceCount(0);
  EXPECT_GT(sequence_length, input_ids.size());

  // Conv state cannot be rewound, exactly like the dense LFM2 models. The generator's up-front guard
  // has to reject it; a throw from deeper in the state would come after the sequence was rewound.
  std::string message;
  try {
    generator->RewindTo(0);
  } catch (const std::runtime_error& e) {
    message = e.what();
  }
  EXPECT_NE(message.find("RewindTo is currently not supported for " + model_type), std::string::npos) << message;
  EXPECT_EQ(generator->GetSequenceCount(0), sequence_length);
}

// An LFM3 model whose layer 3 is a sliding_attention layer. A window_size above 0 adds the
// decoder.sliding_window section that the builder writes with windowed_kv_cache, the default.
fs_std::path WriteLfm3Model(const std::string& suffix, int window_size, bool share_buffer) {
  const auto model_dir = WriteModelWithLayerTypes(suffix, "[\"conv\", \"full_attention\", \"conv\", \"sliding_attention\"]");
  std::string config = ReadFile(model_dir / "genai_config.json");
  ReplaceFirst(config, "\"type\": \"lfm2\"", "\"type\": \"lfm3\"");
  if (window_size > 0) {
    ReplaceFirst(config, "\"conv_cache_size\": 4,",
                 "\"conv_cache_size\": 4, \"sliding_window\": {\"window_size\": " + std::to_string(window_size) +
                     ", \"slide_key_value_cache\": false, \"slide_inputs\": false, \"layers\": [3], \"cache_slack\": 0},");
  }
  if (share_buffer) {
    ReplaceFirst(config, "\"past_present_share_buffer\": false", "\"past_present_share_buffer\": true");
  }
  WriteFile(model_dir / "genai_config.json", config);
  return model_dir;
}

// Sequence lengths of the KV buffers bound before the first run: the full_attention layer's past, and
// the sliding_attention layer's past and present.
void ExpectKvLengths(const fs_std::path& model_dir, int max_length, int64_t full_past, int64_t sliding,
                     int chunk_size = 0) {
  auto model = OgaModel::Create(model_dir.string().c_str());
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  if (chunk_size > 0) {
    params->SetSearchOption("chunk_size", chunk_size);
  }
  auto generator = OgaGenerator::Create(*model, *params);
  EXPECT_EQ(generator->GetInput("past_key_values.1.key")->Shape()[2], full_past);
  EXPECT_EQ(generator->GetInput("past_key_values.3.key")->Shape()[2], sliding);
  EXPECT_EQ(generator->GetOutput("present.3.key")->Shape()[2], sliding);
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

// A layer type the cache does not know must not be taken for a conv layer.
TEST(LFM2CacheLayerTypesValidationTest, UnknownLayerType) {
  SkipIfModelUnavailable();

  const auto model_dir = WriteModelWithLayerTypes(
      "unknown",
      "[\"conv\", \"full_attention\", \"linear_attention\", \"full_attention\"]");
  const std::string message = CaptureRuntimeErrorMessage(model_dir);
  EXPECT_NE(message.find("LFM2Cache: layer_types[2] is \"linear_attention\""), std::string::npos) << message;
}

TEST(LFM2CacheLayerTypesValidationTest, ValidMatchingLayers) {
  SkipIfModelUnavailable();

  const auto model_dir = WriteModelWithLayerTypes(
      "valid",
      "[\"conv\", \"full_attention\", \"conv\", \"full_attention\"]");

  EXPECT_NO_THROW({
    CreateGeneratorForModel(model_dir);
  });
}

// The MoE variant (LFM2-8B-A1B, LFM2.5-8B-A1B, LFM2-24B-A2B) is exported as "lfm2_moe".
TEST(LFM2CacheLayerTypesValidationTest, Lfm2MoeTypeUsesLfm2Runtime) {
  SkipIfModelUnavailable();
  ExpectLfm2Runtime("lfm2_moe", "moe_type");
}

// An LFM2-VL checkpoint exported without exclude_embeds is labeled "lfm2_vl_text": a plain LFM2
// decoder rather than a stage of the vision pipeline.
TEST(LFM2CacheLayerTypesValidationTest, Lfm2VlTextTypeUsesLfm2Runtime) {
  SkipIfModelUnavailable();
  ExpectLfm2Runtime("lfm2_vl_text", "vl_text_type");
}

// An LFM2-Audio checkpoint exported without exclude_embeds is labeled "lfm2_audio_text": a plain
// LFM2 decoder rather than a stage of the speech pipeline.
TEST(LFM2CacheLayerTypesValidationTest, Lfm2AudioTextTypeUsesLfm2Runtime) {
  SkipIfModelUnavailable();
  ExpectLfm2Runtime("lfm2_audio_text", "audio_text_type");
}

// LFM3 is exported as "lfm3", and its sliding_attention layers keep a KV cache like the
// full_attention ones rather than a conv state.
TEST(LFM2CacheLayerTypesValidationTest, Lfm3TypeUsesLfm2Runtime) {
  SkipIfModelUnavailable();
  ExpectLfm2Runtime("lfm3", "lfm3_type", "[\"conv\", \"full_attention\", \"conv\", \"sliding_attention\"]");
}

// The sliding layer keeps the window plus the CPU slack of 16 instead of max_length, in one buffer bound
// as past and present: the attention op evicts from it in place.
TEST(LFM2CacheLayerTypesValidationTest, Lfm3SlidingLayerKeepsOnlyTheWindow) {
  SkipIfModelUnavailable();
  ExpectKvLengths(WriteLfm3Model("lfm3_window", 4, true), 64, 64, 4 + 16);
}

// The attention op stages a long step itself, so chunk_size adds no rows: ORT's CUDA kernel accepts only
// a buffer of exactly the window.
TEST(LFM2CacheLayerTypesValidationTest, Lfm3SlidingLayerIgnoresChunkSize) {
  SkipIfModelUnavailable();
  ExpectKvLengths(WriteLfm3Model("lfm3_window_chunked", 4, true), 64, 64, 4 + 16, 32);
}

// ORT rejects a buffer shorter than the window, so a shorter max_length does not shrink it.
TEST(LFM2CacheLayerTypesValidationTest, Lfm3SlidingLayerHoldsTheWindowPastMaxLength) {
  SkipIfModelUnavailable();
  ExpectKvLengths(WriteLfm3Model("lfm3_window_short", 16, true), 8, 8, 16);
}

// Without a shared buffer (beam search, for one) the full layer grows from an empty past, while the
// sliding layer keeps its fixed buffer from the first run.
TEST(LFM2CacheLayerTypesValidationTest, Lfm3SlidingLayerStaysFixedWithoutSharedBuffer) {
  SkipIfModelUnavailable();
  ExpectKvLengths(WriteLfm3Model("lfm3_window_unshared", 4, false), 64, 0, 4 + 16);
}

// Exports without windowed_kv_cache have no decoder.sliding_window, so the sliding layer keeps max_length.
TEST(LFM2CacheLayerTypesValidationTest, Lfm3SlidingLayerWithoutWindowKeepsMaxLength) {
  SkipIfModelUnavailable();
  ExpectKvLengths(WriteLfm3Model("lfm3_no_window", 0, true), 64, 64, 64);
}
