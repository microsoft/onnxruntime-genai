// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "ort_genai.h"
#include "test_utils.h"

namespace {

constexpr std::string_view kSharedInitializers =
    R"("shared_initializers":[{"name":"transformer.wte.weight",)"
    R"("data_file":"weights.data","offset":"0","length":"128000",)"
    R"("data_type":1,"shape":[1000,32]}])";

std::string ReadAsset(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Cannot open memory API test asset: " + path.string());
  }
  std::string result{std::istreambuf_iterator<char>{input}, {}};
  if (input.bad() || result.empty()) {
    throw std::runtime_error("Cannot read memory API test asset: " + path.string());
  }
  return result;
}

struct ScopedTokenizerAssets {
  ScopedTokenizerAssets()
      : path{std::filesystem::temp_directory_path() /
             ("ortgenai_memory_assets_" +
              std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))} {
    if (!std::filesystem::create_directory(path)) {
      throw std::runtime_error("Cannot create tokenizer test directory: " + path.string());
    }
  }

  ~ScopedTokenizerAssets() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
    EXPECT_FALSE(error) << error.message();
  }

  void Write(const char* filename, std::string_view contents) const {
    std::ofstream output;
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.open(path / filename, std::ios::binary);
    output << contents;
    output.close();
  }

  const std::filesystem::path path;
};

void ExpectError(OgaResult* raw_result, std::string_view message) {
  std::unique_ptr<OgaResult> result{raw_result};
  ASSERT_NE(result, nullptr);
  EXPECT_NE(std::string_view{result->GetError()}.find(message), std::string_view::npos)
      << result->GetError();
}

class MemoryAssetsTest : public ::testing::Test {
 protected:
  std::unique_ptr<OgaConfig> CreateConfig() const {
    auto config = OgaConfig::CreateFromBuffer(config_json_);
    config->ClearProviders();
    return config;
  }

  void AddModelAssets(OgaConfig& config) const {
    config.AddModelData("past.onnx", graph_.data(), graph_.size());
    config.AddExternalInitializerData("weights.data", weights_.data(), weights_.size());
  }

  const std::filesystem::path model_dir_{
      MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32"};
  const std::string config_json_{ReadAsset(model_dir_ / "genai_config.json")};
  const std::string graph_{ReadAsset(model_dir_ / "external-data/past.onnx")};
  const std::string weights_{ReadAsset(model_dir_ / "external-data/weights.data")};
  const std::string tokenizer_config_{ReadAsset(model_dir_ / "tokenizer_config.json")};
  const std::string tokenizer_json_{ReadAsset(model_dir_ / "tokenizer.json")};
};

TEST_F(MemoryAssetsTest, ConfigOwnsParsedValues) {
  auto json = config_json_;
  auto config = OgaConfig::CreateFromBuffer(json);
  json.assign(json.size(), ' ');
  AddModelAssets(*config);
  EXPECT_NE(OgaModel::Create(*config), nullptr);
}

TEST_F(MemoryAssetsTest, InvalidConfigArgumentsClearOutput) {
  auto config = CreateConfig();
  OgaConfig* output = config.get();
  ExpectError(OgaCreateConfigFromBuffer(nullptr, 1, &output), "config JSON");
  EXPECT_EQ(output, nullptr);

  output = config.get();
  ExpectError(OgaCreateConfigFromBuffer(config_json_.data(), 0, &output), "config JSON");
  EXPECT_EQ(output, nullptr);
  ExpectError(OgaCreateConfigFromBuffer(config_json_.data(), config_json_.size(), nullptr),
              "output pointer");

  output = config.get();
  constexpr std::string_view invalid_json{"{"};
  ExpectError(OgaCreateConfigFromBuffer(invalid_json.data(), invalid_json.size(), &output),
              "parsing");
  EXPECT_EQ(output, nullptr);
}

TEST_F(MemoryAssetsTest, RejectsMissingGraphWithoutDiskFallback) {
  auto config = CreateConfig();
  config->AddExternalInitializerData("weights.data", weights_.data(), weights_.size());
  OgaModel* output = nullptr;
  ExpectError(OgaCreateModelFromConfig(config.get(), &output), "missing graph");
  EXPECT_EQ(output, nullptr);
}

TEST_F(MemoryAssetsTest, RejectsMissingExternalInitializerBuffers) {
  auto config = CreateConfig();
  config->AddModelData("past.onnx", graph_.data(), graph_.size());
  OgaModel* output = nullptr;
  ExpectError(OgaCreateModelFromConfig(config.get(), &output),
              "missing external initializer buffers");
  EXPECT_EQ(output, nullptr);
}

TEST_F(MemoryAssetsTest, RejectsSharedInitializersFromConfig) {
  auto json = config_json_;
  const auto decoder = json.find("\"decoder\"");
  ASSERT_NE(decoder, std::string::npos);
  const auto object = json.find('{', decoder);
  ASSERT_NE(object, std::string::npos);
  json.insert(object + 1, std::string(kSharedInitializers) + ',');
  auto config = OgaConfig::CreateFromBuffer(json);
  config->ClearProviders();
  AddModelAssets(*config);

  OgaModel* output = nullptr;
  ExpectError(OgaCreateModelFromConfig(config.get(), &output),
              "Memory-only loading does not support shared initializer descriptors");
  EXPECT_EQ(output, nullptr);
}

TEST_F(MemoryAssetsTest, RejectsSharedInitializersFromOverlay) {
  auto config = CreateConfig();
  const auto overlay = R"({"model":{"decoder":{)" + std::string(kSharedInitializers) + "}}}";
  config->Overlay(overlay.c_str());
  AddModelAssets(*config);

  OgaModel* output = nullptr;
  ExpectError(OgaCreateModelFromConfig(config.get(), &output),
              "Memory-only loading does not support shared initializer descriptors");
  EXPECT_EQ(output, nullptr);
}

TEST_F(MemoryAssetsTest, PathBasedSharedInitializersRetainMetadataValidation) {
  auto config = OgaConfig::Create(model_dir_.string().c_str());
  config->ClearProviders();
  auto initializers = std::string(kSharedInitializers);
  constexpr std::string_view data_type = R"("data_type":1)";
  const auto position = initializers.find(data_type);
  ASSERT_NE(position, std::string::npos);
  initializers.replace(position, data_type.size(), R"("data_type":0)");
  const auto overlay = R"({"model":{"decoder":{)" + initializers + "}}}";
  config->Overlay(overlay.c_str());

  OgaModel* output = nullptr;
  ExpectError(OgaCreateModelFromConfig(config.get(), &output),
              "Invalid shared initializer metadata");
  EXPECT_EQ(output, nullptr);
}

TEST_F(MemoryAssetsTest, RejectsTruncatedExternalInitializerBuffer) {
  auto config = CreateConfig();
  config->AddModelData("past.onnx", graph_.data(), graph_.size());
  config->AddExternalInitializerData("weights.data", weights_.data(), weights_.size() - 1);
  EXPECT_THROW(OgaModel::Create(*config), std::runtime_error);
}

TEST_F(MemoryAssetsTest, RejectsDuplicateAndNonFlatInitializerNames) {
  auto config = CreateConfig();
  config->AddExternalInitializerData("weights.data", weights_.data(), weights_.size());
  EXPECT_THROW(config->AddExternalInitializerData("weights.data", weights_.data(), weights_.size()),
               std::runtime_error);
  for (const std::string name : {"", ".", "..", "../weights", "/weights",
                                 "subdir/weights", "subdir\\weights", "C:weights"}) {
    SCOPED_TRACE(name);
    ExpectError(OgaConfigAddExternalInitializerData(config.get(), name.c_str(),
                                                    weights_.data(), weights_.size()),
                "flat relative asset name");
  }
}

TEST_F(MemoryAssetsTest, RejectsInvalidInitializerArguments) {
  auto config = CreateConfig();
  ExpectError(OgaConfigAddExternalInitializerData(nullptr, "weights.data",
                                                  weights_.data(), weights_.size()),
              "memory-only config");
  ExpectError(OgaConfigAddExternalInitializerData(config.get(), nullptr,
                                                  weights_.data(), weights_.size()),
              "initializer bytes");
  ExpectError(OgaConfigAddExternalInitializerData(config.get(), "weights.data", nullptr, 1),
              "initializer bytes");
  ExpectError(OgaConfigAddExternalInitializerData(config.get(), "weights.data",
                                                  weights_.data(), 0),
              "initializer bytes");
}

TEST_F(MemoryAssetsTest, RejectsMemoryAssetsOnPathBasedConfig) {
  auto config = OgaConfig::Create(model_dir_.string().c_str());
  EXPECT_THROW(config->AddExternalInitializerData("weights.data", weights_.data(), weights_.size()),
               std::runtime_error);
  EXPECT_THROW(config->SetTokenizerData(tokenizer_config_, tokenizer_json_), std::runtime_error);
}

TEST_F(MemoryAssetsTest, RejectsMissingAndInvalidTokenizerData) {
  auto config = CreateConfig();
  EXPECT_THROW(OgaTokenizer::Create(*config), std::runtime_error);
  ExpectError(OgaConfigSetTokenizerData(config.get(), nullptr, 0,
                                        tokenizer_json_.data(), tokenizer_json_.size()),
              "tokenizer JSON");
  ExpectError(OgaConfigSetTokenizerData(config.get(), tokenizer_config_.data(),
                                        tokenizer_config_.size(), nullptr, 0),
              "tokenizer JSON");
  config->SetTokenizerData(tokenizer_config_, "{");
  EXPECT_THROW(OgaTokenizer::Create(*config), std::runtime_error);
}

TEST_F(MemoryAssetsTest, TokenizerCopiesJsonAndOutlivesConfig) {
  auto config = CreateConfig();
  auto tokenizer_config = tokenizer_config_;
  auto tokenizer_json = tokenizer_json_;
  config->SetTokenizerData(tokenizer_config, tokenizer_json);
  tokenizer_config.assign(tokenizer_config.size(), ' ');
  tokenizer_json.assign(tokenizer_json.size(), ' ');
  auto tokenizer = OgaTokenizer::Create(*config);
  config.reset();

  auto sequence = OgaSequences::Create();
  tokenizer->Encode("hello", *sequence);
  EXPECT_GT(sequence->SequenceCount(0), 0u);
}

TEST_F(MemoryAssetsTest, ConfigOnlyAddedTokenMatchesPathBasedTokenizer) {
  constexpr const char* token = "<|config_only_added_token|>";
  constexpr int32_t token_id = 60000;
  ASSERT_EQ(tokenizer_json_.find(token), std::string::npos);
  auto tokenizer_config = tokenizer_config_;
  const auto decoder = tokenizer_config.find("\"added_tokens_decoder\"");
  ASSERT_NE(decoder, std::string::npos);
  const auto object = tokenizer_config.find('{', decoder);
  ASSERT_NE(object, std::string::npos);
  tokenizer_config.insert(
      object + 1,
      R"("60000":{"content":"<|config_only_added_token|>","lstrip":false,"normalized":false,"rstrip":false,"single_word":false,"special":true},)");

  ScopedTokenizerAssets assets;
  assets.Write("genai_config.json", config_json_);
  assets.Write("tokenizer_config.json", tokenizer_config);
  assets.Write("tokenizer.json", tokenizer_json_);
  auto path_config = OgaConfig::Create(assets.path.string().c_str());
  auto path_tokenizer = OgaTokenizer::Create(*path_config);
  auto memory_config = CreateConfig();
  memory_config->SetTokenizerData(tokenizer_config, tokenizer_json_);
  auto memory_tokenizer = OgaTokenizer::Create(*memory_config);

  ASSERT_EQ(path_tokenizer->ToTokenId(token), token_id);
  EXPECT_EQ(memory_tokenizer->ToTokenId(token), token_id);
  auto path_tokens = OgaSequences::Create();
  auto memory_tokens = OgaSequences::Create();
  path_tokenizer->Encode(token, *path_tokens);
  memory_tokenizer->Encode(token, *memory_tokens);
  ASSERT_EQ(path_tokens->SequenceCount(0), 1u);
  ASSERT_EQ(memory_tokens->SequenceCount(0), 1u);
  EXPECT_EQ(path_tokens->SequenceData(0)[0], token_id);
  EXPECT_EQ(memory_tokens->SequenceData(0)[0], token_id);
}

TEST_F(MemoryAssetsTest, UnigramClassWithBpeVocabularyLoadsFromBlob) {
  auto config = CreateConfig();
  auto tokenizer_config = tokenizer_config_;
  constexpr std::string_view original_class{"CodeGenTokenizer"};
  const auto position = tokenizer_config.find(original_class);
  ASSERT_NE(position, std::string::npos);
  tokenizer_config.replace(position, original_class.size(), "ChatGLMTokenizer");
  config->SetTokenizerData(tokenizer_config, tokenizer_json_);
  auto tokenizer = OgaTokenizer::Create(*config);
  auto sequence = OgaSequences::Create();
  tokenizer->Encode("hello", *sequence);
  EXPECT_GT(sequence->SequenceCount(0), 0u);
}

TEST_F(MemoryAssetsTest, RejectsBothCustomOpsLibrarySettings) {
  for (const auto* overlay : {
           R"({"model":{"decoder":{"session_options":{"custom_ops_library":"custom.so"}}}})",
           R"({"model":{"decoder":{"session_options":{"custom_ops_libraries":["custom.so"]}}}})"}) {
    SCOPED_TRACE(overlay);
    auto config = CreateConfig();
    config->Overlay(overlay);
    OgaModel* output = nullptr;
    ExpectError(OgaCreateModelFromConfig(config.get(), &output),
                "does not support custom-ops library paths");
    EXPECT_EQ(output, nullptr);
  }
}

TEST_F(MemoryAssetsTest, GeneratesExpectedTokensFromRegisteredBuffers) {
  auto config = CreateConfig();
  AddModelAssets(*config);
  config->SetTokenizerData(tokenizer_config_, tokenizer_json_);
  auto model = OgaModel::Create(*config);
  config.reset();
  auto tokenizer = OgaTokenizer::Create(*model);
  auto parameters = OgaGeneratorParams::Create(*model);
  parameters->SetSearchOption("max_length", 10);
  auto generator = OgaGenerator::Create(*model, *parameters);
  const std::array<int32_t, 4> input{0, 0, 0, 52};
  generator->AppendTokens(input.data(), input.size());
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }
  const std::vector<int32_t> expected{0, 0, 0, 52, 204, 204, 204, 204, 204, 204};
  ASSERT_EQ(generator->GetSequenceCount(0), expected.size());
  EXPECT_EQ(std::vector<int32_t>(generator->GetSequenceData(0),
                                 generator->GetSequenceData(0) + generator->GetSequenceCount(0)),
            expected);
  generator.reset();
  parameters.reset();
  model.reset();

  auto sequence = OgaSequences::Create();
  tokenizer->Encode("hello", *sequence);
  EXPECT_GT(sequence->SequenceCount(0), 0u);
}

}  // namespace
