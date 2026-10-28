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

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

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

struct ScopedAssetDirectory {
  ScopedAssetDirectory()
      : path{std::filesystem::temp_directory_path() /
             ("ortgenai_memory_assets_" +
              std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))} {
    if (!std::filesystem::create_directory(path)) {
      throw std::runtime_error("Cannot create model test directory: " + path.string());
    }
  }

  ~ScopedAssetDirectory() {
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

#if defined(__linux__)
class FileSourceAssetsTest : public MemoryAssetsTest {
 protected:
  void SetUp() override {
    auto config = CreateConfig();
    std::unique_ptr<OgaResult> result{
        OgaConfigSetModelFileDescriptors(config.get(), nullptr, nullptr, 0)};
    if (result &&
        std::string_view{result->GetError()} == "ORT lacks the required initializer file-source API.") {
      GTEST_SKIP() << "The selected ORT runtime does not include initializer file sources.";
    }
    if (result) {
      FAIL() << result->GetError();
    }
  }
};
#endif

#if !defined(__linux__)
TEST_F(MemoryAssetsTest, FileSourcesAreUnsupportedOutsideLinux) {
  auto config = CreateConfig();
  ExpectError(OgaConfigSetModelFileDescriptors(config.get(), nullptr, nullptr, 0),
              "External initializer file descriptors require Linux.");
}
#endif

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

  ScopedAssetDirectory assets;
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

TEST_F(MemoryAssetsTest, EncoderDecoderGeneratesFromRegisteredBuffers) {
  const std::filesystem::path source{MODEL_PATH "marian-cache/dynamic"};
  const auto json = ReadAsset(source / "genai_config.json");
  const auto encoder = ReadAsset(source / "external-data/encoder.onnx");
  const auto decoder = ReadAsset(source / "external-data/decoder.onnx");
  const auto encoder_weights = ReadAsset(source / "external-data/encoder-weights.data");
  const auto decoder_weights = ReadAsset(source / "external-data/decoder-weights.data");

  ScopedAssetDirectory assets;
  assets.Write("genai_config.json", json);
  assets.Write("encoder.onnx", encoder);
  assets.Write("decoder.onnx", decoder);
  assets.Write("encoder-weights.data", encoder_weights);
  assets.Write("decoder-weights.data", decoder_weights);

  const auto check_generation = [](OgaModel& model) {
    auto parameters = OgaGeneratorParams::Create(model);
    parameters->SetSearchOption("batch_size", 2);
    parameters->SetSearchOption("max_length", 5);
    auto generator = OgaGenerator::Create(model, *parameters);
    const std::array<int32_t, 4> input{2, 3, 4, 5};
    generator->AppendTokens(input.data(), input.size());
    for (int step = 0; step < 3; ++step) {
      ASSERT_FALSE(generator->IsDone());
      generator->GenerateNextToken();
    }
    EXPECT_TRUE(generator->IsDone());
    const std::array<std::array<int32_t, 5>, 2> expected{
        {{2, 3, 61, 62, 63}, {4, 5, 97, 98, 99}}};
    for (size_t row = 0; row < expected.size(); ++row) {
      ASSERT_EQ(generator->GetSequenceCount(row), expected[row].size());
      EXPECT_EQ(std::vector<int32_t>(generator->GetSequenceData(row),
                                     generator->GetSequenceData(row) + generator->GetSequenceCount(row)),
                std::vector<int32_t>(expected[row].begin(), expected[row].end()));
    }
  };
  {
    auto config = OgaConfig::Create(assets.path.string().c_str());
    config->ClearProviders();
    auto model = OgaModel::Create(*config);
    check_generation(*model);
  }
  for (const auto* filename :
       {"genai_config.json", "encoder.onnx", "decoder.onnx",
        "encoder-weights.data", "decoder-weights.data"}) {
    ASSERT_TRUE(std::filesystem::remove(assets.path / filename));
  }

  for (const auto* omitted : {"encoder-weights.data", "decoder-weights.data", ""}) {
    SCOPED_TRACE(omitted);
    auto config = OgaConfig::CreateFromBuffer(json);
    config->ClearProviders();
    config->AddModelData("encoder.onnx", encoder.data(), encoder.size());
    config->AddModelData("decoder.onnx", decoder.data(), decoder.size());
    if (std::string_view{omitted} != "encoder-weights.data") {
      config->AddExternalInitializerData("encoder-weights.data", encoder_weights.data(),
                                         encoder_weights.size());
    }
    if (std::string_view{omitted} != "decoder-weights.data") {
      config->AddExternalInitializerData("decoder-weights.data", decoder_weights.data(),
                                         decoder_weights.size());
    }
    if (*omitted) {
      EXPECT_THROW(OgaModel::Create(*config), std::runtime_error);
      continue;
    }
    auto model = OgaModel::Create(*config);
    config.reset();
    for (int iteration = 0; iteration < 2; ++iteration) {
      SCOPED_TRACE(iteration);
      check_generation(*model);
    }
  }
}

#if defined(__linux__)
TEST_F(FileSourceAssetsTest, FileSourcesProvideBothEncoderAndDecoderGraphs) {
  const std::filesystem::path source{MODEL_PATH "marian-cache/dynamic"};
  const auto json = ReadAsset(source / "genai_config.json");
  const std::array<const char*, 4> names{
      "encoder.onnx", "decoder.onnx", "encoder-weights.data", "decoder-weights.data"};
  std::array<int, 4> descriptors{};
  ScopedAssetDirectory assets;
  for (size_t i = 0; i < names.size(); ++i) {
    assets.Write(names[i], ReadAsset(source / "external-data" / names[i]));
    descriptors[i] = open((assets.path / names[i]).c_str(), O_RDONLY | O_CLOEXEC);
    ASSERT_GE(descriptors[i], 0);
  }
  auto config = OgaConfig::CreateFromBuffer(json);
  config->ClearProviders();
  auto* result = OgaConfigSetModelFileDescriptors(
      config.get(), names.data(), descriptors.data(), names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    EXPECT_EQ(close(descriptors[i]), 0);
    ASSERT_TRUE(std::filesystem::remove(assets.path / names[i]));
  }
  OgaCheckResult(result);
  auto model = OgaModel::Create(*config);
  config.reset();
  for (int iteration = 0; iteration < 2; ++iteration) {
    auto parameters = OgaGeneratorParams::Create(*model);
    parameters->SetSearchOption("batch_size", 2);
    parameters->SetSearchOption("max_length", 5);
    auto generator = OgaGenerator::Create(*model, *parameters);
    const std::array<int32_t, 4> input{2, 3, 4, 5};
    generator->AppendTokens(input.data(), input.size());
    for (int step = 0; step < 3; ++step) {
      generator->GenerateNextToken();
      EXPECT_EQ(generator->GetSequenceData(0)[2 + step], 61 + step);
      EXPECT_EQ(generator->GetSequenceData(1)[2 + step], 97 + step);
    }
    EXPECT_TRUE(generator->IsDone());
  }
}

TEST_F(FileSourceAssetsTest, FileSourceFailuresDoNotReplaceValidRegistrations) {
  ScopedAssetDirectory assets;
  assets.Write("weights.data", weights_);
  const int readonly = open((assets.path / "weights.data").c_str(), O_RDONLY | O_CLOEXEC);
  const int writable = open((assets.path / "weights.data").c_str(), O_RDWR | O_CLOEXEC);
  ASSERT_GE(readonly, 0);
  ASSERT_GE(writable, 0);
  auto config = CreateConfig();
  config->AddModelData("past.onnx", graph_.data(), graph_.size());
  const char* name = "weights.data";
  auto* result = OgaConfigSetModelFileDescriptors(config.get(), &name, &readonly, 1);
  ASSERT_EQ(close(readonly), 0);
  OgaCheckResult(result);
  ExpectError(OgaConfigSetModelFileDescriptors(config.get(), &name, &writable, 1),
              "read-only regular");
  ASSERT_EQ(close(writable), 0);
  const int invalid = -1;
  ExpectError(OgaConfigSetModelFileDescriptors(config.get(), &name, &invalid, 1),
              "duplicate");
  for (const char* bad_name : {"../weights.data", "/weights.data", ".", "sub/weights.data"}) {
    ExpectError(OgaConfigSetModelFileDescriptors(config.get(), &bad_name, &invalid, 1),
                "flat relative");
  }
  ASSERT_TRUE(std::filesystem::remove(assets.path / "weights.data"));
  EXPECT_NE(OgaModel::Create(*config), nullptr);
}

TEST_F(FileSourceAssetsTest, FileSourcesRejectMetadataThatCouldBeInterpretedAsAPath) {
  ScopedAssetDirectory assets;
  assets.Write("tokenizer_module.json", "tokenizer.json");
  const int descriptor =
      open((assets.path / "tokenizer_module.json").c_str(), O_RDONLY | O_CLOEXEC);
  ASSERT_GE(descriptor, 0);
  auto config = CreateConfig();
  config->SetTokenizerData(tokenizer_config_, tokenizer_json_);
  const char* name = "tokenizer_module.json";
  auto* result = OgaConfigSetModelFileDescriptors(config.get(), &name, &descriptor, 1);
  ASSERT_EQ(close(descriptor), 0);
  OgaCheckResult(result);
  ASSERT_TRUE(std::filesystem::remove(assets.path / name));
  OgaTokenizer* tokenizer = nullptr;
  ExpectError(OgaCreateTokenizerFromConfig(config.get(), &tokenizer),
              "must contain a JSON object");
  EXPECT_EQ(tokenizer, nullptr);
}

TEST_F(FileSourceAssetsTest, FileSourcesRejectMissingAndTruncatedWeights) {
  auto config = CreateConfig();
  config->AddModelData("past.onnx", graph_.data(), graph_.size());
  config->SetModelFileDescriptors(nullptr, nullptr, 0);
  EXPECT_THROW(OgaModel::Create(*config), std::runtime_error);
  ScopedAssetDirectory assets;
  assets.Write("weights.data", std::string_view{weights_}.substr(0, weights_.size() - 1));
  const int descriptor = open((assets.path / "weights.data").c_str(), O_RDONLY | O_CLOEXEC);
  ASSERT_GE(descriptor, 0);
  const char* name = "weights.data";
  auto* result = OgaConfigSetModelFileDescriptors(config.get(), &name, &descriptor, 1);
  ASSERT_EQ(close(descriptor), 0);
  OgaCheckResult(result);
  ASSERT_TRUE(std::filesystem::remove(assets.path / "weights.data"));
  EXPECT_THROW(OgaModel::Create(*config), std::runtime_error);
}

TEST_F(FileSourceAssetsTest, FileSourcesSurviveUnlinkAndConfigDestruction) {
  ScopedAssetDirectory assets;
  assets.Write("weights.data", weights_);
  const int descriptor = open((assets.path / "weights.data").c_str(), O_RDONLY | O_CLOEXEC);
  ASSERT_GE(descriptor, 0);
  auto config = CreateConfig();
  config->Overlay(R"({"model":{"decoder":{"session_options":{"graph_optimization_level":"ORT_DISABLE_ALL"}}}})");
  auto registered_graph = graph_;
  config->AddModelData("past.onnx", registered_graph.data(), registered_graph.size());
  const char* name = "weights.data";
  OgaResult* result = OgaConfigSetModelFileDescriptors(
      config.get(), &name, &descriptor, 1);
  ASSERT_EQ(close(descriptor), 0);
  OgaCheckResult(result);
  ASSERT_TRUE(std::filesystem::remove(assets.path / "weights.data"));
  auto model = OgaModel::Create(*config);
  config.reset();
  registered_graph.assign(registered_graph.size(), '\0');
  for (int iteration = 0; iteration < 2; ++iteration) {
    auto parameters = OgaGeneratorParams::Create(*model);
    parameters->SetSearchOption("max_length", 6);
    auto generator = OgaGenerator::Create(*model, *parameters);
    const std::array<int32_t, 4> input{0, 0, 0, 52};
    generator->AppendTokens(input.data(), input.size());
    while (!generator->IsDone()) {
      generator->GenerateNextToken();
    }
    ASSERT_EQ(generator->GetSequenceCount(0), 6u);
    EXPECT_EQ(generator->GetSequenceData(0)[4], 204);
    EXPECT_EQ(generator->GetSequenceData(0)[5], 204);
  }
}
#endif

enum class LoadingMode { kPath,
                         kLegacyModelData,
                         kMemoryAssets,
                         kFileSources };

void CheckPipelineLoadingModes(std::initializer_list<LoadingMode> modes) {
  const std::filesystem::path source{MODEL_PATH "engine/dummy-decoder"};
  const auto json = ReadAsset(source / "genai_config.json");
  const auto graph = ReadAsset(source / "external-data/decoder.onnx");
  const auto weights = ReadAsset(source / "external-data/weights.data");

  for (const auto mode : modes) {
    const bool file_sources = mode == LoadingMode::kFileSources;
    const bool memory = mode == LoadingMode::kMemoryAssets || file_sources;
    const bool legacy = mode == LoadingMode::kLegacyModelData;
    SCOPED_TRACE(file_sources ? "file-sources" : memory ? "memory-backed"
                                             : legacy   ? "legacy-model-data"
                                                        : "path-based");
    ScopedAssetDirectory assets;
    const auto filename = assets.path.filename().string() + ".onnx";
    ASSERT_FALSE(std::filesystem::exists(filename));
    assets.Write("genai_config.json", json);
    assets.Write(filename.c_str(), graph);
    assets.Write("weights.data", weights);
    auto config = memory ? OgaConfig::CreateFromBuffer(json)
                         : OgaConfig::Create(assets.path.string().c_str());
    config->ClearProviders();
    const auto overlay =
        R"({"model":{"type":"decoder-pipeline","decoder":{"pipeline":[{"stage":{"filename":")" +
        filename +
        R"(","reset_session_idx":0,)"
        R"("inputs":["input_ids","position_ids","attention_mask","past_key_values.0.key","past_key_values.0.value"],)"
        R"("outputs":["logits","present.0.key","present.0.value"]}}]}}})";
    config->Overlay(overlay.c_str());
    auto registered_graph = graph;
    if ((memory && !file_sources) || legacy) {
      config->AddModelData(filename, registered_graph.data(), registered_graph.size());
    }
    if (memory && !file_sources) {
      config->AddExternalInitializerData("weights.data", weights.data(), weights.size());
    }
#if defined(__linux__)
    if (file_sources) {
      const std::array<const char*, 2> names{filename.c_str(), "weights.data"};
      const std::array<int, 2> descriptors{
          open((assets.path / filename).c_str(), O_RDONLY | O_CLOEXEC),
          open((assets.path / "weights.data").c_str(), O_RDONLY | O_CLOEXEC)};
      auto* result = OgaConfigSetModelFileDescriptors(
          config.get(), names.data(), descriptors.data(), names.size());
      for (int descriptor : descriptors) {
        EXPECT_EQ(close(descriptor), 0);
      }
      OgaCheckResult(result);
    }
#endif
    auto model = OgaModel::Create(*config);
    config.reset();
    if (memory) {
      ASSERT_TRUE(std::filesystem::remove(assets.path / filename));
      ASSERT_TRUE(std::filesystem::remove(assets.path / "weights.data"));
    } else if (legacy) {
      registered_graph.assign(registered_graph.size(), '\0');
    }

    for (int iteration = 0; iteration < 2; ++iteration) {
      SCOPED_TRACE(iteration);
      auto parameters = OgaGeneratorParams::Create(*model);
      parameters->SetSearchOption("batch_size", 2);
      parameters->SetSearchOption("max_length", 3);
      auto generator = OgaGenerator::Create(*model, *parameters);
      const std::array<int32_t, 4> input{0, 2, 0, 3};
      generator->AppendTokens(input.data(), input.size());
      generator->GenerateNextToken();
      for (size_t batch = 0; batch < 2; ++batch) {
        ASSERT_EQ(generator->GetSequenceCount(batch), 3u);
        EXPECT_EQ(generator->GetSequenceData(batch)[2], 0);
      }
    }
  }
}

TEST_F(MemoryAssetsTest, PipelineSessionResetPreservesLoadingMode) {
  CheckPipelineLoadingModes(
      {LoadingMode::kPath, LoadingMode::kLegacyModelData, LoadingMode::kMemoryAssets});
}

#if defined(__linux__)
TEST_F(FileSourceAssetsTest, PipelineSessionResetPreservesFileSources) {
  CheckPipelineLoadingModes({LoadingMode::kFileSources});
}
#endif

}  // namespace
