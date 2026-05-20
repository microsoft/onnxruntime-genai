// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <algorithm>
#include <fstream>
#include <string>
#include "../src/generators.h"
#include "../src/config.h"
#include "../src/models/model_package.h"

// --- JSON Merge Patch (RFC 7386) tests ---

TEST(JsonMergePatch, SimpleOverwrite) {
  std::string base = R"({"a":1,"b":"hello"})";
  std::string patch = R"({"b":"world"})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("\"a\":1"), std::string::npos);
  EXPECT_NE(result.find("\"b\":\"world\""), std::string::npos);
  EXPECT_EQ(result.find("\"hello\""), std::string::npos);
}

TEST(JsonMergePatch, AddNewKey) {
  std::string base = R"({"a":1})";
  std::string patch = R"({"b":2})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("\"a\":1"), std::string::npos);
  EXPECT_NE(result.find("\"b\":2"), std::string::npos);
}

TEST(JsonMergePatch, DeleteKey) {
  std::string base = R"({"a":1,"b":2})";
  std::string patch = R"({"b":null})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("\"a\":1"), std::string::npos);
  EXPECT_EQ(result.find("\"b\""), std::string::npos);
}

TEST(JsonMergePatch, NestedMerge) {
  std::string base = R"({"model":{"decoder":{"hidden_size":768},"type":"decoder_only"}})";
  std::string patch = R"({"model":{"decoder":{"hidden_size":1024,"num_heads":16}}})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("1024"), std::string::npos);
  EXPECT_NE(result.find("16"), std::string::npos);
  EXPECT_NE(result.find("decoder_only"), std::string::npos);
  EXPECT_EQ(result.find("768"), std::string::npos);
}

TEST(JsonMergePatch, ArrayReplacement) {
  std::string base = R"({"arr":[1,2,3]})";
  std::string patch = R"({"arr":[4,5]})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("[4,5]"), std::string::npos);
  EXPECT_EQ(result.find("[1,2,3]"), std::string::npos);
}

TEST(JsonMergePatch, EmptyPatch) {
  std::string base = R"({"a":1})";
  std::string result = Generators::JsonMergePatch(base, "");
  EXPECT_EQ(result, base);
}

TEST(JsonMergePatch, EmptyBase) {
  std::string patch = R"({"a":1})";
  std::string result = Generators::JsonMergePatch("", patch);
  EXPECT_EQ(result, patch);
}

TEST(JsonMergePatch, PatchReplacesNonObject) {
  std::string base = R"("hello")";
  std::string patch = R"({"a":1})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("\"a\":1"), std::string::npos);
}

TEST(JsonMergePatch, BoolAndNullValues) {
  std::string base = R"({"a":true,"b":false,"c":null})";
  std::string patch = R"({"a":false,"c":"restored"})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("\"a\":false"), std::string::npos);
  EXPECT_NE(result.find("\"c\":\"restored\""), std::string::npos);
}

// --- Package detection tests ---

TEST(IsModelPackage, NonExistentPath) {
  EXPECT_FALSE(Generators::IsModelPackage(fs::path("/tmp/nonexistent_model_package_test_path_12345")));
}

TEST(IsModelPackage, FlatDirectory) {
  // A temp dir without manifest.json should not be detected as a package
  std::string tmp_dir = "/tmp/genai_test_flat_dir";
  std::system(("mkdir -p " + tmp_dir).c_str());
  EXPECT_FALSE(Generators::IsModelPackage(fs::path(tmp_dir)));
  std::system(("rm -rf " + tmp_dir).c_str());
}

TEST(IsModelPackage, PackageDirectory) {
  // A temp dir with manifest.json should be detected as a package
  std::string tmp_dir = "/tmp/genai_test_package_dir";
  std::system(("mkdir -p " + tmp_dir).c_str());
  std::system(("touch " + tmp_dir + "/manifest.json").c_str());
  EXPECT_TRUE(Generators::IsModelPackage(fs::path(tmp_dir)));
  std::system(("rm -rf " + tmp_dir).c_str());
}

// --- InjectPackageEp tests ---

#if ORT_HAS_MODEL_PACKAGE
TEST(InjectPackageEp, InjectsEpAtPosition0) {
  Generators::Config::SessionOptions so;
  Generators::InjectPackageEp(so, "CUDAExecutionProvider");

  ASSERT_EQ(so.providers.size(), 1u);
  EXPECT_EQ(so.providers[0], "cuda");
  ASSERT_EQ(so.provider_options.size(), 1u);
  EXPECT_EQ(so.provider_options[0].name, "cuda");
}

TEST(InjectPackageEp, MaintainsPosition0WhenAlreadyPresent) {
  Generators::Config::SessionOptions so;
  so.providers.push_back("cuda");
  so.provider_options.push_back({"cuda", {}, {}});

  // Should be idempotent
  Generators::InjectPackageEp(so, "CUDAExecutionProvider");

  ASSERT_EQ(so.providers.size(), 1u);
  EXPECT_EQ(so.providers[0], "cuda");
}

TEST(InjectPackageEp, RotatesExistingProviderToFront) {
  Generators::Config::SessionOptions so;
  so.providers.push_back("webgpu");
  so.providers.push_back("cuda");
  so.provider_options.push_back({"webgpu", {}, {}});
  so.provider_options.push_back({"cuda", {}, {}});

  Generators::InjectPackageEp(so, "CUDAExecutionProvider");

  ASSERT_EQ(so.providers.size(), 2u);
  EXPECT_EQ(so.providers[0], "cuda");
  EXPECT_EQ(so.providers[1], "webgpu");
}

TEST(InjectPackageEp, SkipsCpuExecutionProvider) {
  Generators::Config::SessionOptions so;
  Generators::InjectPackageEp(so, "CPUExecutionProvider");

  // CPU is a no-op (CPU is always available as implicit fallback)
  EXPECT_TRUE(so.providers.empty());
}

// --- Consumer metadata validation tests ---

// We can't easily test GetGenAIConfigOverlay without a real package context,
// but we can test the JSON parsing utilities it relies on.

// --- Config::FromPackage tests ---
#endif

// The following tests need ORT_HAS_MODEL_PACKAGE
#if ORT_HAS_MODEL_PACKAGE
TEST(ConfigFromPackage, ParsesMergedJsonAndSetsConfigPath) {
  // Test that Config::FromPackage correctly parses a merged JSON string
  // and sets the config_path to the provided path
  std::string merged_json = R"({
    "model": {
      "type": "phi3",
      "context_length": 4096,
      "bos_token_id": 1,
      "eos_token_id": [2],
      "pad_token_id": 0,
      "vocab_size": 32000,
      "decoder": {
        "component": "decoder",
        "head_size": 128,
        "hidden_size": 3072,
        "num_attention_heads": 24,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8
      }
    },
    "search": {
      "max_length": 4096,
      "past_present_share_buffer": true
    }
  })";

  auto config = Generators::Config::FromPackage(
      fs::path("/tmp/fake_package/configs"), merged_json, nullptr);

  EXPECT_EQ(config->model.type, "phi3");
  EXPECT_EQ(config->model.context_length, 4096);
  EXPECT_EQ(config->model.vocab_size, 32000);
  EXPECT_EQ(config->model.decoder.component, "decoder");
  EXPECT_EQ(config->model.decoder.hidden_size, 3072);
  EXPECT_EQ(config->model.decoder.num_attention_heads, 24);
  EXPECT_EQ(config->model.decoder.num_hidden_layers, 32);
  EXPECT_EQ(config->search.max_length, 4096);
  EXPECT_EQ(config->config_path.string(), "/tmp/fake_package/configs");
  EXPECT_FALSE(config->IsPackage());  // no package_state passed
}

TEST(ConfigFromPackage, OverlayAppliedBeforeParsing) {
  // Test that JSON merge patch correctly modifies the config
  std::string base_json = R"({
    "model": {
      "type": "phi3",
      "context_length": 131072,
      "bos_token_id": 1,
      "eos_token_id": [2],
      "pad_token_id": 0,
      "vocab_size": 32000,
      "decoder": {
        "component": "decoder",
        "head_size": 128,
        "hidden_size": 3072,
        "num_attention_heads": 24,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8
      }
    },
    "search": { "max_length": 131072 }
  })";

  // Overlay changes context_length and adds an input name
  std::string overlay = R"({
    "model": {
      "context_length": 4096,
      "decoder": {
        "inputs": { "position_ids": "position_ids" }
      }
    },
    "search": { "max_length": 4096 }
  })";

  std::string merged = Generators::JsonMergePatch(base_json, overlay);
  auto config = Generators::Config::FromPackage(
      fs::path("/tmp/fake_package/configs"), merged, nullptr);

  EXPECT_EQ(config->model.context_length, 4096);
  EXPECT_EQ(config->search.max_length, 4096);
  // Original values should be preserved
  EXPECT_EQ(config->model.decoder.hidden_size, 3072);
  EXPECT_EQ(config->model.decoder.num_hidden_layers, 32);
  // Overlay should add position_ids input
  EXPECT_EQ(config->model.decoder.inputs.position_ids, "position_ids");
}

// --- End-to-end model package tests ---
// These tests load a real phi4 model from a package directory.
// They require the test_models directory to contain the package fixtures.

#include "ort_genai.h"

static const char* kCpuOnlyPkgPath = MODEL_PATH "phi4-cpu-only.ortpackage";
static const char* kMultiVariantPkgPath =
    "/datadisks/jambaykinley/archive/p/packages/phi-4-mini-reasoning.v4.ortpackage";

static bool PackageExists(const char* path) {
  return fs::exists(fs::path(path) / "manifest.json");
}

TEST(ModelPackageE2E, SingleVariantCpuAutoDetect) {
  if (!PackageExists(kCpuOnlyPkgPath)) {
    GTEST_SKIP() << "CPU-only test package not found at " << kCpuOnlyPkgPath;
  }

  // Auto-detect should pick CPUExecutionProvider (only EP available)
  auto model = OgaModel::Create(kCpuOnlyPkgPath);
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(std::string(model->GetType().p_), "phi3");
  // Device type may be uppercase or lowercase depending on the EP
  std::string device_type = model->GetDeviceType().p_;
  std::transform(device_type.begin(), device_type.end(), device_type.begin(), ::tolower);
  EXPECT_EQ(device_type, "cpu");
}

TEST(ModelPackageE2E, MultiVariantExplicitCpu) {
  if (!PackageExists(kMultiVariantPkgPath)) {
    GTEST_SKIP() << "Multi-variant test package not found at " << kMultiVariantPkgPath;
  }

  // Explicitly select CPU EP from a multi-variant package
  auto model = OgaModel::Create(kMultiVariantPkgPath, "CPUExecutionProvider");
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(std::string(model->GetType().p_), "phi3");
  std::string device_type = model->GetDeviceType().p_;
  std::transform(device_type.begin(), device_type.end(), device_type.begin(), ::tolower);
  EXPECT_EQ(device_type, "cpu");
}

TEST(ModelPackageE2E, SingleVariantCpuGenerateTokens) {
  if (!PackageExists(kCpuOnlyPkgPath)) {
    GTEST_SKIP() << "CPU-only test package not found at " << kCpuOnlyPkgPath;
  }

  auto model = OgaModel::Create(kCpuOnlyPkgPath);
  ASSERT_NE(model, nullptr);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 10);
  params->SetSearchOption("batch_size", 1);

  // Use raw token IDs (BOS token for phi4 is 199999)
  std::vector<int32_t> input_ids = {199999};

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids);

  int tokens_generated = 0;
  while (!generator->IsDone() && tokens_generated < 5) {
    generator->GenerateNextToken();
    tokens_generated++;
  }

  auto sequence = generator->GetSequence(0);
  // Should have at least the input token plus some generated tokens
  EXPECT_GT(sequence.size(), 1u);
}

// --- Pipeline (multi-file) package tests ---
// Tests that a decoder-pipeline model with 3 ONNX files (embeds, transformer, lm_head)
// loads correctly from a package. Uses tiny dummy models that return zeros.

static const char* kPipelinePkgPath = MODEL_PATH "pipeline-cpu.ortpackage";

TEST(ModelPackageE2E, PipelineMultiFileLoadAndType) {
  if (!PackageExists(kPipelinePkgPath)) {
    GTEST_SKIP() << "Pipeline test package not found at " << kPipelinePkgPath;
  }

  auto model = OgaModel::Create(kPipelinePkgPath);
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(std::string(model->GetType().p_), "decoder-pipeline");
  std::string device_type = model->GetDeviceType().p_;
  std::transform(device_type.begin(), device_type.end(), device_type.begin(), ::tolower);
  EXPECT_EQ(device_type, "cpu");
}

TEST(ModelPackageE2E, PipelineMultiFileGenerateTokens) {
  if (!PackageExists(kPipelinePkgPath)) {
    GTEST_SKIP() << "Pipeline test package not found at " << kPipelinePkgPath;
  }

  auto model = OgaModel::Create(kPipelinePkgPath);
  ASSERT_NE(model, nullptr);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 5);
  params->SetSearchOption("batch_size", 1);

  // BOS token = 1 per the pipeline config
  std::vector<int32_t> input_ids = {1};

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids);

  int tokens_generated = 0;
  while (!generator->IsDone() && tokens_generated < 3) {
    generator->GenerateNextToken();
    tokens_generated++;
  }

  auto sequence = generator->GetSequence(0);
  // Should have at least the input token plus generated tokens
  EXPECT_GT(sequence.size(), 1u);
}

// --- OgaCreateConfigWithEp API tests ---

TEST(ModelPackageE2E, CreateConfigWithEpFromPackage) {
  if (!PackageExists(kCpuOnlyPkgPath)) {
    GTEST_SKIP() << "CPU-only test package not found at " << kCpuOnlyPkgPath;
  }

  // Create a config with explicit CPU EP, then create a model from it
  auto config = OgaConfig::Create(kCpuOnlyPkgPath, "cpu");
  ASSERT_NE(config, nullptr);

  auto model = OgaModel::Create(*config);
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(std::string(model->GetType().p_), "phi3");
}

TEST(ModelPackageE2E, CreateConfigWithEpNullFallsBackToAutoDetect) {
  if (!PackageExists(kCpuOnlyPkgPath)) {
    GTEST_SKIP() << "CPU-only test package not found at " << kCpuOnlyPkgPath;
  }

  // NULL ep should auto-detect (same as OgaCreateConfig for packages)
  auto config = OgaConfig::Create(kCpuOnlyPkgPath, nullptr);
  ASSERT_NE(config, nullptr);

  auto model = OgaModel::Create(*config);
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(std::string(model->GetType().p_), "phi3");
}
#endif
