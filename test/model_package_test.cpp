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

// --- RFC 7386 JSON Merge Patch: additional edge cases ---

TEST(JsonMergePatch, DeepNestedMerge) {
  std::string base = R"({"a":{"b":{"c":1,"d":2}}})";
  std::string patch = R"({"a":{"b":{"c":99,"e":3}}})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("99"), std::string::npos);
  EXPECT_NE(result.find("\"d\":2"), std::string::npos);
  EXPECT_NE(result.find("\"e\":3"), std::string::npos);
  EXPECT_EQ(result.find("\"c\":1"), std::string::npos);
}

TEST(JsonMergePatch, NullPatchDeletesNestedKey) {
  std::string base = R"({"model":{"decoder":{"hidden_size":768,"num_heads":12}}})";
  std::string patch = R"({"model":{"decoder":{"num_heads":null}}})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("768"), std::string::npos);
  EXPECT_EQ(result.find("num_heads"), std::string::npos);
}

TEST(JsonMergePatch, ScalarPatchReplacesObject) {
  std::string base = R"({"a":{"nested":true}})";
  std::string patch = R"({"a":"scalar"})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("\"a\":\"scalar\""), std::string::npos);
  EXPECT_EQ(result.find("nested"), std::string::npos);
}

TEST(JsonMergePatch, ObjectPatchReplacesScalar) {
  std::string base = R"({"a":"scalar"})";
  std::string patch = R"({"a":{"nested":true}})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("\"nested\":true"), std::string::npos);
  EXPECT_EQ(result.find("\"scalar\""), std::string::npos);
}

TEST(JsonMergePatch, DeleteMissingKeyIsNoop) {
  std::string base = R"({"a":1})";
  std::string patch = R"({"nonexistent":null})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("\"a\":1"), std::string::npos);
  EXPECT_EQ(result.find("nonexistent"), std::string::npos);
}

TEST(JsonMergePatch, EmptyObjectPatchIsNoop) {
  std::string base = R"({"a":1,"b":2})";
  std::string patch = R"({})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("\"a\":1"), std::string::npos);
  EXPECT_NE(result.find("\"b\":2"), std::string::npos);
}

TEST(JsonMergePatch, NullPatchReplacesObject) {
  std::string base = R"({"a":1})";
  std::string patch = R"(null)";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_EQ(result, "null");
}

TEST(JsonMergePatch, ArrayPatchReplacesArray) {
  std::string base = R"({"eos":[1,2]})";
  std::string patch = R"({"eos":[3]})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("[3]"), std::string::npos);
  EXPECT_EQ(result.find("[1,2]"), std::string::npos);
}

TEST(JsonMergePatch, StringPatchReplacesObject) {
  std::string base = R"({"a":{"b":1}})";
  std::string patch = R"("hello")";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_EQ(result, "\"hello\"");
}

// --- EP name normalization tests ---

TEST(NormalizeEpName, ShortAliasesAreCaseInsensitive) {
  EXPECT_EQ(Generators::NormalizeEpName("cuda"), "CUDAExecutionProvider");
  EXPECT_EQ(Generators::NormalizeEpName("CUDA"), "CUDAExecutionProvider");
  EXPECT_EQ(Generators::NormalizeEpName("Cuda"), "CUDAExecutionProvider");
  EXPECT_EQ(Generators::NormalizeEpName("cpu"), "CPUExecutionProvider");
  EXPECT_EQ(Generators::NormalizeEpName("dml"), "DmlExecutionProvider");
  EXPECT_EQ(Generators::NormalizeEpName("qnn"), "QNNExecutionProvider");
  EXPECT_EQ(Generators::NormalizeEpName("webgpu"), "WebGpuExecutionProvider");
}

TEST(NormalizeEpName, FullNamePassesThrough) {
  EXPECT_EQ(Generators::NormalizeEpName("CUDAExecutionProvider"), "CUDAExecutionProvider");
  EXPECT_EQ(Generators::NormalizeEpName("CPUExecutionProvider"), "CPUExecutionProvider");
}

TEST(NormalizeEpName, UnknownReturnsInput) {
  EXPECT_EQ(Generators::NormalizeEpName("UnknownProvider"), "UnknownProvider");
}

// --- EpNameToGenAIProviderName tests ---

TEST(EpNameToGenAIProviderName, MapsKnownEps) {
  EXPECT_EQ(Generators::EpNameToGenAIProviderName("CUDAExecutionProvider"), "cuda");
  EXPECT_EQ(Generators::EpNameToGenAIProviderName("DmlExecutionProvider"), "DML");
  EXPECT_EQ(Generators::EpNameToGenAIProviderName("QNNExecutionProvider"), "QNN");
}

TEST(EpNameToGenAIProviderName, UnknownReturnsInput) {
  EXPECT_EQ(Generators::EpNameToGenAIProviderName("UnknownEP"), "UnknownEP");
}

// --- EffectiveSessionOptions tests ---

TEST(EffectiveSessionOptions, ReturnsRoleSoWhenSet) {
  Generators::Config config;
  config.model.decoder.session_options.intra_op_num_threads = 16;
  std::optional<Generators::Config::SessionOptions> role_so;
  role_so.emplace();
  role_so->intra_op_num_threads = 4;
  const auto& result = Generators::EffectiveSessionOptions(config, role_so);
  EXPECT_EQ(result.intra_op_num_threads.value_or(-1), 4);
}

TEST(EffectiveSessionOptions, FallsBackToDecoderWhenEmpty) {
  Generators::Config config;
  config.model.decoder.session_options.intra_op_num_threads = 16;
  std::optional<Generators::Config::SessionOptions> role_so;
  const auto& result = Generators::EffectiveSessionOptions(config, role_so);
  EXPECT_EQ(result.intra_op_num_threads.value_or(-1), 16);
}

// --- ApplyVariantFileSessionOptions unit tests ---

#if ORT_HAS_MODEL_PACKAGE
using KV = std::pair<std::string, std::string>;
using KVList = std::vector<KV>;

TEST(ApplyVariantFileSessionOptions, FillsUnsetTypedFields) {
  Generators::Config::SessionOptions so;
  KVList variant_so = {
      {"intra_op_num_threads", "4"},
      {"inter_op_num_threads", "2"},
      {"log_severity_level", "3"},
      {"log_verbosity_level", "1"},
      {"enable_cpu_mem_arena", "false"},
      {"enable_mem_pattern", "true"},
      {"log_id", "test"},
      {"enable_profiling", "/tmp/p"},
      {"custom_ops_library", "libfoo.so"},
      {"graph_optimization_level", "ORT_ENABLE_BASIC"},
  };
  Generators::ApplyVariantFileSessionOptions(so, variant_so, {}, "");
  EXPECT_EQ(*so.intra_op_num_threads, 4);
  EXPECT_EQ(*so.inter_op_num_threads, 2);
  EXPECT_EQ(*so.log_severity_level, 3);
  EXPECT_EQ(*so.log_verbosity_level, 1);
  EXPECT_FALSE(*so.enable_cpu_mem_arena);
  EXPECT_TRUE(*so.enable_mem_pattern);
  EXPECT_EQ(*so.log_id, "test");
  EXPECT_EQ(*so.enable_profiling, "/tmp/p");
  EXPECT_EQ(*so.custom_ops_library, "libfoo.so");
  EXPECT_EQ(*so.graph_optimization_level, ORT_ENABLE_BASIC);
}

TEST(ApplyVariantFileSessionOptions, ExistingValuesWin) {
  Generators::Config::SessionOptions so;
  so.intra_op_num_threads = 99;
  so.enable_cpu_mem_arena = true;
  so.log_id = "existing";
  so.graph_optimization_level = ORT_ENABLE_ALL;
  KVList variant_so = {
      {"intra_op_num_threads", "4"},
      {"enable_cpu_mem_arena", "false"},
      {"log_id", "variant"},
      {"graph_optimization_level", "ORT_ENABLE_BASIC"},
  };
  Generators::ApplyVariantFileSessionOptions(so, variant_so, {}, "");
  EXPECT_EQ(*so.intra_op_num_threads, 99);
  EXPECT_TRUE(*so.enable_cpu_mem_arena);
  EXPECT_EQ(*so.log_id, "existing");
  EXPECT_EQ(*so.graph_optimization_level, ORT_ENABLE_ALL);
}

TEST(ApplyVariantFileSessionOptions, UnknownKeysGoToConfigEntries) {
  Generators::Config::SessionOptions so;
  KVList variant_so = {{"session.custom_key", "val"}};
  Generators::ApplyVariantFileSessionOptions(so, variant_so, {}, "");
  ASSERT_EQ(so.config_entries.size(), 1u);
  EXPECT_EQ(so.config_entries[0].first, "session.custom_key");
  EXPECT_EQ(so.config_entries[0].second, "val");
}

TEST(ApplyVariantFileSessionOptions, ExistingConfigEntryWins) {
  Generators::Config::SessionOptions so;
  so.config_entries.emplace_back("session.key", "existing");
  KVList variant_so = {{"session.key", "variant"}};
  Generators::ApplyVariantFileSessionOptions(so, variant_so, {}, "");
  EXPECT_EQ(so.config_entries[0].second, "existing");
  EXPECT_EQ(so.config_entries.size(), 1u);
}

TEST(ApplyVariantFileSessionOptions, InvalidGraphOptLevelThrows) {
  Generators::Config::SessionOptions so;
  KVList variant_so = {{"graph_optimization_level", "INVALID"}};
  EXPECT_THROW(
      Generators::ApplyVariantFileSessionOptions(so, variant_so, {}, ""),
      std::runtime_error);
}

TEST(ApplyVariantFileSessionOptions, CpuEpSkipsProviderOptions) {
  Generators::Config::SessionOptions so;
  Generators::ApplyVariantFileSessionOptions(so, {}, {}, "CPUExecutionProvider");
  EXPECT_TRUE(so.provider_options.empty());
}

TEST(ApplyVariantFileSessionOptions, CpuEpWithProviderOptionsThrows) {
  Generators::Config::SessionOptions so;
  KVList po = {{"device_id", "0"}};
  EXPECT_THROW(
      Generators::ApplyVariantFileSessionOptions(so, {}, po, "CPUExecutionProvider"),
      std::runtime_error);
}

TEST(ApplyVariantFileSessionOptions, EmptyEpSkipsProviderOptions) {
  Generators::Config::SessionOptions so;
  Generators::ApplyVariantFileSessionOptions(so, {}, {}, "");
  EXPECT_TRUE(so.provider_options.empty());
}

TEST(ApplyVariantFileSessionOptions, AppendsNewProviderEntry) {
  Generators::Config::SessionOptions so;
  KVList po = {{"htp_mode", "burst"}, {"soc_model", "60"}};
  Generators::ApplyVariantFileSessionOptions(so, {}, po, "CUDAExecutionProvider");
  ASSERT_EQ(so.provider_options.size(), 1u);
  EXPECT_EQ(so.provider_options[0].name, "cuda");
  EXPECT_EQ(so.provider_options[0].options.size(), 2u);
}

TEST(ApplyVariantFileSessionOptions, BackfillsMissingProviderOptionKeys) {
  Generators::Config::SessionOptions so;
  Generators::Config::ProviderOptions existing;
  existing.name = "cuda";
  existing.options.emplace_back("device_id", "1");
  so.provider_options.push_back(existing);
  KVList po = {{"device_id", "0"}, {"new_key", "val"}};
  Generators::ApplyVariantFileSessionOptions(so, {}, po, "CUDAExecutionProvider");
  ASSERT_EQ(so.provider_options.size(), 1u);
  EXPECT_EQ(so.provider_options[0].options[0].second, "1");  // existing wins
  EXPECT_EQ(so.provider_options[0].options.size(), 2u);       // new_key added
  auto new_key_it = std::find_if(so.provider_options[0].options.begin(),
                                  so.provider_options[0].options.end(),
                                  [](const auto& p) { return p.first == "new_key"; });
  ASSERT_NE(new_key_it, so.provider_options[0].options.end());
  EXPECT_EQ(new_key_it->second, "val");
}
#endif  // ORT_HAS_MODEL_PACKAGE (ApplyVariantFileSessionOptions tests)

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

// --- Config::FromPackage tests ---

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
      fs::path("/tmp/fake_package/configs"), merged_json);

  EXPECT_EQ(config->model.type, "phi3");
  EXPECT_EQ(config->model.context_length, 4096);
  EXPECT_EQ(config->model.vocab_size, 32000);
  EXPECT_EQ(config->model.decoder.component, "decoder");
  EXPECT_EQ(config->model.decoder.hidden_size, 3072);
  EXPECT_EQ(config->model.decoder.num_attention_heads, 24);
  EXPECT_EQ(config->model.decoder.num_hidden_layers, 32);
  EXPECT_EQ(config->search.max_length, 4096);
  EXPECT_EQ(config->config_path.string(), "/tmp/fake_package/configs");
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
      fs::path("/tmp/fake_package/configs"), merged);

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

// --- Targeted regression tests for the package-load error paths and merge logic ---
//
// Each test loads a small fixture that exercises one branch of NormalizePackageIntoConfig
// or ApplyVariantFileOptions. The fixtures live alongside the existing package fixtures.

static const char* kKitchenSinkPkgPath = MODEL_PATH "kitchen-sink-cpu.ortpackage";
static const char* kBadPipelineMismatchPkgPath = MODEL_PATH "bad-pipeline-mismatch.ortpackage";
static const char* kBadOverlayComponentPkgPath = MODEL_PATH "bad-overlay-component.ortpackage";
static const char* kBadCpuPoPkgPath = MODEL_PATH "bad-cpu-po.ortpackage";

TEST(ModelPackageNormalize, MergesVariantSoAndGenAIOverlay) {
  if (!PackageExists(kKitchenSinkPkgPath)) {
    GTEST_SKIP() << "Kitchen-sink test package not found at " << kKitchenSinkPkgPath;
  }
  // The fixture's variant.json declares every typed SO field; its
  // consumer_metadata.genai_config_overlay re-sets a subset. One load exercises both
  // halves of the merge contract:
  //   * overlay-set fields carry the genai_config value (genai wins on conflict),
  //   * variant-only fields carry the variant value (variant fills gaps).
  auto config = Generators::CreateConfig(Generators::GetOrtEnv(), kKitchenSinkPkgPath,
                                          /*settings=*/nullptr, /*ep=*/nullptr);
  ASSERT_NE(config, nullptr);
  const auto& so = config->model.decoder.session_options;

  // Fields the overlay re-sets: genai_config wins.
  EXPECT_EQ(so.intra_op_num_threads.value_or(-1), 99);
  EXPECT_TRUE(so.enable_cpu_mem_arena.value_or(false));
  EXPECT_EQ(so.log_id.value_or(""), "genai-log-id");
  EXPECT_EQ(so.graph_optimization_level.value_or(ORT_DISABLE_ALL), ORT_ENABLE_ALL);

  // Fields only the variant sets: variant fills the gap.
  EXPECT_EQ(so.inter_op_num_threads.value_or(-1), 2);
  EXPECT_EQ(so.log_severity_level.value_or(-1), 3);
  EXPECT_EQ(so.log_verbosity_level.value_or(-1), 1);
  EXPECT_TRUE(so.enable_mem_pattern.value_or(false));
  EXPECT_EQ(so.enable_profiling.value_or(""), "/tmp/variant-profile");
  EXPECT_EQ(so.custom_ops_library.value_or(""), "libcustomops.so");

  // Unknown-key fallback for arbitrary entries.
  auto entry = std::find_if(so.config_entries.begin(), so.config_entries.end(),
                            [](const auto& p) { return p.first == "session.disable_prepacking"; });
  ASSERT_NE(entry, so.config_entries.end());
  EXPECT_EQ(entry->second, "1");
}

TEST(ModelPackageNormalize, PipelineSizeMismatchThrows) {
  if (!PackageExists(kBadPipelineMismatchPkgPath)) {
    GTEST_SKIP() << "bad-pipeline-mismatch fixture not found";
  }
  // Pipeline declares 2 stages, variant has 1 file. NormalizePackageIntoConfig requires
  // equal counts for positional mapping.
  EXPECT_THROW(
      Generators::CreateConfig(Generators::GetOrtEnv(), kBadPipelineMismatchPkgPath,
                                /*settings=*/nullptr, /*ep=*/nullptr),
      std::runtime_error);
}

TEST(ModelPackageNormalize, OverlayIntroducedComponentThrows) {
  if (!PackageExists(kBadOverlayComponentPkgPath)) {
    GTEST_SKIP() << "bad-overlay-component fixture not found";
  }
  // The variant's consumer_metadata overlay introduces model.vision.component =
  // "vision-not-in-package". Selection happened against the base config (decoder only), so
  // the new component reference must be rejected.
  EXPECT_THROW(
      Generators::CreateConfig(Generators::GetOrtEnv(), kBadOverlayComponentPkgPath,
                                /*settings=*/nullptr, /*ep=*/nullptr),
      std::runtime_error);
}

TEST(ModelPackageNormalize, CpuVariantWithProviderOptionsThrows) {
  if (!PackageExists(kBadCpuPoPkgPath)) {
    GTEST_SKIP() << "bad-cpu-po fixture not found";
  }
  // CPU has no GenAI provider tag, so non-empty variant provider_options under a CPU file
  // would be silently dropped. ApplyVariantFileOptions rejects that as a producer error.
  EXPECT_THROW(
      Generators::CreateConfig(Generators::GetOrtEnv(), kBadCpuPoPkgPath,
                                /*settings=*/nullptr, /*ep=*/nullptr),
      std::runtime_error);
}
#endif
