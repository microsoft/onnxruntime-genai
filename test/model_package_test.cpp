// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include "../src/generators.h"
#include "../src/config.h"
#include "../src/models/model_package.h"

namespace {

// RAII helper: create a fresh unique temp directory and remove it on scope exit so tests are
// portable (no POSIX-only `mkdir -p`/`rm -rf` shelling out) and isolated from each other.
class ScopedTempDir {
 public:
  explicit ScopedTempDir(const std::string& tag) {
    auto base = std::filesystem::temp_directory_path();
    // Use a per-process+per-test-tag suffix to keep parallel ctest invocations safe.
    auto pid = std::to_string(static_cast<unsigned long long>(
        std::hash<std::string>{}(tag) ^
        std::hash<std::thread::id>{}(std::this_thread::get_id())));
    path_ = base / ("genai_test_" + tag + "_" + pid);
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
    std::filesystem::create_directories(path_);
  }
  ~ScopedTempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  ScopedTempDir(const ScopedTempDir&) = delete;
  ScopedTempDir& operator=(const ScopedTempDir&) = delete;

  const std::filesystem::path& path() const { return path_; }
  std::string string() const { return path_.string(); }
  std::filesystem::path operator/(const std::string& tail) const { return path_ / tail; }

  // Create an empty file at <root>/<rel>, creating any intermediate directories.
  void touch(const std::string& rel) const {
    auto target = path_ / rel;
    std::filesystem::create_directories(target.parent_path());
    std::ofstream(target).close();
  }

 private:
  std::filesystem::path path_;
};

}  // namespace

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

TEST(JsonMergePatch, UnicodeEscapeRoundTrip) {
  // Regression: parsing \uXXXX and re-serializing must not double-escape the backslash.
  // Inputs use \u0041 ('A'), \u00E9 ('é'), and a surrogate pair for U+1F600 ("😀").
  std::string base = R"({"keep":"x"})";
  std::string patch = R"({"a":"\u0041","b":"caf\u00e9","c":"\uD83D\uDE00"})";
  std::string result = Generators::JsonMergePatch(base, patch);
  // The literal "\u" sequence must not survive (it would be invalid JSON when re-parsed by ORT)
  // and must definitely not have been double-escaped to "\\u".
  EXPECT_EQ(result.find("\\\\u"), std::string::npos);
  // The decoded characters should appear as UTF-8 bytes in the output.
  EXPECT_NE(result.find("\"a\":\"A\""), std::string::npos);
  EXPECT_NE(result.find("caf\xc3\xa9"), std::string::npos);       // 'é' = 0xC3 0xA9
  EXPECT_NE(result.find("\xf0\x9f\x98\x80"), std::string::npos);  // U+1F600 = 0xF0 0x9F 0x98 0x80
  EXPECT_NE(result.find("\"keep\":\"x\""), std::string::npos);
}

TEST(JsonMergePatch, EscapesControlCharactersOnOutput) {
  // Parsing \u0001 into a raw 0x01 byte and emitting it raw would produce invalid JSON.
  // The serializer must escape U+0000..U+001F that don't have a short escape back to \u00XX.
  std::string base = R"({})";
  std::string patch = R"({"ctl":"a\u0001b"})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("\\u0001"), std::string::npos);
  // The raw 0x01 byte must not appear in the serialized output.
  EXPECT_EQ(result.find('\x01'), std::string::npos);
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

// --- Package detection tests ---

TEST(IsModelPackage, NonExistentPath) {
  // A path that does not exist should never be flagged as a model package, regardless of OS.
  auto missing = std::filesystem::temp_directory_path() /
                 "genai_test_nonexistent_model_package_path_12345";
  std::error_code ec;
  std::filesystem::remove_all(missing, ec);
  EXPECT_FALSE(Generators::IsModelPackage(fs::path(missing.string())));
}

TEST(IsModelPackage, FlatDirectory) {
  // A temp dir without manifest.json and without any component-shaped child is not a package.
  ScopedTempDir tmp("flat_dir");
  EXPECT_FALSE(Generators::IsModelPackage(fs::path(tmp.string())));
}

TEST(IsModelPackage, PackageDirectoryWithManifest) {
  // A temp dir with a top-level manifest.json is detected as a package.
  ScopedTempDir tmp("package_with_manifest");
  tmp.touch("manifest.json");
  EXPECT_TRUE(Generators::IsModelPackage(fs::path(tmp.string())));
}

TEST(IsModelPackage, ManifestlessIsNotPackage) {
  // The redesigned ORT model_package schema requires a top-level manifest.json. Component
  // directories alone (with or without a per-component file) no longer qualify.
  ScopedTempDir tmp("manifestless_package");
  tmp.touch("decoder/component.json");
  EXPECT_FALSE(Generators::IsModelPackage(fs::path(tmp.string())));
}

TEST(IsModelPackage, ConfigsOnlyIsNotPackage) {
  // A directory containing only configs/ (no manifest) must not be flagged as a package.
  // configs/ is a GenAI-side convention, not a model_package marker.
  ScopedTempDir tmp("configs_only");
  tmp.touch("configs/genai_config.json");
  EXPECT_FALSE(Generators::IsModelPackage(fs::path(tmp.string())));
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
// or the genai_config_overlay.json merge.

static const char* kKitchenSinkPkgPath = MODEL_PATH "kitchen-sink-cpu.ortpackage";
static const char* kBadOverlayComponentPkgPath = MODEL_PATH "bad-overlay-component.ortpackage";

TEST(ModelPackageNormalize, OverlayMergePatchSemantics) {
  if (!PackageExists(kKitchenSinkPkgPath)) {
    GTEST_SKIP() << "Kitchen-sink test package not found at " << kKitchenSinkPkgPath;
  }
  // The fixture's base genai_config.json sets a subset of decoder.session_options; its
  // <variant>/genai_config_overlay.json sets the rest AND deliberately re-sets two base fields
  // to different values. Loading the package exercises both halves of the new contract:
  //   * base-only fields survive (overlay didn't touch them),
  //   * overlay-only fields appear,
  //   * fields set in both follow merge-patch semantics: the overlay wins.
  auto config = Generators::CreateConfig(Generators::GetOrtEnv(), kKitchenSinkPkgPath,
                                         /*settings=*/nullptr, /*ep=*/nullptr);
  ASSERT_NE(config, nullptr);
  const auto& so = config->model.decoder.session_options;

  // Base-only fields: overlay did not touch them, so they remain.
  EXPECT_EQ(so.intra_op_num_threads.value_or(-1), 99);
  EXPECT_TRUE(so.enable_cpu_mem_arena.value_or(false));

  // Fields set in both base and overlay: overlay wins (RFC 7396 merge patch).
  EXPECT_EQ(so.log_id.value_or(""), "overlay-log-id");
  EXPECT_EQ(so.graph_optimization_level.value_or(ORT_DISABLE_ALL), ORT_ENABLE_BASIC);

  // Overlay-only fields populate previously-unset typed slots.
  EXPECT_EQ(so.inter_op_num_threads.value_or(-1), 2);
  EXPECT_EQ(so.log_severity_level.value_or(-1), 3);
  EXPECT_EQ(so.log_verbosity_level.value_or(-1), 1);
  EXPECT_TRUE(so.enable_mem_pattern.value_or(false));
  EXPECT_EQ(so.enable_profiling.value_or(""), "/tmp/variant-profile");
  EXPECT_EQ(so.custom_ops_library.value_or(""), "libcustomops.so");

  // Unknown keys land in config_entries.
  auto entry = std::find_if(so.config_entries.begin(), so.config_entries.end(),
                            [](const auto& p) { return p.first == "session.disable_prepacking"; });
  ASSERT_NE(entry, so.config_entries.end());
  EXPECT_EQ(entry->second, "1");

  // Per-role asset_dir is recorded and points at the selected variant directory.
  EXPECT_FALSE(config->model.decoder.asset_dir.string().empty());
  EXPECT_NE(config->model.decoder.asset_dir.string().find("cpu"), std::string::npos);
}

TEST(ModelPackageNormalize, OverlayIntroducedComponentThrows) {
  if (!PackageExists(kBadOverlayComponentPkgPath)) {
    GTEST_SKIP() << "bad-overlay-component fixture not found";
  }
  // The variant's genai_config_overlay.json introduces model.vision.component =
  // "vision-not-in-package". Component selection ran against the base config (decoder only),
  // so the new component reference must be rejected by NormalizePackageIntoConfig.
  EXPECT_THROW(
      Generators::CreateConfig(Generators::GetOrtEnv(), kBadOverlayComponentPkgPath,
                               /*settings=*/nullptr, /*ep=*/nullptr),
      std::runtime_error);
}
#endif
