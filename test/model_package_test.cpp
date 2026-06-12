// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "generators.h"
#include "models/model_package.h"
#include "ort_genai_c.h"

namespace {

namespace fs_std = std::filesystem;

// Creates a fresh, empty per-test directory under the build's temp area.
fs_std::path MakeTempDir(const std::string& suffix) {
  static int counter = 0;
  ++counter;
  const auto dir = fs_std::temp_directory_path() /
                   ("ortgenai_mp_test_" + suffix + "_" + std::to_string(counter));
  std::error_code ec;
  fs_std::remove_all(dir, ec);
  fs_std::create_directories(dir);
  return dir;
}

void WriteFile(const fs_std::path& path, const std::string& contents) {
  fs_std::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  out << contents;
}

// Builds a minimal model package that the ORT runtime model_package API can open. The
// package has a single component named "model". Each variant in `variants` becomes a
// directory under `models/model/` with a placeholder `model.onnx` so any consumer that
// walked the on-disk contents would still find something there.
fs_std::path WritePackage(const std::string& suffix,
                          const std::vector<std::pair<std::string, std::string>>& variants) {
  const auto root = MakeTempDir(suffix);

  WriteFile(root / "manifest.json",
            "{\n"
            "  \"schema_version\": 1,\n"
            "  \"components\": [\"model\"]\n"
            "}\n");

  std::string metadata = "{\n  \"component_name\": \"model\",\n  \"variants\": {\n";
  for (size_t i = 0; i < variants.size(); ++i) {
    const auto& [name, ep] = variants[i];
    metadata += "    \"" + name + "\": { \"ep\": \"" + ep + "\" }";
    metadata += (i + 1 == variants.size()) ? "\n" : ",\n";
  }
  metadata += "  }\n}\n";
  WriteFile(root / "models" / "model" / "metadata.json", metadata);

  for (const auto& [name, ep] : variants) {
    (void)ep;
    WriteFile(root / "models" / "model" / name / "model.onnx", "placeholder");
  }
  return root;
}

}  // namespace

TEST(ModelPackageDetection, FlatDirIsNotPackage) {
  const auto root = MakeTempDir("flat");
  WriteFile(root / "genai_config.json", "{}");
  WriteFile(root / "model.onnx", "placeholder");
  EXPECT_FALSE(Generators::IsModelPackage(fs::path{root.string()}));
}

TEST(ModelPackageDetection, MissingPathIsNotPackage) {
  EXPECT_FALSE(Generators::IsModelPackage(fs::path{"/this/path/does/not/exist/12345"}));
}

TEST(ModelPackageDetection, RegularFileIsNotPackage) {
  const auto root = MakeTempDir("file");
  const auto file_path = root / "thing.json";
  WriteFile(file_path, "{}");
  EXPECT_FALSE(Generators::IsModelPackage(fs::path{file_path.string()}));
}

TEST(ModelPackageDetection, ManifestMakesPackage) {
  const auto root = MakeTempDir("manifest");
  WriteFile(root / "manifest.json", "{ \"schema_version\": 1 }");
  EXPECT_TRUE(Generators::IsModelPackage(fs::path{root.string()}));
}

TEST(ModelPackageDetection, RootMetadataAloneIsNotPackage) {
  // A bare metadata.json at the package root is not a usable model package by itself: the
  // ORT runtime requires a manifest.json. IsModelPackage rejects it accordingly.
  const auto root = MakeTempDir("rootmeta");
  WriteFile(root / "metadata.json", "{ \"variants\": {} }");
  EXPECT_FALSE(Generators::IsModelPackage(fs::path{root.string()}));
}

TEST(ConfigResolvePath, EmptyReturnsConfigPath) {
  Generators::Config config;
  config.config_path = fs::path{"/var/models/flat"};
  EXPECT_EQ(config.ResolvePath("").string(), "/var/models/flat");
}

TEST(ConfigResolvePath, PlainRelativeJoinsConfigPath) {
  Generators::Config config;
  config.config_path = fs::path{"/var/models/flat"};
  EXPECT_EQ(config.ResolvePath("tokenizer").string(), "/var/models/flat/tokenizer");
}

TEST(ConfigResolvePath, AbsoluteReturnedAsIs) {
  Generators::Config config;
  config.config_path = fs::path{"/var/models/flat"};
  EXPECT_EQ(config.ResolvePath("/absolute/elsewhere").string(), "/absolute/elsewhere");
}

TEST(ConfigResolvePath, PackageSchemeJoinsPackageRoot) {
  Generators::Config config;
  config.config_path = fs::path{"/var/models/pkg/variants/cpu"};
  config.package_root = fs::path{"/var/models/pkg"};
  EXPECT_EQ(config.ResolvePath("package:shared/tokenizers").string(),
            "/var/models/pkg/shared/tokenizers");
}

TEST(ConfigResolvePath, PackageSchemeBareReturnsPackageRoot) {
  Generators::Config config;
  config.config_path = fs::path{"/var/models/pkg/variants/cpu"};
  config.package_root = fs::path{"/var/models/pkg"};
  EXPECT_EQ(config.ResolvePath("package:").string(), "/var/models/pkg");
}

TEST(ConfigResolvePath, PackageSchemeWithoutPackageRootThrows) {
  Generators::Config config;
  config.config_path = fs::path{"/var/models/flat"};
  EXPECT_THROW(config.ResolvePath("package:shared"), std::runtime_error);
}

TEST(ConfigResolvePath, UnknownSchemeTreatedAsRelativePath) {
  // Unknown scheme-like values (anything other than "package:") are treated as ordinary
  // relative paths joined with config_path. Includes values such as "sha256:..." that may
  // gain meaning in a future iteration.
  Generators::Config config;
  config.config_path = fs::path{"/var/models/flat"};
  EXPECT_EQ(config.ResolvePath("foo:bar").string(), "/var/models/flat/foo:bar");
  EXPECT_EQ(config.ResolvePath("sha256:abcdef").string(), "/var/models/flat/sha256:abcdef");
}

#if ORT_GENAI_HAS_MODEL_PACKAGE

TEST(ModelPackageSelect, SingleCpuVariantAutoDetects) {
  const auto root = WritePackage("autocpu", {{"cpu", "CPUExecutionProvider"}});
  auto& env = Generators::GetOrtEnv();
  auto result = Generators::OpenAndSelectVariant(env, fs::path{root.string()}, "");
  EXPECT_EQ(result.package_root.string(), root.string());
  EXPECT_EQ(fs_std::path(result.variant_dir.string()).filename().string(), "cpu");
}

TEST(ModelPackageSelect, AmbiguousEpsRequireExplicitEp) {
  const auto root = WritePackage(
      "ambig", {{"cpu", "CPUExecutionProvider"}, {"cuda", "CUDAExecutionProvider"}});
  auto& env = Generators::GetOrtEnv();
  EXPECT_THROW(Generators::OpenAndSelectVariant(env, fs::path{root.string()}, ""),
               std::exception);
}

TEST(ModelPackageSelect, ExplicitEpPicksCpu) {
  const auto root = WritePackage(
      "explicit_cpu", {{"cpu", "CPUExecutionProvider"}, {"cuda", "CUDAExecutionProvider"}});
  auto& env = Generators::GetOrtEnv();
  auto result = Generators::OpenAndSelectVariant(env, fs::path{root.string()}, "cpu");
  EXPECT_EQ(fs_std::path(result.variant_dir.string()).filename().string(), "cpu");
}

TEST(ModelPackageSelect, ExplicitFullEpNamePicksVariant) {
  const auto root = WritePackage(
      "explicit_full", {{"cpu", "CPUExecutionProvider"}, {"cuda", "CUDAExecutionProvider"}});
  auto& env = Generators::GetOrtEnv();
  auto result = Generators::OpenAndSelectVariant(env, fs::path{root.string()},
                                                 "CPUExecutionProvider");
  EXPECT_EQ(fs_std::path(result.variant_dir.string()).filename().string(), "cpu");
}

TEST(CreateConfigFromPackage, ReadsTokenizerDirFromVariantConfig) {
  // Build a package whose variant config sets tokenizer_dir to a package-rooted path so we
  // can verify package_root is populated and the resolver routes through it.
  const auto root = MakeTempDir("cfg_tokdir");
  WriteFile(root / "manifest.json",
            "{ \"schema_version\": 1, \"components\": [\"model\"] }");
  WriteFile(root / "models" / "model" / "metadata.json",
            "{ \"component_name\": \"model\","
            " \"variants\": { \"cpu\": { \"ep\": \"CPUExecutionProvider\" } } }");
  WriteFile(root / "models" / "model" / "cpu" / "genai_config.json",
            "{\n"
            "  \"model\": {\n"
            "    \"type\": \"tiny-test-model\",\n"
            "    \"tokenizer_dir\": \"package:shared\",\n"
            "    \"vocab_size\": 16,\n"
            "    \"context_length\": 32\n"
            "  },\n"
            "  \"search\": {}\n"
            "}\n");
  WriteFile(root / "models" / "model" / "cpu" / "model.onnx", "placeholder");
  WriteFile(root / "shared" / "tokenizer.json", "placeholder");

  auto& env = Generators::GetOrtEnv();
  auto config = Generators::CreateConfig(env, root.string().c_str(), "cpu");
  ASSERT_NE(config, nullptr);
  EXPECT_EQ(config->package_root.string(), root.string());
  EXPECT_EQ(config->model.tokenizer_dir, "package:shared");
  EXPECT_EQ(config->ResolvePath(config->model.tokenizer_dir).string(),
            (root / "shared").string());
  EXPECT_EQ(config->ResolvePath("model.onnx").string(),
            (root / "models" / "model" / "cpu" / "model.onnx").string());
}

TEST(CreateConfigFromPackage, EpOnFlatDirIsRejected) {
  const auto root = MakeTempDir("flat_with_ep");
  WriteFile(root / "genai_config.json",
            "{\n  \"model\": { \"type\": \"tiny-test-model\","
            " \"vocab_size\": 16, \"context_length\": 32 },\n"
            "  \"search\": {}\n}\n");
  auto& env = Generators::GetOrtEnv();
  EXPECT_THROW(Generators::CreateConfig(env, root.string().c_str(), "cpu"),
               std::runtime_error);
}

TEST(OgaCreateFromPackageCApi, RejectsFlatDirectories) {
  const auto root = MakeTempDir("oga_flat");
  WriteFile(root / "genai_config.json",
            "{\n  \"model\": { \"type\": \"tiny-test-model\","
            " \"vocab_size\": 16, \"context_length\": 32 },\n"
            "  \"search\": {}\n}\n");

  OgaConfig* cfg = nullptr;
  auto* err = OgaCreateConfigFromPackage(root.string().c_str(), nullptr, &cfg);
  ASSERT_NE(err, nullptr);
  OgaDestroyResult(err);
  EXPECT_EQ(cfg, nullptr);
}

#endif  // ORT_GENAI_HAS_MODEL_PACKAGE
