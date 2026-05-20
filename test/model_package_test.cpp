// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for the ORT-backed `Generators::ModelPackageContext` and the
// `ParseVariantManifest` helper used by GenAI's multi-file runner.
//
// Each test materialises a synthetic v4 package layout under
// `std::filesystem::temp_directory_path() / <unique>` and exercises one slice
// of the surface. The fixture cleans up after itself.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <string_view>

#include "json.h"
#include "models/model_package.h"
#include "models/onnxruntime_api.h"

namespace Generators::test {

namespace {

class TempDir {
 public:
  TempDir() {
    std::random_device rd;
    std::uniform_int_distribution<uint64_t> dist;
    path_ = std::filesystem::temp_directory_path() /
            ("genai-model-package-test-" + std::to_string(dist(rd)));
    std::filesystem::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;

  const std::filesystem::path& path() const { return path_; }
  fs::path fs_path() const { return fs::path(path_.string()); }

 private:
  std::filesystem::path path_;
};

bool SafePackagePathFragment(std::string_view value) {
  return !value.empty() &&
         value != "." &&
         value != ".." &&
         value.find("..") == std::string_view::npos &&
         value.find('/') == std::string_view::npos &&
         value.find('\\') == std::string_view::npos &&
         value.find(':') == std::string_view::npos;
}

void TouchDeclaredVariantFiles(const std::filesystem::path& variant_json_path,
                               std::string_view contents) {
  if (variant_json_path.filename() != "variant.json") return;

  JSON::Document doc;
  try {
    doc = JSON::ParseDocument(std::string(contents));
  } catch (...) {
    return;
  }
  if (!doc.IsObject()) return;

  auto files_it = doc.AsObject().find("files");
  if (files_it == doc.AsObject().end() || !files_it->second.IsArray()) return;

  for (const auto& file_doc : files_it->second.AsArray()) {
    if (!file_doc.IsObject()) continue;
    auto filename_it = file_doc.AsObject().find("filename");
    if (filename_it == file_doc.AsObject().end() || !filename_it->second.IsString()) continue;

    const std::string& filename = filename_it->second.AsString();
    if (!SafePackagePathFragment(filename)) continue;

    const std::filesystem::path model_path = variant_json_path.parent_path() / filename;
    if (std::filesystem::exists(model_path)) continue;

    std::ofstream model_file(model_path, std::ios::binary);
    ASSERT_TRUE(model_file.is_open()) << "cannot open " << model_path;
  }
}

void WriteFile(const std::filesystem::path& p, std::string_view contents) {
  std::filesystem::create_directories(p.parent_path());
  std::ofstream out(p, std::ios::binary);
  ASSERT_TRUE(out.is_open()) << "cannot open " << p;
  out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  out.close();
  TouchDeclaredVariantFiles(p, contents);
}

#if ORT_API_VERSION >= 27

// Materialise a representative two-component package:
//   <root>/manifest.json                   {"schema_version":1,"components":["decoder","embedding"]}
//   <root>/configs/                        (empty: not exercised here directly)
//   <root>/models/decoder/metadata.json           cpu, cuda variants (in this order)
//   <root>/models/decoder/cpu/variant.json        single file, overlay sets context_length=2048
//   <root>/models/decoder/cuda/variant.json       single file, overlay sets context_length=4096
//   <root>/models/embedding/metadata.json         single variant cpu, EP compat = [CPU, WebGPU]
//   <root>/models/embedding/cpu/variant.json      single file, no consumer_metadata
void BuildTwoComponentPackage(const std::filesystem::path& root) {
  WriteFile(root / "manifest.json", R"({
    "schema_version": 1,
    "components": ["decoder", "embedding"]
  })");

  std::filesystem::create_directories(root / "configs");

  WriteFile(root / "models" / "decoder" / "metadata.json", R"({
    "variants": {
      "cpu": {
        "ep_compatibility": [{"ep": "CPUExecutionProvider"}]
      },
      "cuda": {
        "ep_compatibility": [
          {"ep": "CUDAExecutionProvider", "compatibility_string": "sm_80,sm_90"}
        ]
      }
    }
  })");

  WriteFile(root / "models" / "decoder" / "cpu" / "variant.json", R"({
    "files": [{"filename": "model.onnx"}],
    "consumer_metadata": {
      "genai_config_overlay": {"model": {"context_length": 2048}}
    }
  })");

  WriteFile(root / "models" / "decoder" / "cuda" / "variant.json", R"({
    "files": [{"filename": "model.onnx"}],
    "consumer_metadata": {
      "genai_config_overlay": {"model": {"context_length": 4096}}
    }
  })");

  WriteFile(root / "models" / "embedding" / "metadata.json", R"({
    "variants": {
      "cpu": {
        "ep_compatibility": [
          {"ep": "CPUExecutionProvider"},
          {"ep": "WebGpuExecutionProvider"}
        ]
      }
    }
  })");

  WriteFile(root / "models" / "embedding" / "cpu" / "variant.json", R"({
    "files": [{"filename": "embed.onnx"}]
  })");
}

// Find first object that contains key 'k' and return the value's number,
// helper to keep test bodies readable when dispatching on overlay shape.
double NumberAt(const JSON::Document& d, std::initializer_list<const char*> path) {
  const JSON::Document* cur = &d;
  for (const char* key : path) {
    EXPECT_TRUE(cur->IsObject()) << "expected object at " << key;
    auto it = cur->AsObject().find(key);
    EXPECT_NE(it, cur->AsObject().end()) << "missing key " << key;
    cur = &it->second;
  }
  EXPECT_TRUE(cur->IsNumber());
  return cur->AsNumber();
}

ModelPackageSelectionOptions OnePriority(std::string ep, std::optional<std::string> device = std::nullopt) {
  ModelPackageSelectionOptions opts;
  opts.ep_priority.push_back({std::move(ep), std::move(device)});
  return opts;
}

bool CanSelectEp(const std::string& ep) {
  static std::map<std::string, bool> cache;
  auto it = cache.find(ep);
  if (it != cache.end()) return it->second;

  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1,"components":["decoder"]})");
  WriteFile(dir.path() / "models" / "decoder" / "metadata.json",
            std::string{R"({"variants":{"v":{"ep_compatibility":[{"ep":")"} + ep + R"("}]}}}"});
  WriteFile(dir.path() / "models" / "decoder" / "v" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");

  bool can_select = false;
  try {
    auto ctx = ModelPackageContext::Open(dir.fs_path());
    can_select = ctx != nullptr && ctx->SelectComponent(0, OnePriority(ep)) != nullptr;
  } catch (...) {
    can_select = false;
  }

  return cache.emplace(ep, can_select).first->second;
}

std::size_t FindVariantIndex(const ModelPackageContext& ctx, std::size_t cix, std::string_view variant_name) {
  for (std::size_t vix = 0; vix < ctx.NumVariants(cix); ++vix) {
    if (ctx.VariantName(cix, vix) == variant_name) return vix;
  }
  throw std::runtime_error("variant not found: " + std::string(variant_name));
}

#define SKIP_IF_CANNOT_SELECT_EP(ep)                                                              \
  if (!CanSelectEp(ep)) {                                                                         \
    GTEST_SKIP() << ep << " is not available for ORT model-package selection in this test build"; \
  }

#endif  // ORT_API_VERSION >= 27

}  // namespace

// ============================================================================
// Detection
// ============================================================================

TEST(ModelPackageContextTest, EmptyDirectoryReturnsNullptr) {
  TempDir dir;
  EXPECT_EQ(ModelPackageContext::Open(dir.fs_path()), nullptr);
}

TEST(ModelPackageContextTest, FlatDirReturnsNullptr) {
  TempDir dir;
  // A flat-dir model: only genai_config.json + ONNX files at the root, no
  // subdirs that could contain metadata.json. Detection must say nullptr.
  WriteFile(dir.path() / "genai_config.json", R"({"model":{"type":"phi3"}})");
  WriteFile(dir.path() / "model.onnx", "fake");
  EXPECT_EQ(ModelPackageContext::Open(dir.fs_path()), nullptr);
}

TEST(ModelPackageContextTest, ConfigsBucketAloneIsNotV4) {
  // A directory with only a configs/ subdirectory but no manifest and no
  // component metadata.json should NOT be treated as v4 — configs/ alone
  // is too weak a marker (a flat-dir model could plausibly contain one).
  TempDir dir;
  std::filesystem::create_directories(dir.path() / "configs");
  WriteFile(dir.path() / "configs" / "genai_config.json", R"({"model":{}})");
  EXPECT_EQ(ModelPackageContext::Open(dir.fs_path()), nullptr);
}

TEST(ModelPackageContextTest, NoManifestModelsMetadataReturnsNullptr) {
  // Match ORT's package-root parser: models/<component>/metadata.json alone
  // is not a package marker unless manifest.json is present.
  TempDir dir;
  WriteFile(dir.path() / "models" / "decoder" / "metadata.json", R"({
    "variants": {"cpu": {"ep_compatibility": [{"ep": "CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "models" / "decoder" / "cpu" / "variant.json",
            R"({"files":[{"filename":"model.onnx"}]})");
  EXPECT_EQ(ModelPackageContext::Open(dir.fs_path()), nullptr);
}

#if ORT_API_VERSION < 27

TEST(ModelPackageContextTest, PackageMarkerRequiresOrtModelPackageApi) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1,"components":["decoder"]})");
  EXPECT_THROW(ModelPackageContext::Open(dir.fs_path()), std::exception);
}

#else

TEST(ModelPackageContextTest, ComponentRootMetadataIsRecognized) {
  // ORT also accepts opening a single component root directly.
  TempDir dir;
  WriteFile(dir.path() / "metadata.json", R"({
    "component_name": "decoder",
    "variants": {"cpu": {"ep_compatibility": [{"ep": "CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "cpu" / "variant.json",
            R"({"files":[{"filename":"model.onnx"}]})");
  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  EXPECT_EQ(ctx->NumComponents(), 1u);
  EXPECT_EQ(ctx->ComponentName(0), "decoder");
}

TEST(ModelPackageContextTest, OpenSucceedsOnCanonicalTwoComponentPackage) {
  TempDir dir;
  BuildTwoComponentPackage(dir.path());
  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  EXPECT_EQ(ctx->NumComponents(), 2u);
  EXPECT_EQ(ctx->ComponentName(0), "decoder");
  EXPECT_EQ(ctx->ComponentName(1), "embedding");
}

// ============================================================================
// Manifest validation
// ============================================================================

TEST(ModelPackageContextTest, ManifestNonObjectRootThrows) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"(["not", "an", "object"])");
  EXPECT_THROW(ModelPackageContext::Open(dir.fs_path()), std::exception);
}

TEST(ModelPackageContextTest, ManifestRejectsObjectFormComponents) {
  // Spec format is the bare-string array; object-form entries are
  // explicitly unsupported even when they carry a `name` field. The
  // producer is expected to align with ORT's current schema rather than us forking
  // the parser to track every variant of the shape.
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({
    "schema_version": 1,
    "components": [{"name": "decoder", "metadata": "decoder/metadata.json"}]
  })");
  EXPECT_THROW(ModelPackageContext::Open(dir.fs_path()), std::exception);
}

TEST(ModelPackageContextTest, ManifestUnsupportedSchemaVersionThrows) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version": 99, "components":["x"]})");
  EXPECT_THROW(ModelPackageContext::Open(dir.fs_path()), std::exception);
}

TEST(ModelPackageContextTest, ManifestRejectsStringSchemaVersion) {
  // ORT's current parser requires integer schema_version.
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":"1.0","components":["decoder"]})");
  EXPECT_THROW(ModelPackageContext::Open(dir.fs_path()), std::exception);
}

TEST(ModelPackageContextTest, ManifestAcceptsNumberSchemaVersion) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1,"components":["decoder"]})");
  WriteFile(dir.path() / "models" / "decoder" / "metadata.json", R"({
    "variants": {"cpu": {"ep_compatibility":[{"ep":"CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "models" / "decoder" / "cpu" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");
  EXPECT_NE(ModelPackageContext::Open(dir.fs_path()), nullptr);
}

TEST(ModelPackageContextTest, ManifestRejectsUnsupportedSchemaVersionString) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":"2","components":["decoder"]})");
  EXPECT_THROW(ModelPackageContext::Open(dir.fs_path()), std::exception);
}

TEST(ModelPackageContextTest, ManifestRejectsMissingSchemaVersion) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"components":["decoder"]})");
  EXPECT_THROW(ModelPackageContext::Open(dir.fs_path()), std::exception);
}

TEST(ModelPackageContextTest, ManifestRejectsDuplicateComponents) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1,"components":["decoder","decoder"]})");
  EXPECT_THROW(ModelPackageContext::Open(dir.fs_path()), std::exception);
}

TEST(ModelPackageContextTest, ManifestRejectsPathTraversalInComponentName) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1,"components":["../escape"]})");
  EXPECT_THROW(ModelPackageContext::Open(dir.fs_path()), std::exception);
}

TEST(ModelPackageContextTest, ManifestListedComponentMissingMetadataThrows) {
  TempDir dir;
  // manifest declares `decoder`, but there is no `models/decoder/metadata.json`.
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1,"components":["decoder"]})");
  std::filesystem::create_directories(dir.path() / "models" / "decoder");
  EXPECT_THROW(ModelPackageContext::Open(dir.fs_path()), std::exception);
}

TEST(ModelPackageContextTest, ManifestListedMissingComponentDirectoryIsSkipped) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1,"components":["missing","decoder"]})");
  WriteFile(dir.path() / "models" / "decoder" / "metadata.json", R"({
    "variants":{"cpu":{"ep_compatibility":[{"ep":"CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "models" / "decoder" / "cpu" / "variant.json",
            R"({"files":[{"filename":"model.onnx"}]})");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  ASSERT_EQ(ctx->NumComponents(), 1u);
  EXPECT_EQ(ctx->ComponentName(0), "decoder");
}

TEST(ModelPackageContextTest, ManifestAbsentComponentsScansSubdirs) {
  // manifest present but has no `components` field → fall back to scanning.
  // Only children under models/ that contain metadata.json are treated as components.
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1})");
  std::filesystem::create_directories(dir.path() / "configs");
  std::filesystem::create_directories(dir.path() / "models" / ".hidden");
  std::filesystem::create_directories(dir.path() / "models" / "_reserved");
  WriteFile(dir.path() / "models" / "decoder" / "metadata.json", R"({
    "variants":{"cpu":{"ep_compatibility":[{"ep":"CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "models" / "decoder" / "cpu" / "variant.json",
            R"({"files":[{"filename":"model.onnx"}]})");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  EXPECT_EQ(ctx->NumComponents(), 1u);
  EXPECT_EQ(ctx->ComponentName(0), "decoder");
}

TEST(ModelPackageContextTest, ComponentNameInMetadataMustMatchDirectory) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1,"components":["decoder"]})");
  WriteFile(dir.path() / "models" / "decoder" / "metadata.json", R"({
    "component_name": "encoder",
    "variants":{"cpu":{"ep_compatibility":[{"ep":"CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "models" / "decoder" / "cpu" / "variant.json",
            R"({"files":[{"filename":"model.onnx"}]})");

  EXPECT_THROW(ModelPackageContext::Open(dir.fs_path()), std::exception);
}

// ============================================================================
// Variant traversal & parser order
// ============================================================================

TEST(ModelPackageContextTest, VariantTraversalReturnsAllVariants) {
  // ORT's current parser does not promise metadata declaration order.
  TempDir dir;
  BuildTwoComponentPackage(dir.path());
  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);

  EXPECT_EQ(ctx->NumVariants(0), 2u);
  std::vector<std::string> names{ctx->VariantName(0, 0), ctx->VariantName(0, 1)};
  std::sort(names.begin(), names.end());
  EXPECT_EQ(names[0], "cpu");
  EXPECT_EQ(names[1], "cuda");
}

TEST(ModelPackageContextTest, VariantTraversalDoesNotRequireDeclarationOrder) {
  // Author metadata.json with `zeta` before `alpha`; only membership matters
  // because ORT does not contractually expose declaration order.
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1,"components":["decoder"]})");
  WriteFile(dir.path() / "models" / "decoder" / "metadata.json", R"({
    "variants": {
      "zeta":  {"ep_compatibility": [{"ep": "CPUExecutionProvider"}]},
      "alpha": {"ep_compatibility": [{"ep": "CPUExecutionProvider"}]}
    }
  })");
  WriteFile(dir.path() / "models" / "decoder" / "zeta" / "variant.json", R"({"files":[{"filename":"m.onnx"}]})");
  WriteFile(dir.path() / "models" / "decoder" / "alpha" / "variant.json", R"({"files":[{"filename":"m.onnx"}]})");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  ASSERT_EQ(ctx->NumVariants(0), 2u);
  std::vector<std::string> names{ctx->VariantName(0, 0), ctx->VariantName(0, 1)};
  std::sort(names.begin(), names.end());
  EXPECT_EQ(names[0], "alpha");
  EXPECT_EQ(names[1], "zeta");
}

TEST(ModelPackageContextTest, VariantEpCompatibilityCarriesDeviceAndCompatString) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1,"components":["decoder"]})");
  WriteFile(dir.path() / "models" / "decoder" / "metadata.json", R"({
    "variants": {
      "openvino-npu": {
        "ep_compatibility": [
          {"ep": "OpenVINOExecutionProvider", "device": "NPU"}
        ]
      },
      "qnn": {
        "ep_compatibility": [
          {"ep": "QNNExecutionProvider", "compatibility_string": "soc_60|soc_69"}
        ]
      }
    }
  })");
  WriteFile(dir.path() / "models" / "decoder" / "openvino-npu" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");
  WriteFile(dir.path() / "models" / "decoder" / "qnn" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  ASSERT_EQ(ctx->NumVariants(0), 2u);

  const auto ov = ctx->VariantEpCompatibility(0, FindVariantIndex(*ctx, 0, "openvino-npu"));
  ASSERT_EQ(ov.size(), 1u);
  EXPECT_EQ(ov[0].ep, "OpenVINOExecutionProvider");
  ASSERT_TRUE(ov[0].device.has_value());
  EXPECT_EQ(*ov[0].device, "NPU");
  EXPECT_FALSE(ov[0].compatibility_string.has_value());

  const auto qnn = ctx->VariantEpCompatibility(0, FindVariantIndex(*ctx, 0, "qnn"));
  ASSERT_EQ(qnn.size(), 1u);
  EXPECT_EQ(qnn[0].ep, "QNNExecutionProvider");
  EXPECT_FALSE(qnn[0].device.has_value());
  ASSERT_TRUE(qnn[0].compatibility_string.has_value());
  EXPECT_EQ(*qnn[0].compatibility_string, "soc_60|soc_69");
}

// ============================================================================
// EP defaulting traversal helper
// ============================================================================

TEST(ModelPackageContextTest, EpsCompatibleWithUnionsAcrossVariants) {
  TempDir dir;
  BuildTwoComponentPackage(dir.path());
  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);

  // decoder: cpu (CPU) + cuda (CUDA) → {CPU, CUDA}; order is ORT-owned.
  const auto decoder_eps = ctx->EpsCompatibleWith(0);
  ASSERT_EQ(decoder_eps.size(), 2u);
  EXPECT_NE(std::find(decoder_eps.begin(), decoder_eps.end(), "CPUExecutionProvider"),
            decoder_eps.end());
  EXPECT_NE(std::find(decoder_eps.begin(), decoder_eps.end(), "CUDAExecutionProvider"),
            decoder_eps.end());

  // embedding: one variant declaring CPU + WebGpu → both, in declared order.
  const auto embedding_eps = ctx->EpsCompatibleWith(1);
  ASSERT_EQ(embedding_eps.size(), 2u);
  EXPECT_EQ(embedding_eps[0], "CPUExecutionProvider");
  EXPECT_EQ(embedding_eps[1], "WebGpuExecutionProvider");
}

TEST(ModelPackageContextTest, EpsCompatibleWithDedupesAcrossVariants) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1,"components":["decoder"]})");
  WriteFile(dir.path() / "models" / "decoder" / "metadata.json", R"({
    "variants": {
      "v1": {"ep_compatibility": [{"ep": "CPUExecutionProvider"}]},
      "v2": {"ep_compatibility": [{"ep": "CPUExecutionProvider"}]}
    }
  })");
  WriteFile(dir.path() / "models" / "decoder" / "v1" / "variant.json", R"({"files":[{"filename":"m.onnx"}]})");
  WriteFile(dir.path() / "models" / "decoder" / "v2" / "variant.json", R"({"files":[{"filename":"m.onnx"}]})");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  const auto eps = ctx->EpsCompatibleWith(0);
  ASSERT_EQ(eps.size(), 1u);
  EXPECT_EQ(eps[0], "CPUExecutionProvider");
}

// ============================================================================
// Variant selection
// ============================================================================

TEST(ModelPackageContextTest, SelectComponentReturnsNullptrForUnmatchedEp) {
  TempDir dir;
  BuildTwoComponentPackage(dir.path());
  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  EXPECT_THROW(ctx->SelectComponent(0, OnePriority("DmlExecutionProvider")), std::exception);
}

TEST(ModelPackageContextTest, SelectComponentPicksMatchingVariant) {
  if (!CanSelectEp("CUDAExecutionProvider") || !CanSelectEp("CPUExecutionProvider")) {
    GTEST_SKIP() << "CPU/CUDA EPs are not both available for ORT model-package selection in this test build";
  }

  TempDir dir;
  BuildTwoComponentPackage(dir.path());
  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);

  auto cuda = ctx->SelectComponent(0, OnePriority("CUDAExecutionProvider"));
  ASSERT_NE(cuda, nullptr);
  EXPECT_EQ(cuda->VariantFolderPath().string(),
            (dir.path() / "models" / "decoder" / "cuda").string());
  EXPECT_EQ(cuda->SelectedEp(), "CUDAExecutionProvider");

  auto cpu = ctx->SelectComponent(0, OnePriority("CPUExecutionProvider"));
  ASSERT_NE(cpu, nullptr);
  EXPECT_EQ(cpu->VariantFolderPath().string(),
            (dir.path() / "models" / "decoder" / "cpu").string());
  EXPECT_EQ(cpu->SelectedEp(), "CPUExecutionProvider");
}

TEST(ModelPackageContextTest, SelectComponentRespectsEpPriorityOrder) {
  if (!CanSelectEp("CUDAExecutionProvider") || !CanSelectEp("CPUExecutionProvider")) {
    GTEST_SKIP() << "CPU/CUDA EPs are not both available for ORT model-package selection in this test build";
  }

  // decoder has cpu and cuda variants. Pass priority [CUDA, CPU] — cuda wins;
  // pass priority [CPU, CUDA] — cpu wins. Proves that a variant matching a
  // higher-priority EP outranks a variant matching a lower one regardless of
  // parser order.
  TempDir dir;
  BuildTwoComponentPackage(dir.path());
  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);

  ModelPackageSelectionOptions cuda_first;
  cuda_first.ep_priority = {{"CUDAExecutionProvider", std::nullopt},
                            {"CPUExecutionProvider", std::nullopt}};
  auto a = ctx->SelectComponent(0, cuda_first);
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(a->VariantFolderPath().string(),
            (dir.path() / "models" / "decoder" / "cuda").string());

  ModelPackageSelectionOptions cpu_first;
  cpu_first.ep_priority = {{"CPUExecutionProvider", std::nullopt},
                           {"CUDAExecutionProvider", std::nullopt}};
  auto b = ctx->SelectComponent(0, cpu_first);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(b->VariantFolderPath().string(),
            (dir.path() / "models" / "decoder" / "cpu").string());
}

TEST(ModelPackageContextTest, SelectComponentUsesOnlyFirstPriorityEp) {
  TempDir dir;
  BuildTwoComponentPackage(dir.path());
  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);

  ModelPackageSelectionOptions dml_then_cpu;
  dml_then_cpu.ep_priority = {{"DmlExecutionProvider", std::nullopt},
                              {"CPUExecutionProvider", std::nullopt}};
  EXPECT_THROW(ctx->SelectComponent(0, dml_then_cpu), std::exception);
}

TEST(ModelPackageContextTest, SelectComponentEmptyPriorityReturnsNullptr) {
  TempDir dir;
  BuildTwoComponentPackage(dir.path());
  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);

  ModelPackageSelectionOptions empty;
  EXPECT_EQ(ctx->SelectComponent(0, empty), nullptr);
}

TEST(ModelPackageContextTest, SelectComponentDeviceAwareMatch) {
  SKIP_IF_CANNOT_SELECT_EP("OpenVINOExecutionProvider");

  // Component has two variants pinning OpenVINO/GPU and OpenVINO/NPU. The
  // device discriminator in the caller's priority entry selects the matching
  // variant. A device-less request still matches, like ORT does when no
  // hardware-device list is available.
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1,"components":["decoder"]})");
  WriteFile(dir.path() / "models" / "decoder" / "metadata.json", R"({
    "variants": {
      "openvino-gpu": {"ep_compatibility": [
        {"ep": "OpenVINOExecutionProvider", "device": "GPU"}]},
      "openvino-npu": {"ep_compatibility": [
        {"ep": "OpenVINOExecutionProvider", "device": "NPU"}]}
    }
  })");
  WriteFile(dir.path() / "models" / "decoder" / "openvino-gpu" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");
  WriteFile(dir.path() / "models" / "decoder" / "openvino-npu" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);

  auto gpu = ctx->SelectComponent(0, OnePriority("OpenVINOExecutionProvider", "GPU"));
  ASSERT_NE(gpu, nullptr);
  EXPECT_EQ(gpu->VariantFolderPath().string(),
            (dir.path() / "models" / "decoder" / "openvino-gpu").string());

  auto npu = ctx->SelectComponent(0, OnePriority("OpenVINOExecutionProvider", "NPU"));
  ASSERT_NE(npu, nullptr);
  EXPECT_EQ(npu->VariantFolderPath().string(),
            (dir.path() / "models" / "decoder" / "openvino-npu").string());

  auto bare = ctx->SelectComponent(0, OnePriority("OpenVINOExecutionProvider"));
  EXPECT_NE(bare, nullptr);
}

TEST(ModelPackageContextTest, SelectComponentTieBreaksByParserOrder) {
  SKIP_IF_CANNOT_SELECT_EP("CPUExecutionProvider");

  // Two variants both compatible with CPU. ORT uses parser order for ties.
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1,"components":["decoder"]})");
  WriteFile(dir.path() / "models" / "decoder" / "metadata.json", R"({
    "variants": {
      "second": {"ep_compatibility": [{"ep": "CPUExecutionProvider"}]},
      "first":  {"ep_compatibility": [{"ep": "CPUExecutionProvider"}]}
    }
  })");
  WriteFile(dir.path() / "models" / "decoder" / "second" / "variant.json", R"({"files":[{"filename":"m.onnx"}]})");
  WriteFile(dir.path() / "models" / "decoder" / "first" / "variant.json", R"({"files":[{"filename":"m.onnx"}]})");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  auto inst = ctx->SelectComponent(0, OnePriority("CPUExecutionProvider"));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->VariantFolderPath().string(),
            (dir.path() / "models" / "decoder" / "first").string());
}

// ============================================================================
// ConsumerMetadata
// ============================================================================

TEST(ModelPackageContextTest, ConsumerMetadataIsReturnedVerbatim) {
  SKIP_IF_CANNOT_SELECT_EP("CPUExecutionProvider");

  TempDir dir;
  BuildTwoComponentPackage(dir.path());
  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);

  auto cpu = ctx->SelectComponent(0, OnePriority("CPUExecutionProvider"));
  ASSERT_NE(cpu, nullptr);

  const std::string blob = cpu->ConsumerMetadata();
  ASSERT_FALSE(blob.empty());
  // Round-trip through DOM and assert structural equivalence — the
  // serializer canonicalises whitespace, so byte equality would be brittle.
  const auto parsed = JSON::ParseDocument(blob);
  ASSERT_TRUE(parsed.IsObject());
  EXPECT_DOUBLE_EQ(NumberAt(parsed, {"genai_config_overlay", "model", "context_length"}), 2048.0);
}

TEST(ModelPackageContextTest, ConsumerMetadataAbsentReturnsEmpty) {
  SKIP_IF_CANNOT_SELECT_EP("CPUExecutionProvider");

  TempDir dir;
  BuildTwoComponentPackage(dir.path());
  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  auto inst = ctx->SelectComponent(1, OnePriority("CPUExecutionProvider"));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->ConsumerMetadata(), "");
}

// ============================================================================
// Shared assets
// ============================================================================

TEST(ModelPackageContextTest, SharedAssetsPathPointsAtConfigsBucket) {
  TempDir dir;
  BuildTwoComponentPackage(dir.path());
  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  EXPECT_EQ(ctx->SharedAssetsPath().string(),
            (dir.path() / "configs").string());
}

// ============================================================================
// FileCount
// ============================================================================

TEST(ModelPackageContextTest, FileCountReflectsVariantJsonFilesArray) {
  SKIP_IF_CANNOT_SELECT_EP("QNNExecutionProvider");

  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1,"components":["decoder"]})");
  WriteFile(dir.path() / "models" / "decoder" / "metadata.json", R"({
    "variants": {"qnn": {"ep_compatibility":[{"ep":"QNNExecutionProvider"}]}}
  })");
  // Multi-file QNN-style variant.
  WriteFile(dir.path() / "models" / "decoder" / "qnn" / "variant.json", R"({
    "files": [
      {"filename": "embedding.onnx"},
      {"filename": "prompt.onnx"},
      {"filename": "token_gen.onnx"},
      {"filename": "lm_head.onnx"}
    ]
  })");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  auto inst = ctx->SelectComponent(0, OnePriority("QNNExecutionProvider"));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->FileCount(), 4u);
}

// ============================================================================
// ResolveSharedWeight
// ============================================================================

TEST(ModelPackageContextTest, ResolveSharedWeightReturnsBlobPath) {
  SKIP_IF_CANNOT_SELECT_EP("CPUExecutionProvider");

  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1,"components":["decoder"]})");
  WriteFile(dir.path() / "models" / "decoder" / "metadata.json", R"({
    "variants":{"cpu":{"ep_compatibility":[{"ep":"CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "models" / "decoder" / "cpu" / "variant.json", R"({
    "files":[{"filename":"m.onnx",
              "shared_files":{"m.onnx.data":"abc123"}}]
  })");
  // The blob name inside the checksum directory is producer-chosen.
  WriteFile(dir.path() / "models" / "decoder" / "shared_weights" / "abc123" / "weights.data",
            "fake-weights");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  auto inst = ctx->SelectComponent(0, OnePriority("CPUExecutionProvider"));
  ASSERT_NE(inst, nullptr);

  const auto resolved = inst->ResolveSharedWeight("abc123");
  EXPECT_EQ(resolved.string(),
            (dir.path() / "models" / "decoder" / "shared_weights" / "abc123" / "weights.data").string());
}

TEST(ModelPackageContextTest, ResolveSharedWeightThrowsOnMissingChecksum) {
  SKIP_IF_CANNOT_SELECT_EP("CPUExecutionProvider");

  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1,"components":["decoder"]})");
  WriteFile(dir.path() / "models" / "decoder" / "metadata.json", R"({
    "variants":{"cpu":{"ep_compatibility":[{"ep":"CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "models" / "decoder" / "cpu" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  auto inst = ctx->SelectComponent(0, OnePriority("CPUExecutionProvider"));
  ASSERT_NE(inst, nullptr);
  EXPECT_THROW(inst->ResolveSharedWeight("doesnotexist"), std::exception);
}

TEST(ModelPackageContextTest, ResolveSharedWeightThrowsOnEmptyDirectory) {
  SKIP_IF_CANNOT_SELECT_EP("CPUExecutionProvider");

  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1,"components":["decoder"]})");
  WriteFile(dir.path() / "models" / "decoder" / "metadata.json", R"({
    "variants":{"cpu":{"ep_compatibility":[{"ep":"CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "models" / "decoder" / "cpu" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");
  std::filesystem::create_directories(dir.path() / "models" / "decoder" / "shared_weights" / "empty");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  auto inst = ctx->SelectComponent(0, OnePriority("CPUExecutionProvider"));
  ASSERT_NE(inst, nullptr);
  EXPECT_THROW(inst->ResolveSharedWeight("empty"), std::exception);
}

TEST(ModelPackageContextTest, ResolveSharedWeightThrowsOnMultipleBlobs) {
  SKIP_IF_CANNOT_SELECT_EP("CPUExecutionProvider");

  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1,"components":["decoder"]})");
  WriteFile(dir.path() / "models" / "decoder" / "metadata.json", R"({
    "variants":{"cpu":{"ep_compatibility":[{"ep":"CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "models" / "decoder" / "cpu" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");
  WriteFile(dir.path() / "models" / "decoder" / "shared_weights" / "abc" / "a.bin", "1");
  WriteFile(dir.path() / "models" / "decoder" / "shared_weights" / "abc" / "b.bin", "2");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  auto inst = ctx->SelectComponent(0, OnePriority("CPUExecutionProvider"));
  ASSERT_NE(inst, nullptr);
  EXPECT_THROW(inst->ResolveSharedWeight("abc"), std::exception);
}

TEST(ModelPackageContextTest, ResolveSharedWeightRejectsPathTraversalChecksum) {
  SKIP_IF_CANNOT_SELECT_EP("CPUExecutionProvider");

  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1,"components":["decoder"]})");
  WriteFile(dir.path() / "models" / "decoder" / "metadata.json", R"({
    "variants":{"cpu":{"ep_compatibility":[{"ep":"CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "models" / "decoder" / "cpu" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  auto inst = ctx->SelectComponent(0, OnePriority("CPUExecutionProvider"));
  ASSERT_NE(inst, nullptr);
  EXPECT_THROW(inst->ResolveSharedWeight("../escape"), std::exception);
}

#endif  // ORT_API_VERSION < 27

// ============================================================================
// ParseVariantManifest helper (the pipeline runner uses this)
// ============================================================================

TEST(ParseVariantManifestTest, ParsesFilesArrayInOrder) {
  TempDir dir;
  // Author the files in a specific order; we expect that order to round-trip.
  WriteFile(dir.path() / "variant.json", R"({
    "files": [
      {"filename": "stage1.onnx"},
      {"filename": "stage2.onnx"},
      {"filename": "stage3.onnx"}
    ]
  })");

  const auto vm = ParseVariantManifest(fs::path(dir.path().string()));
  ASSERT_EQ(vm.files.size(), 3u);
  EXPECT_EQ(vm.files[0].filename, "stage1.onnx");
  EXPECT_EQ(vm.files[1].filename, "stage2.onnx");
  EXPECT_EQ(vm.files[2].filename, "stage3.onnx");
}

TEST(ParseVariantManifestTest, ParsesPerFileOptionsAsScalarStrings) {
  TempDir dir;
  WriteFile(dir.path() / "variant.json", R"({
    "files": [{
      "filename": "m.onnx",
      "session_options": {
        "intra_op_num_threads": 4,
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "use_deterministic_compute": true
      },
      "provider_options": {
        "htp_performance_mode": "burst"
      }
    }]
  })");

  const auto vm = ParseVariantManifest(fs::path(dir.path().string()));
  ASSERT_EQ(vm.files.size(), 1u);
  const auto& f = vm.files[0];

  auto so_threads = f.session_options.find("intra_op_num_threads");
  ASSERT_NE(so_threads, f.session_options.end());
  EXPECT_EQ(so_threads->second, "4");

  auto so_opt = f.session_options.find("graph_optimization_level");
  ASSERT_NE(so_opt, f.session_options.end());
  EXPECT_EQ(so_opt->second, "ORT_ENABLE_ALL");

  auto so_det = f.session_options.find("use_deterministic_compute");
  ASSERT_NE(so_det, f.session_options.end());
  EXPECT_EQ(so_det->second, "true");

  auto po_perf = f.provider_options.find("htp_performance_mode");
  ASSERT_NE(po_perf, f.provider_options.end());
  EXPECT_EQ(po_perf->second, "burst");
}

TEST(ParseVariantManifestTest, ParsesSharedFilesMap) {
  TempDir dir;
  WriteFile(dir.path() / "variant.json", R"({
    "files": [{
      "filename": "m.onnx",
      "shared_files": {
        "m.onnx.data": "abc123",
        "extra.bin": "def456"
      }
    }]
  })");

  const auto vm = ParseVariantManifest(fs::path(dir.path().string()));
  ASSERT_EQ(vm.files.size(), 1u);
  const auto& sf = vm.files[0].shared_files;
  ASSERT_EQ(sf.size(), 2u);
  EXPECT_EQ(sf.at("m.onnx.data"), "abc123");
  EXPECT_EQ(sf.at("extra.bin"), "def456");
}

TEST(ParseVariantManifestTest, RejectsMissingFiles) {
  TempDir dir;
  WriteFile(dir.path() / "variant.json", R"({"consumer_metadata":{}})");
  EXPECT_THROW(ParseVariantManifest(fs::path(dir.path().string())), std::exception);
}

TEST(ParseVariantManifestTest, RejectsEmptyFiles) {
  TempDir dir;
  WriteFile(dir.path() / "variant.json", R"({"files":[]})");
  EXPECT_THROW(ParseVariantManifest(fs::path(dir.path().string())), std::exception);
}

TEST(ParseVariantManifestTest, RejectsNestedSessionOptions) {
  TempDir dir;
  WriteFile(dir.path() / "variant.json", R"({
    "files":[{"filename":"m.onnx","session_options":{"nested":{"x":1}}}]
  })");
  EXPECT_THROW(ParseVariantManifest(fs::path(dir.path().string())), std::exception);
}

TEST(ParseVariantManifestTest, RejectsPathTraversalInFilename) {
  TempDir dir;
  WriteFile(dir.path() / "variant.json", R"({"files":[{"filename":"../escape.onnx"}]})");
  EXPECT_THROW(ParseVariantManifest(fs::path(dir.path().string())), std::exception);
}

TEST(ParseVariantManifestTest, RejectsPathTraversalInSharedFilesChecksum) {
  TempDir dir;
  WriteFile(dir.path() / "variant.json", R"({
    "files":[{"filename":"m.onnx","shared_files":{"x.bin":"../escape"}}]
  })");
  EXPECT_THROW(ParseVariantManifest(fs::path(dir.path().string())), std::exception);
}

TEST(ParseVariantManifestTest, RejectsAbsentVariantJson) {
  TempDir dir;  // empty
  EXPECT_THROW(ParseVariantManifest(fs::path(dir.path().string())), std::exception);
}

}  // namespace Generators::test
