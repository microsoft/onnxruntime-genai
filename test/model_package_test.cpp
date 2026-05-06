// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for the stub directory-walker implementation of
// `Generators::ModelPackageContext` and the `ParseVariantManifest` helper.
// The stub is a complete drop-in for the abstract surface and is used both
// directly here and as the placeholder backend until ORT v4 lands.
//
// Each test materialises a synthetic v4 package layout under
// `std::filesystem::temp_directory_path() / <unique>` and exercises one slice
// of the surface. The fixture cleans up after itself.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <string_view>

#include "json.h"
#include "models/model_package.h"

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

void WriteFile(const std::filesystem::path& p, std::string_view contents) {
  std::filesystem::create_directories(p.parent_path());
  std::ofstream out(p, std::ios::binary);
  ASSERT_TRUE(out.is_open()) << "cannot open " << p;
  out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
}

// Materialise a representative two-component package:
//   <root>/manifest.json                   {"schema_version":1,"components":["decoder","embedding"]}
//   <root>/configs/                        (empty: not exercised here directly)
//   <root>/decoder/metadata.json           cpu, cuda variants (in this order)
//   <root>/decoder/cpu/variant.json        single file, overlay sets context_length=2048
//   <root>/decoder/cuda/variant.json       single file, overlay sets context_length=4096
//   <root>/embedding/metadata.json         single variant cpu, EP compat = [CPU, WebGPU]
//   <root>/embedding/cpu/variant.json      single file, no consumer_metadata
void BuildTwoComponentPackage(const std::filesystem::path& root) {
  WriteFile(root / "manifest.json", R"({
    "schema_version": 1,
    "components": ["decoder", "embedding"]
  })");

  std::filesystem::create_directories(root / "configs");

  WriteFile(root / "decoder" / "metadata.json", R"({
    "variants": {
      "cpu": {
        "ep_compatibility": [{"ep": "CPUExecutionProvider"}]
      },
      "cuda": {
        "ep_compatibility": [
          {"ep": "CUDAExecutionProvider", "compatibility": ["sm_80", "sm_90"]}
        ]
      }
    }
  })");

  WriteFile(root / "decoder" / "cpu" / "variant.json", R"({
    "files": [{"filename": "model.onnx"}],
    "consumer_metadata": {
      "genai_config_overlay": {"model": {"context_length": 2048}}
    }
  })");

  WriteFile(root / "decoder" / "cuda" / "variant.json", R"({
    "files": [{"filename": "model.onnx"}],
    "consumer_metadata": {
      "genai_config_overlay": {"model": {"context_length": 4096}}
    }
  })");

  WriteFile(root / "embedding" / "metadata.json", R"({
    "variants": {
      "cpu": {
        "ep_compatibility": [
          {"ep": "CPUExecutionProvider"},
          {"ep": "WebGpuExecutionProvider"}
        ]
      }
    }
  })");

  WriteFile(root / "embedding" / "cpu" / "variant.json", R"({
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

TEST(ModelPackageContextTest, NoManifestButComponentMetadataIsRecognized) {
  // Spec says manifest.json is optional. A package with no manifest but a
  // component subdirectory containing metadata.json + variant directory
  // must be recognized.
  TempDir dir;
  WriteFile(dir.path() / "decoder" / "metadata.json", R"({
    "variants": {"cpu": {"ep_compatibility": [{"ep": "CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "decoder" / "cpu" / "variant.json",
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
  // producer is expected to align with the spec rather than us forking
  // the parser to track every variant of the shape.
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({
    "components": [{"name": "decoder", "metadata": "decoder/metadata.json"}]
  })");
  EXPECT_THROW(ModelPackageContext::Open(dir.fs_path()), std::exception);
}

TEST(ModelPackageContextTest, ManifestUnsupportedSchemaVersionThrows) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version": 99, "components":["x"]})");
  EXPECT_THROW(ModelPackageContext::Open(dir.fs_path()), std::exception);
}

TEST(ModelPackageContextTest, ManifestAcceptsStringSchemaVersion) {
  // Spec calls for a string `schema_version`; "1" and "1.0" both name v1.
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":"1.0","components":["decoder"]})");
  WriteFile(dir.path() / "decoder" / "metadata.json", R"({
    "variants": {"cpu": {"ep_compatibility":[{"ep":"CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "decoder" / "cpu" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");
  EXPECT_NE(ModelPackageContext::Open(dir.fs_path()), nullptr);
}

TEST(ModelPackageContextTest, ManifestAcceptsNumberSchemaVersion) {
  // Numeric `1` is coerced to "1" so historic producers still load.
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1,"components":["decoder"]})");
  WriteFile(dir.path() / "decoder" / "metadata.json", R"({
    "variants": {"cpu": {"ep_compatibility":[{"ep":"CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "decoder" / "cpu" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");
  EXPECT_NE(ModelPackageContext::Open(dir.fs_path()), nullptr);
}

TEST(ModelPackageContextTest, ManifestRejectsUnsupportedSchemaVersionString) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":"2","components":["decoder"]})");
  EXPECT_THROW(ModelPackageContext::Open(dir.fs_path()), std::exception);
}

TEST(ModelPackageContextTest, ManifestRejectsDuplicateComponents) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"components":["decoder","decoder"]})");
  EXPECT_THROW(ModelPackageContext::Open(dir.fs_path()), std::exception);
}

TEST(ModelPackageContextTest, ManifestRejectsPathTraversalInComponentName) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"components":["../escape"]})");
  EXPECT_THROW(ModelPackageContext::Open(dir.fs_path()), std::exception);
}

TEST(ModelPackageContextTest, ManifestListedComponentMissingMetadataThrows) {
  TempDir dir;
  // manifest declares `decoder`, but there is no `decoder/metadata.json`.
  WriteFile(dir.path() / "manifest.json", R"({"components":["decoder"]})");
  std::filesystem::create_directories(dir.path() / "decoder");
  EXPECT_THROW(ModelPackageContext::Open(dir.fs_path()), std::exception);
}

TEST(ModelPackageContextTest, ManifestAbsentComponentsScansSubdirs) {
  // manifest present but has no `components` field → fall back to scanning.
  // configs/ must be excluded; hidden dirs (.foo) and underscore-prefixed
  // (_foo) must also be excluded.
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1})");
  std::filesystem::create_directories(dir.path() / "configs");
  std::filesystem::create_directories(dir.path() / ".hidden");
  std::filesystem::create_directories(dir.path() / "_reserved");
  WriteFile(dir.path() / "decoder" / "metadata.json", R"({
    "variants":{"cpu":{"ep_compatibility":[{"ep":"CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "decoder" / "cpu" / "variant.json",
            R"({"files":[{"filename":"model.onnx"}]})");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  EXPECT_EQ(ctx->NumComponents(), 1u);
  EXPECT_EQ(ctx->ComponentName(0), "decoder");
}

// ============================================================================
// Variant traversal & metadata.json declaration order
// ============================================================================

TEST(ModelPackageContextTest, VariantTraversalPreservesMetadataDeclarationOrder) {
  // Even if the on-disk filesystem order would differ (here it's identical
  // alphabetically: cpu < cuda), what matters is that metadata.json is the
  // authoritative source — declaration order is what the EP-preference
  // tie-break leans on. We assert the order matches what we wrote.
  TempDir dir;
  BuildTwoComponentPackage(dir.path());
  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);

  EXPECT_EQ(ctx->NumVariants(0), 2u);
  EXPECT_EQ(ctx->VariantName(0, 0), "cpu");
  EXPECT_EQ(ctx->VariantName(0, 1), "cuda");
}

TEST(ModelPackageContextTest, VariantOrderCanInvertFilesystemAlpha) {
  // Author the metadata.json with `zeta` BEFORE `alpha` — filesystem alpha
  // sort would put alpha first, but metadata.json wins.
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"components":["decoder"]})");
  WriteFile(dir.path() / "decoder" / "metadata.json", R"({
    "variants": {
      "zeta":  {"ep_compatibility": [{"ep": "CPUExecutionProvider"}]},
      "alpha": {"ep_compatibility": [{"ep": "CPUExecutionProvider"}]}
    }
  })");
  WriteFile(dir.path() / "decoder" / "zeta" / "variant.json", R"({"files":[{"filename":"m.onnx"}]})");
  WriteFile(dir.path() / "decoder" / "alpha" / "variant.json", R"({"files":[{"filename":"m.onnx"}]})");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  ASSERT_EQ(ctx->NumVariants(0), 2u);
  EXPECT_EQ(ctx->VariantName(0, 0), "zeta");
  EXPECT_EQ(ctx->VariantName(0, 1), "alpha");
}

TEST(ModelPackageContextTest, VariantEpCompatibilityCarriesDeviceAndCompatStrings) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"components":["decoder"]})");
  WriteFile(dir.path() / "decoder" / "metadata.json", R"({
    "variants": {
      "openvino-npu": {
        "ep_compatibility": [
          {"ep": "OpenVINOExecutionProvider", "device": "NPU"}
        ]
      },
      "qnn": {
        "ep_compatibility": [
          {"ep": "QNNExecutionProvider", "compatibility": ["soc_60", "soc_69"]}
        ]
      }
    }
  })");
  WriteFile(dir.path() / "decoder" / "openvino-npu" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");
  WriteFile(dir.path() / "decoder" / "qnn" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  ASSERT_EQ(ctx->NumVariants(0), 2u);

  const auto ov = ctx->VariantEpCompatibility(0, 0);
  ASSERT_EQ(ov.size(), 1u);
  EXPECT_EQ(ov[0].ep, "OpenVINOExecutionProvider");
  ASSERT_TRUE(ov[0].device.has_value());
  EXPECT_EQ(*ov[0].device, "NPU");
  EXPECT_TRUE(ov[0].compatibility.empty());

  const auto qnn = ctx->VariantEpCompatibility(0, 1);
  ASSERT_EQ(qnn.size(), 1u);
  EXPECT_EQ(qnn[0].ep, "QNNExecutionProvider");
  EXPECT_FALSE(qnn[0].device.has_value());
  ASSERT_EQ(qnn[0].compatibility.size(), 2u);
  EXPECT_EQ(qnn[0].compatibility[0], "soc_60");
  EXPECT_EQ(qnn[0].compatibility[1], "soc_69");
}

// ============================================================================
// EP defaulting traversal helper
// ============================================================================

TEST(ModelPackageContextTest, EpsCompatibleWithUnionsAcrossVariants) {
  TempDir dir;
  BuildTwoComponentPackage(dir.path());
  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);

  // decoder: cpu (CPU) + cuda (CUDA) → {CPU, CUDA} in first-seen order.
  const auto decoder_eps = ctx->EpsCompatibleWith(0);
  ASSERT_EQ(decoder_eps.size(), 2u);
  EXPECT_EQ(decoder_eps[0], "CPUExecutionProvider");
  EXPECT_EQ(decoder_eps[1], "CUDAExecutionProvider");

  // embedding: one variant declaring CPU + WebGpu → both, in declared order.
  const auto embedding_eps = ctx->EpsCompatibleWith(1);
  ASSERT_EQ(embedding_eps.size(), 2u);
  EXPECT_EQ(embedding_eps[0], "CPUExecutionProvider");
  EXPECT_EQ(embedding_eps[1], "WebGpuExecutionProvider");
}

TEST(ModelPackageContextTest, EpsCompatibleWithDedupesAcrossVariants) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"components":["decoder"]})");
  WriteFile(dir.path() / "decoder" / "metadata.json", R"({
    "variants": {
      "v1": {"ep_compatibility": [{"ep": "CPUExecutionProvider"}]},
      "v2": {"ep_compatibility": [{"ep": "CPUExecutionProvider"}]}
    }
  })");
  WriteFile(dir.path() / "decoder" / "v1" / "variant.json", R"({"files":[{"filename":"m.onnx"}]})");
  WriteFile(dir.path() / "decoder" / "v2" / "variant.json", R"({"files":[{"filename":"m.onnx"}]})");

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
  EXPECT_EQ(ctx->SelectComponent(0, OnePriority("DmlExecutionProvider")), nullptr);
}

TEST(ModelPackageContextTest, SelectComponentPicksMatchingVariant) {
  TempDir dir;
  BuildTwoComponentPackage(dir.path());
  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);

  auto cuda = ctx->SelectComponent(0, OnePriority("CUDAExecutionProvider"));
  ASSERT_NE(cuda, nullptr);
  EXPECT_EQ(cuda->VariantFolderPath().string(),
            (dir.path() / "decoder" / "cuda").string());
  EXPECT_EQ(cuda->SelectedEp(), "CUDAExecutionProvider");

  auto cpu = ctx->SelectComponent(0, OnePriority("CPUExecutionProvider"));
  ASSERT_NE(cpu, nullptr);
  EXPECT_EQ(cpu->VariantFolderPath().string(),
            (dir.path() / "decoder" / "cpu").string());
  EXPECT_EQ(cpu->SelectedEp(), "CPUExecutionProvider");
}

TEST(ModelPackageContextTest, SelectComponentRespectsEpPriorityOrder) {
  // decoder has cpu and cuda variants. Pass priority [CUDA, CPU] — cuda wins;
  // pass priority [CPU, CUDA] — cpu wins. Proves that a variant matching a
  // higher-priority EP outranks a variant matching a lower one regardless of
  // metadata declaration order.
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
            (dir.path() / "decoder" / "cuda").string());

  ModelPackageSelectionOptions cpu_first;
  cpu_first.ep_priority = {{"CPUExecutionProvider", std::nullopt},
                           {"CUDAExecutionProvider", std::nullopt}};
  auto b = ctx->SelectComponent(0, cpu_first);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(b->VariantFolderPath().string(),
            (dir.path() / "decoder" / "cpu").string());
}

TEST(ModelPackageContextTest, SelectComponentEmptyPriorityDefaultsToCpu) {
  TempDir dir;
  BuildTwoComponentPackage(dir.path());
  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);

  // Spec: empty captured EP list is treated as [CPUExecutionProvider].
  ModelPackageSelectionOptions empty;
  auto inst = ctx->SelectComponent(0, empty);
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->VariantFolderPath().string(),
            (dir.path() / "decoder" / "cpu").string());
}

TEST(ModelPackageContextTest, SelectComponentDeviceAwareMatch) {
  // Component has two variants pinning OpenVINO/GPU and OpenVINO/NPU. The
  // device discriminator in the caller's priority entry must select the
  // correct variant; passing OpenVINO without a device must NOT match a
  // device-pinned variant.
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"components":["decoder"]})");
  WriteFile(dir.path() / "decoder" / "metadata.json", R"({
    "variants": {
      "openvino-gpu": {"ep_compatibility": [
        {"ep": "OpenVINOExecutionProvider", "device": "GPU"}]},
      "openvino-npu": {"ep_compatibility": [
        {"ep": "OpenVINOExecutionProvider", "device": "NPU"}]}
    }
  })");
  WriteFile(dir.path() / "decoder" / "openvino-gpu" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");
  WriteFile(dir.path() / "decoder" / "openvino-npu" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);

  auto gpu = ctx->SelectComponent(0, OnePriority("OpenVINOExecutionProvider", "GPU"));
  ASSERT_NE(gpu, nullptr);
  EXPECT_EQ(gpu->VariantFolderPath().string(),
            (dir.path() / "decoder" / "openvino-gpu").string());

  auto npu = ctx->SelectComponent(0, OnePriority("OpenVINOExecutionProvider", "NPU"));
  ASSERT_NE(npu, nullptr);
  EXPECT_EQ(npu->VariantFolderPath().string(),
            (dir.path() / "decoder" / "openvino-npu").string());

  // Device-less OpenVINO request must not silently pick a device-pinned
  // variant — selecting an NPU build for a caller who didn't ask for NPU
  // would be unsafe.
  auto bare = ctx->SelectComponent(0, OnePriority("OpenVINOExecutionProvider"));
  EXPECT_EQ(bare, nullptr);
}

TEST(ModelPackageContextTest, SelectComponentTieBreaksByMetadataOrder) {
  // Two variants both compatible with CPU. Metadata declares `second`
  // before `first` — a CPU priority must select `second`.
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"components":["decoder"]})");
  WriteFile(dir.path() / "decoder" / "metadata.json", R"({
    "variants": {
      "second": {"ep_compatibility": [{"ep": "CPUExecutionProvider"}]},
      "first":  {"ep_compatibility": [{"ep": "CPUExecutionProvider"}]}
    }
  })");
  WriteFile(dir.path() / "decoder" / "second" / "variant.json", R"({"files":[{"filename":"m.onnx"}]})");
  WriteFile(dir.path() / "decoder" / "first" / "variant.json", R"({"files":[{"filename":"m.onnx"}]})");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  auto inst = ctx->SelectComponent(0, OnePriority("CPUExecutionProvider"));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->VariantFolderPath().string(),
            (dir.path() / "decoder" / "second").string());
}

// ============================================================================
// ConsumerMetadata
// ============================================================================

TEST(ModelPackageContextTest, ConsumerMetadataIsReturnedVerbatim) {
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
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"components":["decoder"]})");
  WriteFile(dir.path() / "decoder" / "metadata.json", R"({
    "variants": {"qnn": {"ep_compatibility":[{"ep":"QNNExecutionProvider"}]}}
  })");
  // Multi-file QNN-style variant.
  WriteFile(dir.path() / "decoder" / "qnn" / "variant.json", R"({
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
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"components":["decoder"]})");
  WriteFile(dir.path() / "decoder" / "metadata.json", R"({
    "variants":{"cpu":{"ep_compatibility":[{"ep":"CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "decoder" / "cpu" / "variant.json", R"({
    "files":[{"filename":"m.onnx",
              "shared_files":{"m.onnx.data":"abc123"}}]
  })");
  // The blob name inside the checksum directory is producer-chosen.
  WriteFile(dir.path() / "decoder" / "shared_weights" / "abc123" / "weights.data",
            "fake-weights");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  auto inst = ctx->SelectComponent(0, OnePriority("CPUExecutionProvider"));
  ASSERT_NE(inst, nullptr);

  const auto resolved = inst->ResolveSharedWeight("abc123");
  EXPECT_EQ(resolved.string(),
            (dir.path() / "decoder" / "shared_weights" / "abc123" / "weights.data").string());
}

TEST(ModelPackageContextTest, ResolveSharedWeightThrowsOnMissingChecksum) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"components":["decoder"]})");
  WriteFile(dir.path() / "decoder" / "metadata.json", R"({
    "variants":{"cpu":{"ep_compatibility":[{"ep":"CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "decoder" / "cpu" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  auto inst = ctx->SelectComponent(0, OnePriority("CPUExecutionProvider"));
  ASSERT_NE(inst, nullptr);
  EXPECT_THROW(inst->ResolveSharedWeight("doesnotexist"), std::exception);
}

TEST(ModelPackageContextTest, ResolveSharedWeightThrowsOnEmptyDirectory) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"components":["decoder"]})");
  WriteFile(dir.path() / "decoder" / "metadata.json", R"({
    "variants":{"cpu":{"ep_compatibility":[{"ep":"CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "decoder" / "cpu" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");
  std::filesystem::create_directories(dir.path() / "decoder" / "shared_weights" / "empty");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  auto inst = ctx->SelectComponent(0, OnePriority("CPUExecutionProvider"));
  ASSERT_NE(inst, nullptr);
  EXPECT_THROW(inst->ResolveSharedWeight("empty"), std::exception);
}

TEST(ModelPackageContextTest, ResolveSharedWeightThrowsOnMultipleBlobs) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"components":["decoder"]})");
  WriteFile(dir.path() / "decoder" / "metadata.json", R"({
    "variants":{"cpu":{"ep_compatibility":[{"ep":"CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "decoder" / "cpu" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");
  WriteFile(dir.path() / "decoder" / "shared_weights" / "abc" / "a.bin", "1");
  WriteFile(dir.path() / "decoder" / "shared_weights" / "abc" / "b.bin", "2");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  auto inst = ctx->SelectComponent(0, OnePriority("CPUExecutionProvider"));
  ASSERT_NE(inst, nullptr);
  EXPECT_THROW(inst->ResolveSharedWeight("abc"), std::exception);
}

TEST(ModelPackageContextTest, ResolveSharedWeightRejectsPathTraversalChecksum) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"components":["decoder"]})");
  WriteFile(dir.path() / "decoder" / "metadata.json", R"({
    "variants":{"cpu":{"ep_compatibility":[{"ep":"CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "decoder" / "cpu" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");

  auto ctx = ModelPackageContext::Open(dir.fs_path());
  ASSERT_NE(ctx, nullptr);
  auto inst = ctx->SelectComponent(0, OnePriority("CPUExecutionProvider"));
  ASSERT_NE(inst, nullptr);
  EXPECT_THROW(inst->ResolveSharedWeight("../escape"), std::exception);
}

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

TEST(ParseVariantManifestTest, ParsesPerFileSessionOptionsAsTypedJson) {
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

  // Number value preserved as JSON double.
  auto so_threads = f.session_options.find("intra_op_num_threads");
  ASSERT_NE(so_threads, f.session_options.end());
  ASSERT_TRUE(so_threads->second.IsNumber());
  EXPECT_DOUBLE_EQ(so_threads->second.AsNumber(), 4.0);

  // String value preserved as JSON string.
  auto so_opt = f.session_options.find("graph_optimization_level");
  ASSERT_NE(so_opt, f.session_options.end());
  ASSERT_TRUE(so_opt->second.IsString());
  EXPECT_EQ(so_opt->second.AsString(), "ORT_ENABLE_ALL");

  // Boolean value preserved as JSON bool.
  auto so_det = f.session_options.find("use_deterministic_compute");
  ASSERT_NE(so_det, f.session_options.end());
  ASSERT_TRUE(so_det->second.IsBool());
  EXPECT_TRUE(so_det->second.AsBool());

  auto po_perf = f.provider_options.find("htp_performance_mode");
  ASSERT_NE(po_perf, f.provider_options.end());
  ASSERT_TRUE(po_perf->second.IsString());
  EXPECT_EQ(po_perf->second.AsString(), "burst");
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
