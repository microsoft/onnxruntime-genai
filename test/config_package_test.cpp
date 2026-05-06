// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for the v4-model-package code path through `Generators::Config`:
//  * detection (vs. flat-dir)
//  * `Config::shared_assets_path` is set to <package>/configs/
//  * per-component `consumer_metadata.genai_config_overlay` is RFC-7386 merged
//    over the package-shipped `configs/genai_config.json` base
//  * EP defaulting (intersection across components, fail-with-diagnostic)
//  * caller json_overlay is still applied as the final layer (layer-2)
//  * `model.<role>.component` references are validated
//  * package state is stashed on Config (`model_package`, `component_instances`)
//
// The flat-dir path is exercised by `test/c_api_tests.cpp` and the existing
// model tests; we add one regression case here to assert it still works after
// the W3 split.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <string_view>

#include "generators.h"

namespace Generators::test {

namespace {

class TempDir {
 public:
  TempDir() {
    std::random_device rd;
    std::uniform_int_distribution<uint64_t> dist;
    path_ = std::filesystem::temp_directory_path() /
            ("genai-config-package-test-" + std::to_string(dist(rd)));
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

// Minimal valid genai_config.json the strict streaming parser will accept.
// The tests that need fields beyond `model.type`/`model.context_length`/
// `model.decoder.*`/`search.*` will overlay them on top.
constexpr std::string_view kBaseGenaiConfig = R"({
  "model": {
    "type": "phi3",
    "context_length": 1024,
    "vocab_size": 32000,
    "decoder": {
      "session_options": {
        "log_id": "base",
        "provider_options": []
      },
      "filename": "model.onnx",
      "head_size": 96,
      "hidden_size": 3072,
      "num_attention_heads": 32,
      "num_hidden_layers": 32,
      "num_key_value_heads": 32,
      "inputs": {
        "input_ids": "input_ids",
        "attention_mask": "attention_mask",
        "position_ids": "position_ids",
        "past_key_names": "past_key_values.%d.key",
        "past_value_names": "past_key_values.%d.value"
      },
      "outputs": {
        "logits": "logits",
        "present_key_names": "present.%d.key",
        "present_value_names": "present.%d.value"
      }
    }
  },
  "search": {
    "max_length": 2048,
    "min_length": 0,
    "do_sample": false,
    "num_beams": 1
  }
})";

// Build a single-component (decoder only) CPU-only package under `root`.
// `decoder_overlay` is JSON spliced into `consumer_metadata.genai_config_overlay`
// — pass an empty string for "no overlay".
void BuildSingleComponentPackage(const std::filesystem::path& root,
                                 std::string_view decoder_overlay = "") {
  WriteFile(root / "manifest.json", R"({
    "schema_version": 1,
    "components": ["decoder"]
  })");

  WriteFile(root / "configs" / "genai_config.json", kBaseGenaiConfig);

  WriteFile(root / "decoder" / "metadata.json", R"({
    "variants": {
      "cpu": { "ep_compatibility": [{"ep": "CPUExecutionProvider"}] }
    }
  })");

  std::string variant_json;
  variant_json += R"({"files":[{"filename":"model.onnx"}])";
  if (!decoder_overlay.empty()) {
    variant_json += R"(,"consumer_metadata":{"genai_config_overlay":)";
    variant_json += std::string(decoder_overlay);
    variant_json += "}";
  }
  variant_json += "}";
  WriteFile(root / "decoder" / "cpu" / "variant.json", variant_json);

  // Empty placeholder so file presence tests aren't surprised.
  WriteFile(root / "decoder" / "cpu" / "model.onnx", "");
}

}  // namespace

// ============================================================================
// Detection
// ============================================================================

TEST(ConfigPackageTest, FlatDirRegressionStillLoads) {
  // A flat-dir model with only `genai_config.json` must NOT be detected as a
  // package and the existing strict-parser path must produce a valid Config.
  TempDir dir;
  WriteFile(dir.path() / "genai_config.json", std::string(kBaseGenaiConfig));

  Config config{dir.fs_path(), ""};
  EXPECT_EQ(config.config_path.string(), dir.fs_path().string());
  EXPECT_EQ(config.shared_assets_path.string(), dir.fs_path().string());
  EXPECT_EQ(config.model_package, nullptr);
  EXPECT_TRUE(config.component_instances.empty());
  EXPECT_EQ(config.model.context_length, 1024);
  EXPECT_EQ(config.model.type, "phi3");
}

TEST(ConfigPackageTest, PackageDetectionPopulatesPackageState) {
  TempDir dir;
  BuildSingleComponentPackage(dir.path());

  Config config{dir.fs_path(), ""};
  ASSERT_NE(config.model_package, nullptr);
  EXPECT_EQ(config.shared_assets_path.string(),
            (dir.path() / "configs").string());
  ASSERT_EQ(config.component_instances.size(), 1u);
  ASSERT_EQ(config.component_instances.count("decoder"), 1u);
  EXPECT_NE(config.component_instances.at("decoder"), nullptr);
  // config_path retains the package root for legacy callers.
  EXPECT_EQ(config.config_path.string(), dir.fs_path().string());
}

// ============================================================================
// Per-component overlay merge
// ============================================================================

TEST(ConfigPackageTest, OverlayMergesIntoBase) {
  // Decoder variant ships an overlay that bumps context_length and tags the
  // component name into model.decoder.component. Both fields must be reflected
  // in the resulting Config (proves the merge ran AND the streaming parser
  // recognises the new `component` key).
  TempDir dir;
  BuildSingleComponentPackage(dir.path(), R"({
    "model": {
      "context_length": 4096,
      "decoder": { "component": "decoder", "head_size": 128 }
    }
  })");

  Config config{dir.fs_path(), ""};
  EXPECT_EQ(config.model.context_length, 4096);
  EXPECT_EQ(config.model.decoder.head_size, 128);
  EXPECT_EQ(config.model.decoder.component, "decoder");
}

TEST(ConfigPackageTest, OverlayWithoutConsumerMetadataIsNoop) {
  // When variant.json has no consumer_metadata at all, the base config must
  // load unmodified.
  TempDir dir;
  BuildSingleComponentPackage(dir.path(), "");

  Config config{dir.fs_path(), ""};
  EXPECT_EQ(config.model.context_length, 1024);
  EXPECT_EQ(config.model.decoder.head_size, 96);
}

TEST(ConfigPackageTest, NonObjectGenaiConfigOverlayThrows) {
  // Spec: `genai_config_overlay` MUST be a JSON object (the merge target is
  // the genai_config.json root, also an object). Strings/arrays/etc must be
  // rejected with a clear message.
  TempDir dir;
  BuildSingleComponentPackage(dir.path(), R"("not-an-object")");

  EXPECT_THROW(Config(dir.fs_path(), ""), std::exception);
}

// ============================================================================
// Caller (layer-2) overlay still applies last
// ============================================================================

TEST(ConfigPackageTest, CallerJsonOverlayWinsOverPackageOverlay) {
  // Package overlay sets head_size=128. Caller overlay (RuntimeSettings /
  // OgaConfigOverlay channel) sets head_size=144 — caller wins.
  TempDir dir;
  BuildSingleComponentPackage(dir.path(), R"({
    "model": {
      "decoder": { "component": "decoder", "head_size": 128 }
    }
  })");

  constexpr std::string_view caller_overlay =
      R"({"model":{"decoder":{"head_size":144}}})";
  Config config{dir.fs_path(), caller_overlay};

  EXPECT_EQ(config.model.decoder.head_size, 144);
  // Other overlay-supplied fields survive.
  EXPECT_EQ(config.model.decoder.component, "decoder");
}

// ============================================================================
// Role -> component reference validation
// ============================================================================

TEST(ConfigPackageTest, UnknownRoleComponentReferenceThrows) {
  // Overlay tags decoder.component="ghost" but no component named ghost
  // exists. ValidateRoleComponentReferences must reject.
  TempDir dir;
  BuildSingleComponentPackage(dir.path(), R"({
    "model": { "decoder": { "component": "ghost" } }
  })");

  EXPECT_THROW(Config(dir.fs_path(), ""), std::exception);
}

TEST(ConfigPackageTest, EmptyRoleComponentIsAllowed) {
  // A role without a `component` field must NOT trigger the validator —
  // only non-empty references are checked.
  TempDir dir;
  BuildSingleComponentPackage(dir.path(), R"({
    "model": { "decoder": { "head_size": 96 } }
  })");

  EXPECT_NO_THROW(Config(dir.fs_path(), ""));
}

// ============================================================================
// EP defaulting
// ============================================================================

TEST(ConfigPackageTest, MultiEpIntersectionFailsWithDiagnostic) {
  // Two components: decoder (CPU + CUDA) and embedding (CPU + CUDA). The
  // intersection has two EPs; defaulting must throw.
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({
    "schema_version": 1,
    "components": ["decoder", "embedding"]
  })");
  WriteFile(dir.path() / "configs" / "genai_config.json", kBaseGenaiConfig);
  WriteFile(dir.path() / "decoder" / "metadata.json", R"({
    "variants": {
      "cpu":  {"ep_compatibility":[{"ep":"CPUExecutionProvider"}]},
      "cuda": {"ep_compatibility":[{"ep":"CUDAExecutionProvider"}]}
    }
  })");
  WriteFile(dir.path() / "decoder" / "cpu"  / "variant.json", R"({"files":[{"filename":"m.onnx"}]})");
  WriteFile(dir.path() / "decoder" / "cuda" / "variant.json", R"({"files":[{"filename":"m.onnx"}]})");
  WriteFile(dir.path() / "embedding" / "metadata.json", R"({
    "variants": {
      "cpu":  {"ep_compatibility":[{"ep":"CPUExecutionProvider"}]},
      "cuda": {"ep_compatibility":[{"ep":"CUDAExecutionProvider"}]}
    }
  })");
  WriteFile(dir.path() / "embedding" / "cpu"  / "variant.json", R"({"files":[{"filename":"e.onnx"}]})");
  WriteFile(dir.path() / "embedding" / "cuda" / "variant.json", R"({"files":[{"filename":"e.onnx"}]})");

  try {
    Config config{dir.fs_path(), ""};
    FAIL() << "Expected ambiguous-EP throw";
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    // The diagnostic must be helpful — mention both candidate EPs.
    EXPECT_NE(msg.find("CPUExecutionProvider"), std::string::npos) << msg;
    EXPECT_NE(msg.find("CUDAExecutionProvider"), std::string::npos) << msg;
  }
}

TEST(ConfigPackageTest, EmptyEpIntersectionFailsWithDiagnostic) {
  // decoder supports only CUDA; embedding supports only CPU. Intersection
  // is empty.
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({
    "schema_version": 1,
    "components": ["decoder", "embedding"]
  })");
  WriteFile(dir.path() / "configs" / "genai_config.json", kBaseGenaiConfig);
  WriteFile(dir.path() / "decoder" / "metadata.json", R"({
    "variants": {"cuda": {"ep_compatibility":[{"ep":"CUDAExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "decoder" / "cuda" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");
  WriteFile(dir.path() / "embedding" / "metadata.json", R"({
    "variants": {"cpu": {"ep_compatibility":[{"ep":"CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "embedding" / "cpu" / "variant.json",
            R"({"files":[{"filename":"e.onnx"}]})");

  EXPECT_THROW(Config(dir.fs_path(), ""), std::exception);
}

TEST(ConfigPackageTest, SingleEpIntersectionAutoSelects) {
  // decoder + embedding both support CPU; only one component is wired but
  // the cross-component intersection is the well-defined set {CPU}, so
  // defaulting must succeed silently.
  TempDir dir;
  BuildSingleComponentPackage(dir.path());

  EXPECT_NO_THROW(Config(dir.fs_path(), ""));
}

// ============================================================================
// Component field parsing for all 5 mapped roles
// ============================================================================

TEST(ConfigPackageTest, AllFiveRoleComponentFieldsAreParseable) {
  // Author a degenerate package whose overlay sets every role's `component`
  // field. The role-validation pass must accept it (each name is wired in
  // the manifest), and the field must land on the corresponding sub-struct.
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({
    "schema_version": 1,
    "components": ["decoder", "encoder", "vision", "speech", "embedding"]
  })");
  WriteFile(dir.path() / "configs" / "genai_config.json", kBaseGenaiConfig);

  // Five mapped components, all CPU-only.
  for (const auto& cname : {"decoder", "encoder", "vision", "speech", "embedding"}) {
    WriteFile(dir.path() / cname / "metadata.json", R"({
      "variants": {"cpu": {"ep_compatibility":[{"ep":"CPUExecutionProvider"}]}}
    })");
    WriteFile(dir.path() / cname / "cpu" / "variant.json",
              R"({"files":[{"filename":"m.onnx"}]})");
  }

  // Decoder-only carries the overlay (a single role's overlay can populate
  // the whole config; this is just the path of least JSON resistance).
  WriteFile(dir.path() / "decoder" / "cpu" / "variant.json", R"({
    "files":[{"filename":"m.onnx"}],
    "consumer_metadata": {
      "genai_config_overlay": {
        "model": {
          "decoder":   { "component": "decoder" },
          "encoder":   { "component": "encoder" },
          "vision":    { "component": "vision" },
          "speech":    { "component": "speech" },
          "embedding": { "component": "embedding" }
        }
      }
    }
  })");

  Config config{dir.fs_path(), ""};
  EXPECT_EQ(config.model.decoder.component, "decoder");
  EXPECT_EQ(config.model.encoder.component, "encoder");
  EXPECT_EQ(config.model.vision.component, "vision");
  EXPECT_EQ(config.model.speech.component, "speech");
  EXPECT_EQ(config.model.embedding.component, "embedding");
}

}  // namespace Generators::test
