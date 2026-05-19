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
// the package-load split.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <string_view>

#include "generators.h"
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

#if ORT_API_VERSION >= 27

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

  WriteFile(root / "models" / "decoder" / "metadata.json", R"({
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
  WriteFile(root / "models" / "decoder" / "cpu" / "variant.json", variant_json);

  // Empty placeholder so file presence tests aren't surprised.
  WriteFile(root / "models" / "decoder" / "cpu" / "model.onnx", "");
}

ModelPackageSelectionOptions OnePriority(std::string ep) {
  ModelPackageSelectionOptions opts;
  opts.ep_priority.push_back({std::move(ep), std::nullopt});
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

#define SKIP_IF_CANNOT_SELECT_EP(ep)          \
  if (!CanSelectEp(ep)) {                     \
    GTEST_SKIP() << ep << " is not available for ORT model-package selection in this test build"; \
  }

#endif  // ORT_API_VERSION >= 27

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

#if ORT_API_VERSION < 27

TEST(ConfigPackageTest, PackageMarkerRequiresOrtModelPackageApi) {
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({"schema_version":1,"components":["decoder"]})");
  EXPECT_THROW(Config(dir.fs_path(), ""), std::exception);
}

#else

TEST(ConfigPackageTest, PackageDetectionPopulatesPackageState) {
  SKIP_IF_CANNOT_SELECT_EP("CPUExecutionProvider");

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
  SKIP_IF_CANNOT_SELECT_EP("CPUExecutionProvider");

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
  SKIP_IF_CANNOT_SELECT_EP("CPUExecutionProvider");

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
  SKIP_IF_CANNOT_SELECT_EP("CPUExecutionProvider");

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
  SKIP_IF_CANNOT_SELECT_EP("CPUExecutionProvider");

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
  WriteFile(dir.path() / "models" / "decoder" / "metadata.json", R"({
    "variants": {
      "cpu":  {"ep_compatibility":[{"ep":"CPUExecutionProvider"}]},
      "cuda": {"ep_compatibility":[{"ep":"CUDAExecutionProvider"}]}
    }
  })");
  WriteFile(dir.path() / "models" / "decoder" / "cpu" / "variant.json", R"({"files":[{"filename":"m.onnx"}]})");
  WriteFile(dir.path() / "models" / "decoder" / "cuda" / "variant.json", R"({"files":[{"filename":"m.onnx"}]})");
  WriteFile(dir.path() / "models" / "embedding" / "metadata.json", R"({
    "variants": {
      "cpu":  {"ep_compatibility":[{"ep":"CPUExecutionProvider"}]},
      "cuda": {"ep_compatibility":[{"ep":"CUDAExecutionProvider"}]}
    }
  })");
  WriteFile(dir.path() / "models" / "embedding" / "cpu" / "variant.json", R"({"files":[{"filename":"e.onnx"}]})");
  WriteFile(dir.path() / "models" / "embedding" / "cuda" / "variant.json", R"({"files":[{"filename":"e.onnx"}]})");

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
  WriteFile(dir.path() / "models" / "decoder" / "metadata.json", R"({
    "variants": {"cuda": {"ep_compatibility":[{"ep":"CUDAExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "models" / "decoder" / "cuda" / "variant.json",
            R"({"files":[{"filename":"m.onnx"}]})");
  WriteFile(dir.path() / "models" / "embedding" / "metadata.json", R"({
    "variants": {"cpu": {"ep_compatibility":[{"ep":"CPUExecutionProvider"}]}}
  })");
  WriteFile(dir.path() / "models" / "embedding" / "cpu" / "variant.json",
            R"({"files":[{"filename":"e.onnx"}]})");

  EXPECT_THROW(Config(dir.fs_path(), ""), std::exception);
}

TEST(ConfigPackageTest, SingleEpIntersectionAutoSelects) {
  SKIP_IF_CANNOT_SELECT_EP("CPUExecutionProvider");

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
  SKIP_IF_CANNOT_SELECT_EP("CPUExecutionProvider");

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
    WriteFile(dir.path() / "models" / cname / "metadata.json", R"({
      "variants": {"cpu": {"ep_compatibility":[{"ep":"CPUExecutionProvider"}]}}
    })");
    WriteFile(dir.path() / "models" / cname / "cpu" / "variant.json",
              R"({"files":[{"filename":"m.onnx"}]})");
  }

  // Decoder-only carries the overlay (a single role's overlay can populate
  // the whole config; this is just the path of least JSON resistance).
  WriteFile(dir.path() / "models" / "decoder" / "cpu" / "variant.json", R"({
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

// ============================================================================
// Public optional `ep` argument
// ============================================================================

TEST(ConfigPackageTest, UserEpBypassesDefaultingInPackage) {
  if (!CanSelectEp("CUDAExecutionProvider") || !CanSelectEp("CPUExecutionProvider")) {
    GTEST_SKIP() << "CPU/CUDA EPs are not both available for ORT model-package selection in this test build";
  }

  // Two components that both ship CPU and CUDA variants. Defaulting would
  // throw (multi-EP intersection), but a user-supplied `ep` resolves the
  // ambiguity.
  TempDir dir;
  WriteFile(dir.path() / "manifest.json", R"({
    "schema_version": 1,
    "components": ["decoder", "embedding"]
  })");
  WriteFile(dir.path() / "configs" / "genai_config.json", kBaseGenaiConfig);
  for (const auto& cname : {"decoder", "embedding"}) {
    WriteFile(dir.path() / "models" / cname / "metadata.json", R"({
      "variants": {
        "cpu":  {"ep_compatibility":[{"ep":"CPUExecutionProvider"}]},
        "cuda": {"ep_compatibility":[{"ep":"CUDAExecutionProvider"}]}
      }
    })");
    WriteFile(dir.path() / "models" / cname / "cpu" / "variant.json", R"({"files":[{"filename":"m.onnx"}]})");
    WriteFile(dir.path() / "models" / cname / "cuda" / "variant.json", R"({"files":[{"filename":"m.onnx"}]})");
  }

  // Without `ep`, defaulting throws on the multi-EP intersection.
  EXPECT_THROW(Config(dir.fs_path(), ""), std::exception);

  // With `ep="CPUExecutionProvider"`, the user's choice is captured and
  // SelectComponent succeeds for every component.
  EXPECT_NO_THROW(Config(dir.fs_path(), "", "CPUExecutionProvider"));

  // With `ep="CUDAExecutionProvider"`, likewise the CUDA variants win.
  EXPECT_NO_THROW(Config(dir.fs_path(), "", "CUDAExecutionProvider"));
}

TEST(ConfigPackageTest, UserEpThatNoComponentSupportsThrowsWithDiagnostic) {
  SKIP_IF_CANNOT_SELECT_EP("CUDAExecutionProvider");

  // Single-component CPU-only package; user requests CUDA.
  // SelectComponent finds no matching variant and the diagnostic must
  // list the component's compatible EPs.
  TempDir dir;
  BuildSingleComponentPackage(dir.path());
  try {
    Config config{dir.fs_path(), "", "CUDAExecutionProvider"};
    FAIL() << "Expected unsupported-EP throw";
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("CUDAExecutionProvider"), std::string::npos) << msg;
    EXPECT_NE(msg.find("CPUExecutionProvider"), std::string::npos) << msg
                                                                   << " (compatible-EPs hint missing)";
    EXPECT_NE(msg.find("decoder"), std::string::npos) << msg
                                                      << " (component name missing)";
  }
}

TEST(ConfigPackageTest, EmptyUserEpUsesDefaulting) {
  SKIP_IF_CANNOT_SELECT_EP("CPUExecutionProvider");

  // Empty ep should be equivalent to omitting it (defaulting runs).
  TempDir dir;
  BuildSingleComponentPackage(dir.path());
  EXPECT_NO_THROW(Config(dir.fs_path(), "", ""));
  EXPECT_NO_THROW(Config(dir.fs_path(), ""));
}

#endif  // ORT_API_VERSION < 27

TEST(ConfigPackageTest, UserEpInFlatDirThrows) {
  // Flat-dir mode (no manifest.json). A non-empty `ep` raises a clear error
  // pointing at the OgaConfigClearProviders / OgaConfigAppendProvider channel.
  TempDir dir;
  WriteFile(dir.path() / "genai_config.json", kBaseGenaiConfig);

  // Sanity: empty ep still loads.
  EXPECT_NO_THROW(Config(dir.fs_path(), ""));

  try {
    Config config{dir.fs_path(), "", "CUDAExecutionProvider"};
    FAIL() << "Expected flat-dir + ep throw";
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("v4 model package"), std::string::npos) << msg;
    EXPECT_NE(msg.find("OgaConfigAppendProvider"), std::string::npos) << msg;
  }
}

}  // namespace Generators::test
