// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ort_genai.h"
// Included only for the compile-time ORT_GENAI_HAS_MODEL_PACKAGE gate; the tests below
// drive the feature exclusively through the public C++ API (ort_genai.h).
#include "models/model_package.h"

#if ORT_GENAI_HAS_MODEL_PACKAGE

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

std::string ReadFile(const fs_std::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// A variant of the package's single "model" component. With `valid_config` the variant gets
// a loadable genai_config.json so selecting it succeeds; without one, selecting it fails.
struct VariantSpec {
  std::string name;
  std::string ep;
  bool valid_config = true;
};

// Builds a minimal model package the ORT model_package API can open: a single "model"
// component declared inline in the manifest, whose variants each get a placeholder
// model.onnx and, optionally, a config. The component is inline so the variant directories
// live at the package root with no models/ nesting.
fs_std::path WritePackage(const std::string& suffix, const std::vector<VariantSpec>& variants) {
  const auto root = MakeTempDir(suffix);

  // The single "model" component is declared inline in the manifest, so its variant
  // directories sit at the package root: variant_directory defaults to the variant name
  // relative to the component directory, which is the package root for an inline component.
  // No models/ nesting and no separate component.json — everything stays at the top level.
  std::string manifest =
      "{\n  \"schema_version\": \"1.0\",\n  \"components\": {\n    \"model\": {\n      \"variants\": {\n";
  for (size_t i = 0; i < variants.size(); ++i) {
    manifest += "        \"" + variants[i].name + "\": { \"ep\": \"" + variants[i].ep + "\" }";
    manifest += (i + 1 == variants.size()) ? "\n" : ",\n";
  }
  manifest += "      }\n    }\n  }\n}\n";
  WriteFile(root / "manifest.json", manifest);

  for (const auto& variant : variants) {
    const auto variant_dir = root / variant.name;
    WriteFile(variant_dir / "model.onnx", "placeholder");
    if (variant.valid_config) {
      WriteFile(variant_dir / "genai_config.json",
                "{ \"model\": { \"type\": \"tiny-test-model\","
                " \"vocab_size\": 16, \"context_length\": 32 }, \"search\": {} }");
    }
  }
  return root;
}

// Runs `fn`, expecting it to throw, and returns the exception message. Returns an empty
// string when nothing was thrown.
template <typename Fn>
std::string CaptureThrowMessage(Fn&& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    return e.what();
  }
  return {};
}

// Produces an ONNX model whose initializers live in a sidecar external-data file, by loading
// `src_onnx` with optimizations disabled and exporting an optimized copy with external
// initializers. Writes <out_dir>/model.onnx plus <out_dir>/model.onnx.data and returns the
// model path. Used to exercise the external-initializers folder session option.
fs_std::path ExportModelWithExternalData(const fs_std::path& src_onnx, const fs_std::path& out_dir) {
  // The low-level ORT wrappers below run in this test binary, which has its own copy of the
  // Ort::api pointer (the genai .so initializes its own). Initialize ours before using them.
  Ort::InitApi();
  fs_std::create_directories(out_dir);
  const auto model_path = out_dir / "model.onnx";
  auto session_options = OrtSessionOptions::Create();
  session_options->SetGraphOptimizationLevel(ORT_DISABLE_ALL);
  session_options->SetOptimizedModelFilePath(model_path.c_str());
  session_options->AddConfigEntry("session.optimized_model_external_initializers_file_name",
                                  "model.onnx.data");
  session_options->AddConfigEntry("session.optimized_model_external_initializers_min_size_in_bytes",
                                  "0");
  // Creating the session writes the optimized model + external data as a side effect.
  auto env = OrtEnv::Create();
  OrtSession::Create(*env, src_onnx.c_str(), session_options.get());
  return model_path;
}

}  // namespace

TEST(ModelPackage, RejectsFlatDirectory) {
  const auto root = MakeTempDir("flat");
  WriteFile(root / "genai_config.json", "{}");
  WriteFile(root / "model.onnx", "placeholder");

  const std::string message = CaptureThrowMessage(
      [&] { OgaConfig::CreateFromPackageEp(root.string().c_str(), "cpu"); });
  EXPECT_NE(message.find("is not a model package"), std::string::npos) << message;
}

TEST(ModelPackage, FlatDirectoryWithManifestIsNotAPackage) {
  // A flat model directory may carry its own unrelated manifest.json. The root genai_config.json
  // marks it as flat, so it must not be treated as a model package.
  const auto root = MakeTempDir("flat_with_manifest");
  WriteFile(root / "genai_config.json",
            "{ \"model\": { \"type\": \"tiny-test-model\","
            " \"vocab_size\": 16, \"context_length\": 32 }, \"search\": {} }");
  WriteFile(root / "model.onnx", "placeholder");
  WriteFile(root / "manifest.json", "{ \"unrelated\": true }");

  // Passing an ep treats the path as a package; it must be rejected as a flat directory instead.
  const std::string message = CaptureThrowMessage(
      [&] { OgaConfig::CreateFromPackageEp(root.string().c_str(), "cpu"); });
  EXPECT_NE(message.find("is not a model package"), std::string::npos) << message;

  // Loading it as a flat directory (no ep) succeeds despite the manifest.json.
  EXPECT_NO_THROW(OgaConfig::Create(root.string().c_str()));
}

TEST(ModelPackage, RejectsMissingPath) {
  EXPECT_THROW(OgaConfig::CreateFromPackageEp("/this/path/does/not/exist/12345", "cpu"),
               std::runtime_error);
}

TEST(ModelPackage, RejectsRegularFile) {
  const auto root = MakeTempDir("file");
  const auto file_path = root / "thing.json";
  WriteFile(file_path, "{}");
  EXPECT_THROW(OgaConfig::CreateFromPackageEp(file_path.string().c_str(), "cpu"),
               std::runtime_error);
}

TEST(ModelPackage, AutoDetectsSingleVariant) {
  const auto root = WritePackage("autocpu", {{"cpu", "CPUExecutionProvider"}});
  // No ep argument: the single declared ep is auto-detected and the config loads.
  EXPECT_NO_THROW(OgaConfig::Create(root.string().c_str()));
}

TEST(ModelPackage, AmbiguousEpsWithoutEpThrow) {
  const auto root = WritePackage(
      "ambig", {{"cpu", "CPUExecutionProvider"}, {"cuda", "CUDAExecutionProvider"}});
  const std::string message =
      CaptureThrowMessage([&] { OgaConfig::Create(root.string().c_str()); });
  EXPECT_NE(message.find("execution provider"), std::string::npos) << message;
}

TEST(ModelPackage, ExplicitEpSelectsNamedVariant) {
  // Selecting "cpu" routes to the cpu variant (which has a config) and succeeds. Selecting
  // "cuda" must not silently succeed: the cuda variant has no config (and CUDA may be absent).
  const auto root = WritePackage(
      "explicit",
      {{"cpu", "CPUExecutionProvider", true}, {"cuda", "CUDAExecutionProvider", false}});
  EXPECT_NO_THROW(OgaConfig::CreateFromPackageEp(root.string().c_str(), "cpu"));
  EXPECT_THROW(OgaConfig::CreateFromPackageEp(root.string().c_str(), "cuda"),
               std::runtime_error);
}

TEST(ModelPackage, AcceptsFullEpName) {
  const auto root = WritePackage(
      "fullname", {{"cpu", "CPUExecutionProvider"}, {"cuda", "CUDAExecutionProvider"}});
  EXPECT_NO_THROW(OgaConfig::CreateFromPackageEp(root.string().c_str(), "CPUExecutionProvider"));
}

TEST(ModelPackage, TokenizerResolvesThroughSharedAsset) {
  // End-to-end: a real model lives in the cpu variant while its tokenizer lives in a
  // content-addressed shared asset, referenced via a "sha256:" tokenizer_dir. ORT discovers
  // the shared_assets/sha256-<hex>/ directory at load time; loading the model and tokenizing
  // exercises sha256: resolution through the public API.
  const fs_std::path src_model = fs_std::path(MODEL_PATH) / "hf-internal-testing" /
                                 "tiny-random-gpt2-fp32";

  const auto root = MakeTempDir("e2e_pkg");
  const auto variant_dir = root / "cpu";
  // A valid sha256 URI is "sha256:" + 64 lowercase hex chars; the on-disk asset directory is
  // shared_assets/sha256-<hex>/. The bytes need not match the digest: ORT discovers the
  // directory by name, it does not re-hash the contents at load time.
  const std::string digest(64, 'a');
  const auto asset_dir = root / "shared_assets" / ("sha256-" + digest);
  fs_std::create_directories(variant_dir);
  fs_std::create_directories(asset_dir);

  // Inline single "model" component: the cpu variant directory sits at the package root.
  WriteFile(root / "manifest.json",
            "{ \"schema_version\": \"1.0\", \"components\": { \"model\": { \"variants\":"
            " { \"cpu\": { \"ep\": \"CPUExecutionProvider\" } } } } }");

  // The model file stays beside genai_config.json in the variant directory.
  std::error_code ec;
  fs_std::copy_file(src_model / "past.onnx", variant_dir / "past.onnx",
                    fs_std::copy_options::overwrite_existing, ec);
  ASSERT_FALSE(ec) << ec.message();

  // The tokenizer files move to the content-addressed shared asset directory.
  for (const auto& entry : fs_std::directory_iterator(src_model)) {
    const auto name = entry.path().filename().string();
    if (name == "past.onnx" || name == "genai_config.json") continue;
    fs_std::copy_file(entry.path(), asset_dir / name,
                      fs_std::copy_options::overwrite_existing, ec);
    ASSERT_FALSE(ec) << ec.message();
  }

  // Point tokenizer_dir at the shared asset using the "sha256:" scheme.
  std::string config = ReadFile(src_model / "genai_config.json");
  const std::string anchor = "\"type\": \"gpt2\",";
  const auto pos = config.find(anchor);
  ASSERT_NE(pos, std::string::npos);
  config.insert(pos + anchor.size(), "\n    \"tokenizer_dir\": \"sha256:" + digest + "\",");
  WriteFile(variant_dir / "genai_config.json", config);

  auto oga_config = OgaConfig::CreateFromPackageEp(root.string().c_str(), "cpu");
  auto model = OgaModel::Create(*oga_config);
  auto tokenizer = OgaTokenizer::Create(*model);

  auto sequences = OgaSequences::Create();
  tokenizer->Encode("Hello model package", *sequences);
  EXPECT_EQ(sequences->Count(), 1u);
  EXPECT_GT(sequences->SequenceCount(0), 0u);
}

TEST(ModelPackage, ExternalInitializersFolderResolvesThroughSharedAsset) {
  // A variant's model keeps its weights in a sidecar external-data file that lives in a
  // content-addressed shared asset, referenced from the genai_config session options via
  // "session.model_external_initializers_file_folder_path": "sha256:<hex>". The model loads only
  // if genai resolves that folder through the package (relative/sha256: -> absolute) and hands it
  // to ORT; otherwise ORT looks beside the model file and the weights are missing.
  const fs_std::path src_model = fs_std::path(MODEL_PATH) / "hf-internal-testing" /
                                 "tiny-random-gpt2-fp32";

  const auto root = MakeTempDir("e2e_extdata");
  const auto variant_dir = root / "cpu";
  const std::string digest(64, 'b');
  const auto asset_dir = root / "shared_assets" / ("sha256-" + digest);
  fs_std::create_directories(variant_dir);
  fs_std::create_directories(asset_dir);

  WriteFile(root / "manifest.json",
            "{ \"schema_version\": \"1.0\", \"components\": { \"model\": { \"variants\":"
            " { \"cpu\": { \"ep\": \"CPUExecutionProvider\" } } } } }");

  // Build an external-data model in the variant dir, then move only its weights blob into the
  // shared asset so it can be found solely through the resolved folder option.
  ExportModelWithExternalData(src_model / "past.onnx", variant_dir);
  std::error_code ec;
  fs_std::rename(variant_dir / "model.onnx.data", asset_dir / "model.onnx.data", ec);
  ASSERT_FALSE(ec) << ec.message();

  // Tokenizer files sit beside the model so the genai model directory is complete.
  for (const auto& entry : fs_std::directory_iterator(src_model)) {
    const auto name = entry.path().filename().string();
    if (name == "past.onnx" || name == "genai_config.json") continue;
    fs_std::copy_file(entry.path(), variant_dir / name,
                      fs_std::copy_options::overwrite_existing, ec);
    ASSERT_FALSE(ec) << ec.message();
  }

  // Author a config pointing the decoder at the external-data model, optionally referencing the
  // shared-asset weights folder by sha256: URI.
  auto write_config = [&](bool with_folder_option) {
    std::string config = ReadFile(src_model / "genai_config.json");
    const auto file_pos = config.find("past.onnx");
    ASSERT_NE(file_pos, std::string::npos);
    config.replace(file_pos, std::string("past.onnx").size(), "model.onnx");
    if (with_folder_option) {
      const std::string anchor = "\"session_options\": {";
      const auto pos = config.find(anchor);
      ASSERT_NE(pos, std::string::npos);
      config.insert(pos + anchor.size(),
                    "\n        \"session.model_external_initializers_file_folder_path\": \"sha256:" +
                        digest + "\",");
    }
    WriteFile(variant_dir / "genai_config.json", config);
  };

  // With the folder option, the weights resolve through the shared asset and the model loads.
  write_config(/*with_folder_option=*/true);
  EXPECT_NO_THROW({
    auto oga_config = OgaConfig::CreateFromPackageEp(root.string().c_str(), "cpu");
    auto model = OgaModel::Create(*oga_config);
  });

  // Without it, ORT looks beside model.onnx, the weights blob is absent, and loading fails.
  write_config(/*with_folder_option=*/false);
  const std::string message = CaptureThrowMessage([&] {
    auto oga_config = OgaConfig::CreateFromPackageEp(root.string().c_str(), "cpu");
    auto model = OgaModel::Create(*oga_config);
  });
  EXPECT_FALSE(message.empty());
}

TEST(ModelPackage, FlatDirectoryRejectsSha256Reference) {
  // sha256: references only resolve inside a package. In a plain directory, resolving one must
  // fail with a clear error instead of silently producing a bogus "<dir>/sha256:<hex>" path.
  const fs_std::path src_model = fs_std::path(MODEL_PATH) / "hf-internal-testing" /
                                 "tiny-random-gpt2-fp32";
  const auto dir = MakeTempDir("flat_sha256");
  std::error_code ec;
  for (const auto& entry : fs_std::directory_iterator(src_model)) {
    fs_std::copy_file(entry.path(), dir / entry.path().filename().string(),
                      fs_std::copy_options::overwrite_existing, ec);
    ASSERT_FALSE(ec) << ec.message();
  }

  std::string config = ReadFile(src_model / "genai_config.json");
  const std::string anchor = "\"type\": \"gpt2\",";
  const auto pos = config.find(anchor);
  ASSERT_NE(pos, std::string::npos);
  config.insert(pos + anchor.size(),
                "\n    \"tokenizer_dir\": \"sha256:" + std::string(64, 'a') + "\",");
  WriteFile(dir / "genai_config.json", config);

  const std::string message = CaptureThrowMessage([&] {
    auto model = OgaModel::Create(dir.string().c_str());
    auto tokenizer = OgaTokenizer::Create(*model);
  });
  EXPECT_NE(message.find("sha256:"), std::string::npos) << message;
}

#endif  // ORT_GENAI_HAS_MODEL_PACKAGE
