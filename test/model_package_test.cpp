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
// component whose variants each get a placeholder model.onnx and, optionally, a config.
fs_std::path WritePackage(const std::string& suffix, const std::vector<VariantSpec>& variants) {
  const auto root = MakeTempDir(suffix);

  WriteFile(root / "manifest.json",
            "{ \"schema_version\": 1, \"components\": [\"model\"] }");

  std::string metadata = "{\n  \"component_name\": \"model\",\n  \"variants\": {\n";
  for (size_t i = 0; i < variants.size(); ++i) {
    metadata += "    \"" + variants[i].name + "\": { \"ep\": \"" + variants[i].ep + "\" }";
    metadata += (i + 1 == variants.size()) ? "\n" : ",\n";
  }
  metadata += "  }\n}\n";
  WriteFile(root / "models" / "model" / "metadata.json", metadata);

  for (const auto& variant : variants) {
    const auto variant_dir = root / "models" / "model" / variant.name;
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

}  // namespace

TEST(ModelPackage, RejectsFlatDirectory) {
  const auto root = MakeTempDir("flat");
  WriteFile(root / "genai_config.json", "{}");
  WriteFile(root / "model.onnx", "placeholder");

  const std::string message = CaptureThrowMessage(
      [&] { OgaConfig::CreateFromPackageEp(root.string().c_str(), "cpu"); });
  EXPECT_NE(message.find("is not a model package"), std::string::npos) << message;
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

TEST(ModelPackage, TokenizerResolvesThroughPackageRoot) {
  // End-to-end: a real model lives in the cpu variant while its tokenizer lives in the
  // package's shared/ directory, referenced via a "package:" tokenizer_dir. Loading the
  // model and tokenizing exercises package_root resolution through the public API.
  const fs_std::path src_model = fs_std::path(MODEL_PATH) / "hf-internal-testing" /
                                 "tiny-random-gpt2-fp32";

  const auto root = MakeTempDir("e2e_pkg");
  const auto variant_dir = root / "models" / "model" / "cpu";
  fs_std::create_directories(variant_dir);
  fs_std::create_directories(root / "shared");

  WriteFile(root / "manifest.json",
            "{ \"schema_version\": 1, \"components\": [\"model\"] }");
  WriteFile(root / "models" / "model" / "metadata.json",
            "{ \"component_name\": \"model\","
            " \"variants\": { \"cpu\": { \"ep\": \"CPUExecutionProvider\" } } }");

  // The model file stays beside genai_config.json in the variant directory.
  std::error_code ec;
  fs_std::copy_file(src_model / "past.onnx", variant_dir / "past.onnx",
                    fs_std::copy_options::overwrite_existing, ec);
  ASSERT_FALSE(ec) << ec.message();

  // The tokenizer files move to the package-level shared/ directory.
  for (const auto& entry : fs_std::directory_iterator(src_model)) {
    const auto name = entry.path().filename().string();
    if (name == "past.onnx" || name == "genai_config.json") continue;
    fs_std::copy_file(entry.path(), root / "shared" / name,
                      fs_std::copy_options::overwrite_existing, ec);
    ASSERT_FALSE(ec) << ec.message();
  }

  // Point tokenizer_dir at the shared/ directory using the "package:" scheme.
  std::string config = ReadFile(src_model / "genai_config.json");
  const std::string anchor = "\"type\": \"gpt2\",";
  const auto pos = config.find(anchor);
  ASSERT_NE(pos, std::string::npos);
  config.insert(pos + anchor.size(), "\n    \"tokenizer_dir\": \"package:shared\",");
  WriteFile(variant_dir / "genai_config.json", config);

  auto oga_config = OgaConfig::CreateFromPackageEp(root.string().c_str(), "cpu");
  auto model = OgaModel::Create(*oga_config);
  auto tokenizer = OgaTokenizer::Create(*model);

  auto sequences = OgaSequences::Create();
  tokenizer->Encode("Hello model package", *sequences);
  EXPECT_EQ(sequences->Count(), 1u);
  EXPECT_GT(sequences->SequenceCount(0), 0u);
}

#endif  // ORT_GENAI_HAS_MODEL_PACKAGE
