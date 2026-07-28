// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// End-to-end tests for config path validation. These exercise the check through the
// public C API boundary (OgaConfig::Create) rather than reaching into an internal helper,
// so the tests keep working regardless of where the validation logic lives inside the
// library.

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "ort_genai.h"

namespace Generators::test {
namespace {

namespace fs_std = std::filesystem;

// Creates a fresh, empty per-test directory under the build's temp area.
fs_std::path MakeTempDir(const std::string& suffix) {
  static int counter = 0;
  ++counter;
  const auto dir = fs_std::temp_directory_path() /
                   ("ortgenai_vcp_test_" + suffix + "_" + std::to_string(counter));
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

// Builds a minimal genai_config.json that sets `decoder.filename` to `path`, wraps it in a
// fresh temp directory, and returns the directory path. `decoder.filename` is used as a
// representative config path field — all path fields share the same validation.
fs_std::path WriteConfigWithDecoderFilename(const std::string& suffix, const std::string& path) {
  const auto root = MakeTempDir(suffix);
  // JSON-escape backslashes so Windows paths like "C:\\..." survive parsing.
  std::string escaped;
  escaped.reserve(path.size());
  for (char c : path) {
    if (c == '\\' || c == '"') escaped.push_back('\\');
    escaped.push_back(c);
  }
  const std::string config =
      "{ \"model\": { \"type\": \"tiny-test-model\","
      " \"vocab_size\": 16, \"context_length\": 32,"
      " \"decoder\": { \"filename\": \"" +
      escaped +
      "\" } },"
      " \"search\": {} }";
  WriteFile(root / "genai_config.json", config);
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

TEST(ValidateConfigPathTest, AcceptsSimpleFilename) {
  const auto root = WriteConfigWithDecoderFilename("simple", "model.onnx");
  EXPECT_NO_THROW(OgaConfig::Create(root.string().c_str()));
}

TEST(ValidateConfigPathTest, AcceptsRelativeSubdirectory) {
  const auto root = WriteConfigWithDecoderFilename("subdir", "subdir/model.onnx");
  EXPECT_NO_THROW(OgaConfig::Create(root.string().c_str()));
}

TEST(ValidateConfigPathTest, AcceptsSingleDot) {
  const auto root = WriteConfigWithDecoderFilename("dot", "./model.onnx");
  EXPECT_NO_THROW(OgaConfig::Create(root.string().c_str()));
}

TEST(ValidateConfigPathTest, RejectsAbsolutePathUnix) {
  const auto root = WriteConfigWithDecoderFilename("abs_unix", "/etc/passwd");
  const std::string message =
      CaptureThrowMessage([&] { OgaConfig::Create(root.string().c_str()); });
  EXPECT_NE(message.find("model.decoder.filename"), std::string::npos) << message;
  EXPECT_NE(message.find("relative path"), std::string::npos) << message;
}

#if defined(_WIN32)
TEST(ValidateConfigPathTest, RejectsAbsolutePathWindows) {
  const auto root =
      WriteConfigWithDecoderFilename("abs_win", "C:\\Windows\\System32\\evil.dll");
  const std::string message =
      CaptureThrowMessage([&] { OgaConfig::Create(root.string().c_str()); });
  EXPECT_NE(message.find("model.decoder.filename"), std::string::npos) << message;
  EXPECT_NE(message.find("relative path"), std::string::npos) << message;
}

TEST(ValidateConfigPathTest, RejectsDriveRelativePathWindows) {
  const auto root =
      WriteConfigWithDecoderFilename("drive_win", "C:Windows\\System32\\evil.dll");
  const std::string message =
      CaptureThrowMessage([&] { OgaConfig::Create(root.string().c_str()); });
  EXPECT_NE(message.find("model.decoder.filename"), std::string::npos) << message;
  EXPECT_NE(message.find("relative path"), std::string::npos) << message;
}
#endif

TEST(ValidateConfigPathTest, RejectsParentTraversal) {
  const auto root = WriteConfigWithDecoderFilename("traversal", "../../../etc/passwd");
  const std::string message =
      CaptureThrowMessage([&] { OgaConfig::Create(root.string().c_str()); });
  EXPECT_NE(message.find("model.decoder.filename"), std::string::npos) << message;
  EXPECT_NE(message.find("path traversal"), std::string::npos) << message;
}

TEST(ValidateConfigPathTest, RejectsEmbeddedParentTraversal) {
  const auto root = WriteConfigWithDecoderFilename("embedded", "subdir/../../etc/passwd");
  const std::string message =
      CaptureThrowMessage([&] { OgaConfig::Create(root.string().c_str()); });
  EXPECT_NE(message.find("model.decoder.filename"), std::string::npos) << message;
  EXPECT_NE(message.find("path traversal"), std::string::npos) << message;
}

TEST(ValidateConfigPathTest, RejectsSingleDotDot) {
  const auto root = WriteConfigWithDecoderFilename("dotdot", "..");
  const std::string message =
      CaptureThrowMessage([&] { OgaConfig::Create(root.string().c_str()); });
  EXPECT_NE(message.find("model.decoder.filename"), std::string::npos) << message;
  EXPECT_NE(message.find("path traversal"), std::string::npos) << message;
}

}  // namespace Generators::test
