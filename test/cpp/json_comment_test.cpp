// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests that the runtime accepts // and /* ... */ comments in genai_config.json while
// loading it. These exercise the feature through the public C API boundary (OgaConfig::Create),
// which parses genai_config.json via the internal JSON parser, so they keep working regardless
// of where the comment handling lives inside the library.

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
                   ("ortgenai_json_comment_test_" + suffix + "_" + std::to_string(counter));
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

// Writes `config` as genai_config.json in a fresh temp dir and returns the dir path.
fs_std::path WriteConfig(const std::string& suffix, const std::string& config) {
  const auto root = MakeTempDir(suffix);
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

TEST(JsonCommentTest, CommentsBeforeAndBetweenFields) {
  const auto root = WriteConfig("before_between",
      "{\n"
      "  // A line comment before the first field\n"
      "  \"model\": {\n"
      "    \"type\": \"tiny-test-model\",\n"
      "    /* a block comment between fields */\n"
      "    \"vocab_size\": 16,\n"
      "    // another line comment\n"
      "    \"context_length\": 32\n"
      "  },\n"
      "  \"search\": {}\n"
      "}");
  EXPECT_NO_THROW(OgaConfig::Create(root.string().c_str()));
}

TEST(JsonCommentTest, CommentsInArrays) {
  const auto root = WriteConfig("arrays",
      "{\n"
      "  \"model\": {\n"
      "    \"type\": \"tiny-test-model\",\n"
      "    \"vocab_size\": 16,\n"
      "    \"context_length\": 32,\n"
      "    \"eos_token_id\": [\n"
      "      /* comment before the first element */\n"
      "      1,\n"
      "      // comment between elements\n"
      "      2\n"
      "      // comment after the last element\n"
      "    ]\n"
      "  },\n"
      "  \"search\": {}\n"
      "}");
  EXPECT_NO_THROW(OgaConfig::Create(root.string().c_str()));
}

TEST(JsonCommentTest, CommentLikeTextInsideStrings) {
  // The "//" inside the quoted string must stay part of the string, not start a comment.
  const auto root = WriteConfig("string",
      "{\n"
      "  \"model\": {\n"
      "    \"type\": \"tiny-test-model\",\n"
      "    \"vocab_size\": 16,\n"
      "    \"context_length\": 32,\n"
      "    \"tokenizer_dir\": \"https://example.com\"\n"
      "  },\n"
      "  \"search\": {}\n"
      "}");
  EXPECT_NO_THROW(OgaConfig::Create(root.string().c_str()));
}

TEST(JsonCommentTest, UnterminatedBlockCommentThrows) {
  const auto root = WriteConfig("unterminated",
      "{\n"
      "  \"model\": {\n"
      "    \"type\": \"tiny-test-model\",\n"
      "    /* an unterminated block comment\n"
      "    \"context_length\": 32\n"
      "  },\n"
      "  \"search\": {}\n"
      "}");
  const std::string message =
      CaptureThrowMessage([&] { OgaConfig::Create(root.string().c_str()); });
  EXPECT_NE(message.find("Unterminated block comment"), std::string::npos) << message;
}

}  // namespace Generators::test
