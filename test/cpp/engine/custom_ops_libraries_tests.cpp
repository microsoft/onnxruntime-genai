// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>

#include <gtest/gtest.h>

#include "config.h"
#include "engine_test_helpers.h"

namespace Generators::test {

namespace {

std::string ExceptionMessage(const std::function<void()>& action) {
  try {
    action();
  } catch (const std::exception& error) {
    return error.what();
  }
  return {};
}

// Runs session-option creation without appending providers, so it only exercises the custom-op
// handling this test is about.
void CreateSessionOptions(Model& model, const Config::SessionOptions& options) {
  auto session_options = OrtSessionOptions::Create();
  model.CreateSessionOptionsFromConfig(options, *session_options, /*is_primary_session_options=*/false,
                                       /*disable_graph_capture=*/false, /*cloned_from_parent=*/false,
                                       /*append_providers=*/false);
}

}  // namespace

TEST(CustomOpsLibrariesConfigTest, ParsesListInOrder) {
  Config config;
  OverlayConfig(config, R"({"model":{"decoder":{"session_options":{
    "custom_ops_library":"first.so",
    "custom_ops_libraries":["second.so","third.so"]
  }}}})");

  const auto& options = config.model.decoder.session_options;
  EXPECT_EQ(options.custom_ops_library, "first.so");
  ASSERT_EQ(options.custom_ops_libraries.size(), 2u);
  EXPECT_EQ(options.custom_ops_libraries[0], "second.so");
  EXPECT_EQ(options.custom_ops_libraries[1], "third.so");
}

TEST(CustomOpsLibrariesConfigTest, RejectsNonStringEntries) {
  Config config;
  EXPECT_THROW(OverlayConfig(config, R"({"model":{"decoder":{"session_options":{"custom_ops_libraries":[1]}}}})"),
               std::exception);
}

TEST(CustomOpsLibrariesConfigTest, InheritsOnlyWhenChildSetsNoCustomOps) {
  Config::SessionOptions parent;
  parent.custom_ops_library = "parent_single.so";
  parent.custom_ops_libraries = {"parent_a.so", "parent_b.so"};

  Config::SessionOptions inheriting;
  InheritSessionOptionsFromParent(parent, inheriting);
  EXPECT_EQ(inheriting.custom_ops_library, "parent_single.so");
  EXPECT_EQ(inheriting.custom_ops_libraries, parent.custom_ops_libraries);

  // A child list replaces the parent's custom ops entirely, including the parent's single library.
  Config::SessionOptions overriding_list;
  overriding_list.custom_ops_libraries = {"child.so"};
  InheritSessionOptionsFromParent(parent, overriding_list);
  EXPECT_FALSE(overriding_list.custom_ops_library.has_value());
  EXPECT_EQ(overriding_list.custom_ops_libraries, std::vector<std::string>{"child.so"});

  // A child single library likewise does not pick up the parent's list.
  Config::SessionOptions overriding_single;
  overriding_single.custom_ops_library = "child_single.so";
  InheritSessionOptionsFromParent(parent, overriding_single);
  EXPECT_EQ(overriding_single.custom_ops_library, "child_single.so");
  EXPECT_TRUE(overriding_single.custom_ops_libraries.empty());
}

TEST(CustomOpsLibrariesConfigTest, EveryEntryPassesTheAllowListBeforeAnyIsRegistered) {
  auto model = LoadDummyDecoderModel();
  const auto model_dir = std::filesystem::path{model->config_->config_path.string()};
  // Exists and lives inside the model folder, so it is allowed by the allow-list; the first entry
  // of each case below is therefore never the reason for a failure.
  const std::string allowed_entry = (model_dir / "genai_config.json").string();

  Config::SessionOptions traversal;
  traversal.custom_ops_libraries = {allowed_entry, "subdir/../escape.so"};
  EXPECT_NE(ExceptionMessage([&] { CreateSessionOptions(*model, traversal); })
                .find("must not contain path traversal"),
            std::string::npos);

  Config::SessionOptions single_traversal;
  single_traversal.custom_ops_library = "../escape.so";
  EXPECT_NE(ExceptionMessage([&] { CreateSessionOptions(*model, single_traversal); })
                .find("must not contain path traversal"),
            std::string::npos);
}

TEST(CustomOpsLibrariesConfigTest, RejectsEntriesThatCannotBeResolved) {
  auto model = LoadDummyDecoderModel();
  const auto model_dir = std::filesystem::path{model->config_->config_path.string()};
  const std::string allowed_entry = (model_dir / "genai_config.json").string();

  // A bare name that exists in no allowed directory must not be left to the platform loader's search path.
  Config::SessionOptions missing_in_list;
  missing_in_list.custom_ops_libraries = {allowed_entry, "genai_missing_custom_ops_library.so"};
  EXPECT_NE(ExceptionMessage([&] { CreateSessionOptions(*model, missing_in_list); })
                .find("not found in any allowed directory"),
            std::string::npos);

  Config::SessionOptions missing_single;
  missing_single.custom_ops_library = "genai_missing_custom_ops_library.so";
  EXPECT_NE(ExceptionMessage([&] { CreateSessionOptions(*model, missing_single); })
                .find("not found in any allowed directory"),
            std::string::npos);

  Config::SessionOptions missing_absolute;
  missing_absolute.custom_ops_libraries = {(model_dir / "genai_missing_custom_ops_library.so").string()};
  EXPECT_NE(ExceptionMessage([&] { CreateSessionOptions(*model, missing_absolute); })
                .find("not found in any allowed directory"),
            std::string::npos);
}

TEST(CustomOpsLibrariesConfigTest, RejectsExistingFilesOutsideAllowedDirectories) {
  auto model = LoadDummyDecoderModel();
  const auto model_dir = std::filesystem::path{model->config_->config_path.string()};
  const std::string allowed_entry = (model_dir / "genai_config.json").string();

  std::error_code ec;
  const auto outside_dir = std::filesystem::canonical(std::filesystem::temp_directory_path(), ec) /
                           ("genai_custom_ops_allow_list_test_" + std::to_string(std::random_device{}()));
  ASSERT_TRUE(std::filesystem::create_directories(outside_dir, ec)) << ec.message();
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() {
      std::error_code ignored;
      std::filesystem::remove_all(path, ignored);
    }
  } cleanup{outside_dir};

  const auto is_under = [](const std::filesystem::path& path, const std::filesystem::path& dir) {
    std::error_code error;
    const auto canonical_dir = std::filesystem::canonical(dir, error);
    return error || path.string().rfind(canonical_dir.string(), 0) == 0;
  };
  if (is_under(outside_dir, std::filesystem::current_path()) || is_under(outside_dir, model_dir)) {
    GTEST_SKIP() << "The temp directory is inside an allowed directory, so it cannot be used as an outside location.";
  }

  const auto outside_file = outside_dir / "custom_ops.so";
  {
    std::ofstream out{outside_file};
    out << "not a library";
  }

  Config::SessionOptions outside;
  outside.custom_ops_libraries = {allowed_entry, outside_file.string()};
  Config::SessionOptions outside_single;
  outside_single.custom_ops_library = outside_file.string();
  EXPECT_NE(ExceptionMessage([&] { CreateSessionOptions(*model, outside); }).find("not within an allowed directory"),
            std::string::npos);
  EXPECT_NE(ExceptionMessage([&] { CreateSessionOptions(*model, outside_single); })
                .find("not within an allowed directory"),
            std::string::npos);
}

}  // namespace Generators::test
