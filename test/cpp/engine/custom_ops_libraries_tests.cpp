// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <functional>
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
  EXPECT_THROW(OverlayConfig(config, R"({"model":{"decoder":{"session_options":{"custom_ops_libraries":"single.so"}}}})"),
               std::exception);
}

TEST(CustomOpsLibrariesConfigTest, InheritsOnlyWhenChildLeavesTheListEmpty) {
  Config::SessionOptions parent;
  parent.custom_ops_library = "parent_single.so";
  parent.custom_ops_libraries = {"parent_a.so", "parent_b.so"};
  parent.custom_ops_libraries_present = true;

  Config::SessionOptions inheriting;
  InheritSessionOptionsFromParent(parent, inheriting);
  EXPECT_EQ(inheriting.custom_ops_library, "parent_single.so");
  EXPECT_EQ(inheriting.custom_ops_libraries, parent.custom_ops_libraries);

  Config::SessionOptions overriding;
  overriding.custom_ops_libraries = {"child.so"};
  InheritSessionOptionsFromParent(parent, overriding);
  EXPECT_EQ(overriding.custom_ops_libraries, std::vector<std::string>{"child.so"});

  // An explicit empty array in JSON opts out of inheriting parent custom_ops_libraries.
  Config explicit_empty_cfg;
  OverlayConfig(explicit_empty_cfg, R"({
    "model": {
      "decoder": {
        "hidden_size": 64,
        "session_options": {
          "custom_ops_library": "parent_single.so",
          "custom_ops_libraries": ["parent_a.so", "parent_b.so"]
        }
      },
      "mtp": {
        "enabled": true,
        "filename": "mtp.onnx",
        "num_hidden_layers": 1,
        "num_key_value_heads": 2,
        "head_size": 32,
        "session_options": {
          "custom_ops_libraries": []
        }
      }
    }
  })");
  EXPECT_TRUE(explicit_empty_cfg.model.mtp.session_options.has_value());
  EXPECT_TRUE(explicit_empty_cfg.model.mtp.session_options->custom_ops_libraries_present);
  EXPECT_TRUE(explicit_empty_cfg.model.mtp.session_options->custom_ops_libraries.empty());

  auto mtp_cfg = CreateMtpDecoderConfig(explicit_empty_cfg);
  EXPECT_EQ(mtp_cfg->model.decoder.session_options.custom_ops_library, "parent_single.so");
  EXPECT_TRUE(mtp_cfg->model.decoder.session_options.custom_ops_libraries.empty());
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

  // An absolute file outside the model folder, the working directory and every EP directory.
  const auto outside_dir = std::filesystem::temp_directory_path();
  std::error_code ec;
  const auto cwd = std::filesystem::canonical(std::filesystem::current_path(), ec);
  const auto canonical_outside = std::filesystem::canonical(outside_dir, ec);
  if (!ec && canonical_outside != cwd && cwd.string().rfind(canonical_outside.string(), 0) != 0 &&
      canonical_outside.string().rfind(cwd.string(), 0) != 0) {
    const auto outside_file = canonical_outside / "genai_custom_ops_allow_list_test.so";
    {
      std::ofstream out{outside_file};
      out << "not a library";
    }
    Config::SessionOptions outside;
    outside.custom_ops_libraries = {allowed_entry, outside_file.string()};
    const std::string message = ExceptionMessage([&] { CreateSessionOptions(*model, outside); });
    // custom_ops_library keeps the same check.
    Config::SessionOptions outside_single;
    outside_single.custom_ops_library = outside_file.string();
    const std::string single_message = ExceptionMessage([&] { CreateSessionOptions(*model, outside_single); });
    std::filesystem::remove(outside_file, ec);
    EXPECT_NE(message.find("not within an allowed directory"), std::string::npos) << message;
    EXPECT_NE(single_message.find("not within an allowed directory"), std::string::npos) << single_message;
  }
}

}  // namespace Generators::test
