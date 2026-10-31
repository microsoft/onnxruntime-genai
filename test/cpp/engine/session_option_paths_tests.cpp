// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Session config entries that name a path must reach ORT resolved against the model root, whether they come from the
// loaded config or from a runtime profile.

#include <memory>
#include <optional>
#include <string>

#include <gtest/gtest.h>

#include "config.h"
#include "engine_test_helpers.h"

namespace Generators::test {

namespace {

constexpr const char* kExternalInitializersFolder = "session.model_external_initializers_file_folder_path";
constexpr const char* kEpContextFilePath = "ep.context_file_path";

std::optional<std::string> ReadSessionConfigEntry(const OrtSessionOptions& session_options, const char* key) {
  if (!session_options.HasConfigEntry(key)) return std::nullopt;
  size_t size = 0;
  Ort::ThrowOnError(Ort::api->GetSessionConfigEntry(&session_options, key, nullptr, &size));
  std::string value(size, '\0');
  Ort::ThrowOnError(Ort::api->GetSessionConfigEntry(&session_options, key, value.data(), &size));
  if (!value.empty() && value.back() == '\0') value.pop_back();
  return value;
}

std::string UnderModelDirectory(const Model& model, const std::string& relative_path) {
  const fs::path& model_directory = model.config_->config_path;
  return (model_directory / relative_path).string();
}

}  // namespace

TEST(SessionOptionPathsTest, ExternalInitializersFolderIsResolvedAgainstTheModelDirectory) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
  OverlayConfig(*config, R"({"model":{"decoder":{"session_options":{
    "session.model_external_initializers_file_folder_path":"weights"}}}})");
  auto model = CreateModel(GetOrtEnv(), std::move(config));

  EXPECT_EQ(ReadSessionConfigEntry(*model->GetSessionOptions(""), kExternalInitializersFolder),
            UnderModelDirectory(*model, "weights"));
}

TEST(SessionOptionPathsTest, EpContextFilePathIsResolvedAgainstTheModelDirectory) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
  OverlayConfig(*config, R"({"model":{"decoder":{"session_options":{
    "ep.context_file_path":"ctx/model_ctx.onnx"}}}})");
  auto model = CreateModel(GetOrtEnv(), std::move(config));

  EXPECT_EQ(ReadSessionConfigEntry(*model->GetSessionOptions(""), kEpContextFilePath),
            UnderModelDirectory(*model, "ctx/model_ctx.onnx"));
}

TEST(SessionOptionPathsTest, OtherConfigEntriesAreAppliedAsWritten) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
  OverlayConfig(*config, R"({"model":{"decoder":{"session_options":{"session.disable_prepacking":"1"}}}})");
  auto model = CreateModel(GetOrtEnv(), std::move(config));

  EXPECT_EQ(ReadSessionConfigEntry(*model->GetSessionOptions(""), "session.disable_prepacking"),
            std::optional<std::string>{"1"});
}

#if USE_CUDA
namespace {

// Runtime profiles require CUDA, and the profile below matches every device.
std::shared_ptr<Model> LoadCudaModelWithProfile(const std::string& base_session_options,
                                                const std::string& profile_session_options) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-paged");
  ClearProviders(*config);
  SetProviderOption(*config, "cuda", {}, {});
  if (!base_session_options.empty()) {
    OverlayConfig(*config, R"({"model":{"decoder":{"session_options":)" + base_session_options + "}}}");
  }
  OverlayConfig(*config, R"({"runtime_profiles":[{"id":"profile","eligibility":{"minimum_total_device_memory_bytes":0},
    "overlay":{"model":{"decoder":{"session_options":)" +
                             profile_session_options + "}}}}]}");
  return CreateModel(GetOrtEnv(), std::move(config));
}

}  // namespace

TEST(SessionOptionPathsTest, RuntimeProfileAddedPathEntryIsResolved) {
  auto model = LoadCudaModelWithProfile("", R"({"session.model_external_initializers_file_folder_path":"profile-weights"})");

  EXPECT_EQ(ReadSessionConfigEntry(*model->GetSessionOptions(""), kExternalInitializersFolder),
            UnderModelDirectory(*model, "profile-weights"));
}

TEST(SessionOptionPathsTest, RuntimeProfileChangedPathEntryIsResolved) {
  auto model = LoadCudaModelWithProfile(R"({"ep.context_file_path":"ctx/base.onnx"})",
                                        R"({"ep.context_file_path":"ctx/profile.onnx"})");

  EXPECT_EQ(ReadSessionConfigEntry(*model->GetSessionOptions(""), kEpContextFilePath),
            UnderModelDirectory(*model, "ctx/profile.onnx"));
}

TEST(SessionOptionPathsTest, RuntimeProfileLeavesTheOtherEntriesAsConfigured) {
  auto model = LoadCudaModelWithProfile(
      R"({"session.model_external_initializers_file_folder_path":"weights","ep.context_file_path":"ctx/base.onnx"})",
      R"({"session.disable_prepacking":"1"})");

  const auto& session_options = *model->GetSessionOptions("");
  // The profile does not touch the path entries, so applying it must not turn them back into the configured strings.
  EXPECT_EQ(ReadSessionConfigEntry(session_options, kExternalInitializersFolder),
            UnderModelDirectory(*model, "weights"));
  EXPECT_EQ(ReadSessionConfigEntry(session_options, kEpContextFilePath),
            UnderModelDirectory(*model, "ctx/base.onnx"));
  EXPECT_EQ(ReadSessionConfigEntry(session_options, "session.disable_prepacking"), std::optional<std::string>{"1"});
}
#endif

}  // namespace Generators::test
