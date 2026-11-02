// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_genai_c.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

namespace {

using ResultPtr = std::unique_ptr<OgaResult, decltype(&OgaDestroyResult)>;

void Check(OgaResult* result) {
  ResultPtr owned(result, OgaDestroyResult);
  ASSERT_EQ(result, nullptr)
      << (result ? OgaResultGetError(result) : "");
}

void ExpectError(OgaResult* result, std::string_view expected) {
  ResultPtr owned(result, OgaDestroyResult);
  ASSERT_NE(result, nullptr);
  EXPECT_NE(std::string(OgaResultGetError(result)).find(expected),
            std::string::npos);
}

}  // namespace

TEST(NonGenerativeCApiTest, JsonWorkflowReportsContractErrors) {
  OgaConfig* config{};
  ExpectError(OgaCreateConfig("path-that-does-not-exist", &config), "");
  EXPECT_EQ(config, nullptr);

  OgaNonGenerativeSession* session{};
  ExpectError(OgaCreateNonGenerativeSession(nullptr, &session),
              "config must not be null");
  EXPECT_EQ(session, nullptr);

  const char* response{};
  ExpectError(
      OgaNonGenerativeSessionRunJson(nullptr, "{}", 2, &response),
      "session must not be null");
  EXPECT_EQ(response, nullptr);
}

TEST(NonGenerativeCApiTest, FreeFormRankingRunsWhenConfigured) {
  const char* root_value = std::getenv("ORT_GENAI_NON_GENERATIVE_TEST_ROOT");
  if (!root_value || !*root_value)
    GTEST_SKIP() << "set ORT_GENAI_NON_GENERATIVE_TEST_ROOT";
  const auto package =
      std::filesystem::path(root_value) / "clm-v0.1-8b-fp32";

  OgaConfig* config{};
  Check(OgaCreateConfig(package.string().c_str(), &config));
  OgaNonGenerativeSession* session{};
  Check(OgaCreateNonGenerativeSession(config, &session));
  OgaDestroyConfig(config);
  constexpr std::string_view request =
      R"({"state":null,"instructions":"Choose","candidates":{"first":"same","second":"same"}})";
  const char* response{};
  Check(OgaNonGenerativeSessionRunJson(session, request.data(), request.size(),
                                       &response));
  std::unique_ptr<const char, decltype(&OgaDestroyString)> owned(
      response, OgaDestroyString);
  const std::string output(response);
  EXPECT_NE(output.find(R"("model":"clm")"), std::string::npos);
  EXPECT_NE(output.find(R"("key":"first")"), std::string::npos);
  EXPECT_NE(output.find(R"("probability":0.5)"), std::string::npos);
  OgaDestroyNonGenerativeSession(session);
}

TEST(NonGenerativeCApiTest, NullCreationOutputsFailBeforeAllocating) {
  for (int iteration = 0; iteration < 16; ++iteration) {
    ExpectError(
        OgaCreateNonGenerativeSession(nullptr, nullptr),
        "out must not be null");
  }
}
