// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_genai.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>

TEST(NonGenerativeSessionTest, RealPackagesRunWhenConfigured) {
  const char* root_value = std::getenv("ORT_GENAI_NON_GENERATIVE_TEST_ROOT");
  if (!root_value || !*root_value) GTEST_SKIP() << "set ORT_GENAI_NON_GENERATIVE_TEST_ROOT";
  const std::filesystem::path root(root_value);
  const std::string ranking_request =
      R"({"state":{"weather":"heavy rain"},"instructions":"Choose","candidates":{"no":"Do not take an umbrella","yes":"Take an umbrella"}})";
  const std::string decision_request =
      R"({"state":{"weather":"heavy rain"},"questions":{"umbrella":{"type":"noul","instructions":"Should I take an umbrella?","criteria":{}}}})";
  std::vector<std::string> providers;
  if (const char* provider = std::getenv("ORT_GENAI_NON_GENERATIVE_PROVIDER");
      provider && *provider) {
    providers.emplace_back(provider);
  }
  NonGenerativeSession ranking((root / "clm-v0.1-8b-fp32").string(), providers);
  NonGenerativeSession decision((root / "kev-4b-fp32").string(), providers);
  const auto clm = ranking.Run(ranking_request);
  const auto kev = decision.Run(decision_request);
  const auto repeated_clm = ranking.Run(ranking_request);
  const auto repeated_kev = decision.Run(decision_request);
  EXPECT_NE(clm.find(R"("ranked")"), std::string::npos);
  EXPECT_NE(kev.find(R"("umbrella")"), std::string::npos);
  EXPECT_EQ(clm, repeated_clm);
  EXPECT_EQ(kev, repeated_kev);
  const std::string oversized =
      std::string(R"({"state":")") + std::string(100000, 'x') +
      R"(","questions":{"q":{"type":"noul","criteria":{}}}})";
  EXPECT_THROW(decision.Run(oversized), std::runtime_error);
}
