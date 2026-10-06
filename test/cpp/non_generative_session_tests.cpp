// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_genai.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>

TEST(NonGenerativeSessionTest, PublicTypedValuesPreserveOrderAndTypes) {
  OgaStructuredRequest request;
  request.state = OgaStructuredValue::Object{
      {"enabled", true}, {"count", int64_t{3}}, {"label", "demo"}};
  request.questions.emplace_back(
      "q", OgaQuestion{"choice", "Choose",
                       OgaStructuredValue::Object{{"first", "A"}, {"second", "B"}}});
  ASSERT_EQ(request.questions.front().first, "q");
  const auto& object = std::get<OgaStructuredValue::Object>(request.state.value);
  EXPECT_EQ(object[0].first, "enabled");
  EXPECT_TRUE(std::get<bool>(object[0].second.value));
  EXPECT_EQ(std::get<int64_t>(object[1].second.value), 3);
}

TEST(NonGenerativeSessionTest, RealPackagesRunWhenConfigured) {
  const char* root_value = std::getenv("ORT_GENAI_NON_GENERATIVE_TEST_ROOT");
  if (!root_value || !*root_value) GTEST_SKIP() << "set ORT_GENAI_NON_GENERATIVE_TEST_ROOT";
  const std::filesystem::path root(root_value);
  OgaStructuredRequest request;
  request.state = OgaStructuredValue::Object{{"weather", "heavy rain"}};
  request.questions.emplace_back(
      "umbrella", OgaQuestion{"noul", "Should I take an umbrella?", {}});
  std::vector<std::string> providers;
  if (const char* provider = std::getenv("ORT_GENAI_NON_GENERATIVE_PROVIDER");
      provider && *provider) {
    providers.emplace_back(provider);
  }
  RankingSession ranking((root / "clm-v0.1-8b-fp32").string(), providers);
  DecisionSession decision((root / "kev-4b-fp32").string(), providers);
  const auto clm = ranking.Run(request);
  const auto kev = decision.Decide(request);
  const auto repeated_clm = ranking.Run(request);
  const auto repeated_kev = decision.Decide(request);
  ASSERT_EQ(clm.answers.size(), 1u);
  ASSERT_EQ(kev.answers.size(), 1u);
  ASSERT_TRUE(clm.answers.front().second.noul.has_value());
  ASSERT_TRUE(kev.answers.front().second.noul.has_value());
  EXPECT_EQ(clm.answers.front().second.noul, repeated_clm.answers.front().second.noul);
  EXPECT_EQ(kev.answers.front().second.noul, repeated_kev.answers.front().second.noul);
  EXPECT_GE(ranking.CacheStats().hits, 3u);
  EXPECT_GT(decision.CacheStats().hits, 0u);

  if (providers.empty()) {
    OgaStructuredRequest duplicate_ids;
    duplicate_ids.state =
        OgaStructuredValue::Object{{"weather", "heavy rain"}};
    for (size_t index = 0; index < 4; ++index)
      duplicate_ids.questions.emplace_back(
          "duplicate",
          OgaQuestion{
              "noul",
              "Question " + std::to_string(index),
              {}});
    const auto duplicate_result = decision.Decide(duplicate_ids);
    ASSERT_EQ(duplicate_result.answers.size(), 4u);
    for (const auto& [id, answer] : duplicate_result.answers) {
      EXPECT_EQ(id, "duplicate");
      EXPECT_TRUE(answer.noul.has_value());
    }
  }

  RankingSession isolated((root / "clm-v0.1-8b-fp32").string(), providers);
  EXPECT_EQ(isolated.CacheStats().hits, 0u);
  ranking.ClearCache();
  EXPECT_EQ(ranking.CacheStats().entries, 0u);
  decision.InvalidateCache();
  EXPECT_EQ(decision.CacheStats().entries, 0u);

  ranking.SetCacheCapacity(1, 1024 * 1024);
  ranking.Run(request);
  EXPECT_LE(ranking.CacheStats().entries, 1u);
  EXPECT_GT(ranking.CacheStats().evictions, 0u);

  request.state = std::string(100000, 'x');
  EXPECT_THROW(decision.Decide(request), std::runtime_error);
}
