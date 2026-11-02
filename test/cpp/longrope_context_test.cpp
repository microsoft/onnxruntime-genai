// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <numeric>
#include <vector>
#include <gtest/gtest.h>
#include "ort_genai.h"

namespace {

struct LongRopeContextTest : testing::Test {
  std::unique_ptr<OgaModel> model = OgaModel::Create(MODEL_PATH "longrope-context");

  std::unique_ptr<OgaGenerator> CreateGenerator(int chunk_size = 4) {
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("chunk_size", chunk_size);
    return OgaGenerator::Create(*model, *params);
  }

  void ExpectConsistentCache(OgaGenerator& generator) {
    const auto count = generator.GetSequenceCount(0);
    const auto* tokens = generator.GetSequenceData(0);
    const float sum = std::accumulate(tokens, tokens + count, 0.0f) * (count > 8 ? 2.0f : 1.0f);
    auto tensor = generator.GetLogits();
    const auto* logits = static_cast<const float*>(tensor->Data());
    EXPECT_FLOAT_EQ(logits[0], sum);
    EXPECT_FLOAT_EQ(logits[1], -sum);
    EXPECT_FLOAT_EQ(logits[2], 0);
  }
};

TEST_F(LongRopeContextTest, ChunkedPrefillUsesOneModeForAllChunks) {
  for (int length : {7, 8, 9, 17}) {
    for (int chunk : {0, 1, 4}) {
      SCOPED_TRACE(testing::Message() << "length=" << length << " chunk=" << chunk);
      auto generator = CreateGenerator(chunk);
      generator->AppendTokens(std::vector<int32_t>(length, 1));
      ExpectConsistentCache(*generator);
    }
  }
}

TEST_F(LongRopeContextTest, MissingMetadataDoesNotPreventModelLoading) {
  EXPECT_NO_THROW(OgaModel::Create(MODEL_PATH "longrope-context-without-metadata"));
}

TEST_F(LongRopeContextTest, AppendJumpingOverBoundaryRebuildsCachedPrefix) {
  auto generator = CreateGenerator();
  generator->AppendTokens(std::vector<int32_t>(7, 1));
  generator->AppendTokens(std::vector<int32_t>(3, 1));
  EXPECT_EQ(generator->GetSequenceCount(0), 10u);
  ExpectConsistentCache(*generator);
}

TEST_F(LongRopeContextTest, GeneratorsSharingAModelKeepIndependentCacheModes) {
  auto long_generator = CreateGenerator();
  auto short_generator = CreateGenerator();
  long_generator->AppendTokens(std::vector<int32_t>(17, 1));
  short_generator->AppendTokens(std::vector<int32_t>(7, 1));
  ExpectConsistentCache(*long_generator);
  ExpectConsistentCache(*short_generator);
  long_generator->AppendTokens(std::vector<int32_t>(1, 1));
  short_generator->AppendTokens(std::vector<int32_t>(2, 1));
  ExpectConsistentCache(*long_generator);
  ExpectConsistentCache(*short_generator);
}

TEST_F(LongRopeContextTest, DecodeAcrossBoundaryKeepsSequenceAndCacheConsistent) {
  auto generator = CreateGenerator();
  generator->AppendTokens(std::vector<int32_t>(7, 1));
  for (int i = 0; i < 4; ++i)
    generator->GenerateNextToken();
  EXPECT_EQ(generator->GetSequenceCount(0), 11u);
  ExpectConsistentCache(*generator);
}

TEST_F(LongRopeContextTest, AppendAfterGenerationRebuildsTheCommittedPrefix) {
  auto generator = CreateGenerator();
  generator->AppendTokens(std::vector<int32_t>(7, 1));
  generator->GenerateNextToken();
  generator->AppendTokens(std::vector<int32_t>(3, 1));
  EXPECT_EQ(generator->GetSequenceCount(0), 11u);
  ExpectConsistentCache(*generator);
}

TEST_F(LongRopeContextTest, RewindAcrossBoundaryRebuildsShortCache) {
  auto generator = CreateGenerator();
  generator->AppendTokens(std::vector<int32_t>(10, 1));
  generator->RewindTo(5);
  generator->AppendTokens(std::vector<int32_t>(2, 1));
  EXPECT_EQ(generator->GetSequenceCount(0), 7u);
  ExpectConsistentCache(*generator);
}

TEST_F(LongRopeContextTest, RewindToZeroAllowsReuseInBothModes) {
  auto generator = CreateGenerator();
  for (int length : {17, 7, 9}) {
    generator->RewindTo(0);
    generator->AppendTokens(std::vector<int32_t>(length, 1));
    ExpectConsistentCache(*generator);
  }
}

TEST_F(LongRopeContextTest, UnsupportedBatchingFailsInsteadOfMixingCacheModes) {
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("batch_size", 2);
  EXPECT_THROW(OgaGenerator::Create(*model, *params), std::runtime_error);
  params->SetSearchOption("batch_size", 1);
  params->SetSearchOption("num_beams", 2);
  EXPECT_THROW(OgaGenerator::Create(*model, *params), std::runtime_error);
}

}  // namespace
