// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <fstream>
#include <iterator>
#include <numeric>
#include <string>
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

TEST_F(LongRopeContextTest, LegacyGraphRunsWithoutContextInput) {
  model = OgaModel::Create(MODEL_PATH "longrope-context-without-metadata");
  for (int length : {7, 8, 9}) {
    SCOPED_TRACE(testing::Message() << "length=" << length);
    auto generator = CreateGenerator(0);
    generator->AppendTokens(std::vector<int32_t>(length, 1));
    ExpectConsistentCache(*generator);
    generator->GenerateNextToken();
    EXPECT_EQ(generator->GetSequenceCount(0), static_cast<size_t>(length + 1));
    EXPECT_EQ(generator->GetSequenceData(0)[length], 0);
  }
}

TEST_F(LongRopeContextTest, LegacyGraphSupportsSameRegimeContinuation) {
  model = OgaModel::Create(MODEL_PATH "longrope-context-without-metadata");
  for (int length : {7, 9}) {
    SCOPED_TRACE(testing::Message() << "length=" << length);
    auto generator = CreateGenerator(0);
    generator->AppendTokens(std::vector<int32_t>(length, 1));
    generator->AppendTokens(std::vector<int32_t>{1});
    ExpectConsistentCache(*generator);
    generator->GenerateNextToken();
    EXPECT_EQ(generator->GetSequenceCount(0), static_cast<size_t>(length + 2));
    EXPECT_EQ(generator->GetSequenceData(0)[length + 1], 0);
  }
}

TEST_F(LongRopeContextTest, ReplayUsesLogitsOfCommittedPadToken) {
  for (const auto* path : {MODEL_PATH "longrope-context", MODEL_PATH "longrope-context-runtime-indices"}) {
    auto config = OgaConfig::Create(path);
    config->Overlay(R"({"model":{"pad_token_id":1,"eos_token_id":2}})");
    model = OgaModel::Create(*config);
    for (int chunk : {0, 3, 4}) {
      SCOPED_TRACE(testing::Message() << "path=" << path << " chunk=" << chunk);
      auto generator = CreateGenerator(chunk);
      generator->AppendTokens(std::vector<int32_t>(8, 0));
      std::array<float, 3> forced_logits{0, 10, -10};
      generator->SetLogits(*OgaTensor::Create(forced_logits.data(), std::array<int64_t, 2>{1, 3}));
      generator->GenerateNextToken();
      ASSERT_EQ(generator->GetSequenceCount(0), 9u);
      ASSERT_EQ(generator->GetSequenceData(0)[8], 1);
      ExpectConsistentCache(*generator);
      generator->GenerateNextToken();
      EXPECT_EQ(generator->GetSequenceData(0)[9], 0);
    }
  }
}

TEST_F(LongRopeContextTest, ChunkedPrefillAcceptsDiscardedPaddingOnlyChunks) {
  auto config = OgaConfig::Create(MODEL_PATH "longrope-context-runtime-indices");
  config->Overlay(R"({"model":{"pad_token_id":0,"eos_token_id":2}})");
  model = OgaModel::Create(*config);
  auto generator = CreateGenerator(2);
  ASSERT_NO_THROW(generator->AppendTokens(std::vector<int32_t>{0, 0, 1, 1}));
  ExpectConsistentCache(*generator);
}

TEST_F(LongRopeContextTest, RuntimeIndicesRejectPaddingOnlyFinalSingletonChunks) {
  auto config = OgaConfig::Create(MODEL_PATH "longrope-context-runtime-indices");
  config->Overlay(R"({"model":{"pad_token_id":0,"eos_token_id":2}})");
  model = OgaModel::Create(*config);
  for (int chunk : {0, 1, 2}) {
    for (const auto& prompt : {std::vector<int32_t>{0}, std::vector<int32_t>{1, 1, 0}}) {
      if (chunk == 0 && prompt.size() > 1)
        continue;
      SCOPED_TRACE(testing::Message() << "chunk=" << chunk << " prompt_length=" << prompt.size());
      auto generator = CreateGenerator(chunk);
      try {
        generator->AppendTokens(prompt);
        ADD_FAILURE() << "Accepted a padding-only final singleton prompt chunk";
      } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("non-padding token in each current chunk"), std::string::npos);
      }
    }
  }
}

TEST_F(LongRopeContextTest, RuntimeIndicesValidateAppendedSingletonPrompts) {
  auto config = OgaConfig::Create(MODEL_PATH "longrope-context-runtime-indices");
  config->Overlay(R"({"model":{"pad_token_id":0,"eos_token_id":2}})");
  model = OgaModel::Create(*config);
  for (int chunk : {0, 1, 2}) {
    SCOPED_TRACE(testing::Message() << "chunk=" << chunk);
    for (int length : {1, 8}) {
      SCOPED_TRACE(testing::Message() << "length=" << length);
      auto generator = CreateGenerator(chunk);
      ASSERT_NO_THROW(generator->AppendTokens(std::vector<int32_t>(length, 1)));
      ExpectConsistentCache(*generator);
      EXPECT_THROW(generator->AppendTokens(std::vector<int32_t>{0}), std::runtime_error);

      generator = CreateGenerator(chunk);
      ASSERT_NO_THROW(generator->AppendTokens(std::vector<int32_t>(length, 1)));
      ASSERT_NO_THROW(generator->AppendTokens(std::vector<int32_t>{1}));
      ExpectConsistentCache(*generator);
    }

    auto generator = CreateGenerator(chunk);
    ASSERT_NO_THROW(generator->AppendTokens(std::vector<int32_t>{0, 1}));
    ExpectConsistentCache(*generator);
  }
}

TEST_F(LongRopeContextTest, GeneratedPadTokenRemainsValidWithoutReplay) {
  for (const auto* path : {MODEL_PATH "longrope-context", MODEL_PATH "longrope-context-runtime-indices"}) {
    auto config = OgaConfig::Create(path);
    config->Overlay(R"({"model":{"pad_token_id":1,"eos_token_id":2}})");
    model = OgaModel::Create(*config);
    for (int chunk : {0, 1, 2}) {
      SCOPED_TRACE(testing::Message() << "path=" << path << " chunk=" << chunk);
      auto generator = CreateGenerator(chunk);
      generator->AppendTokens(std::vector<int32_t>{0});
      std::array<float, 3> forced_logits{0, 10, -10};
      generator->SetLogits(*OgaTensor::Create(forced_logits.data(), std::array<int64_t, 2>{1, 3}));
      generator->GenerateNextToken();
      ASSERT_EQ(generator->GetSequenceData(0)[1], 1);
      ExpectConsistentCache(*generator);
      generator->GenerateNextToken();
      EXPECT_EQ(generator->GetSequenceData(0)[2], 0);
      ASSERT_NO_THROW(generator->AppendTokens(std::vector<int32_t>{0}));
      ExpectConsistentCache(*generator);
    }
  }
}

TEST_F(LongRopeContextTest, OrdinaryPromptStillIgnoresTrailingPadding) {
  auto config = OgaConfig::Create(MODEL_PATH "longrope-context");
  config->Overlay(R"({"model":{"pad_token_id":1,"eos_token_id":2}})");
  model = OgaModel::Create(*config);
  for (int chunk : {0, 3, 4}) {
    SCOPED_TRACE(testing::Message() << "chunk=" << chunk);
    auto generator = CreateGenerator(chunk);
    generator->AppendTokens(std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 1});
    auto tensor = generator->GetLogits();
    const auto* logits = static_cast<const float*>(tensor->Data());
    EXPECT_FLOAT_EQ(logits[0], 0);
    EXPECT_FLOAT_EQ(logits[1], 0);
    EXPECT_FLOAT_EQ(logits[2], 0);
  }
}

TEST_F(LongRopeContextTest, RuntimeCapacityAtOrBelowBoundaryUsesShortCache) {
  for (int capacity : {7, 8}) {
    auto config = OgaConfig::Create(MODEL_PATH "longrope-context");
    const auto overlay = "{\"model\":{\"context_length\":" + std::to_string(capacity) +
                         "},\"search\":{\"max_length\":" + std::to_string(capacity) + "}}";
    config->Overlay(overlay.c_str());
    model = OgaModel::Create(*config);
    auto generator = CreateGenerator();
    generator->AppendTokens(std::vector<int32_t>(capacity, 1));
    ExpectConsistentCache(*generator);
  }
}

TEST_F(LongRopeContextTest, EngineRejectsExplicitContextContractAtConstruction) {
  try {
    auto engine = OgaEngine::Create(*model);
    FAIL() << "Engine accepted the unsupported LongRoPE context contract";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("Engine does not support the explicit LongRoPE context contract"),
              std::string::npos);
  }
}

TEST_F(LongRopeContextTest, SpeculativeDecodingRejectsContextContractInEitherModel) {
  for (bool target_has_metadata : {false, true}) {
    auto config = OgaConfig::Create(target_has_metadata ? MODEL_PATH "longrope-context"
                                                        : MODEL_PATH "longrope-context-without-metadata");
    config->Overlay(R"({"model":{"draft":{"filename":"draft.onnx","num_hidden_layers":1,
                      "num_attention_heads":1,"num_key_value_heads":1,"head_size":1,"hidden_size":1,
                      "session_options":{"intra_op_num_threads":1}}}})");
    std::ifstream input(target_has_metadata ? MODEL_PATH "longrope-context-without-metadata/model.onnx"
                                            : MODEL_PATH "longrope-context/model.onnx",
                        std::ios::binary);
    ASSERT_TRUE(input);
    const std::string graph{std::istreambuf_iterator<char>{input}, {}};
    config->AddModelData("draft.onnx", graph.data(), graph.size());
    try {
      auto speculative_model = OgaModel::Create(*config);
      FAIL() << "Speculative decoding accepted the unsupported LongRoPE context contract";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string(error.what()).find("explicit LongRoPE context contract in target or draft models"),
                std::string::npos);
    }
  }
}

TEST_F(LongRopeContextTest, NGramDecodingRejectsContextContractAtConstruction) {
  for (bool configure_model : {false, true}) {
    auto config = OgaConfig::Create(MODEL_PATH "longrope-context");
    if (configure_model)
      config->Overlay(R"({"speculative":{"ngram_size":2}})");
    model = OgaModel::Create(*config);
    auto params = OgaGeneratorParams::Create(*model);
    if (!configure_model)
      params->SetSpeculativeNumber("ngram_size", 2);
    try {
      auto generator = OgaGenerator::Create(*model, *params);
      FAIL() << "N-gram decoding accepted the unsupported LongRoPE context contract";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string(error.what()).find("N-gram decoding does not support the explicit LongRoPE context contract"),
                std::string::npos);
    }
  }
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

#if USE_CUDA
TEST_F(LongRopeContextTest, CudaReplaySynchronizesReusedSequenceBeforeSelectingLogits) {
  auto config = OgaConfig::Create(MODEL_PATH "longrope-context");
  config->AppendProvider("cuda");
  model = OgaModel::Create(*config);
  for (int chunk_size : {0, 4}) {
    auto generator = CreateGenerator(chunk_size);
    auto padded_prompt = std::vector<int32_t>(10, 1);
    padded_prompt.back() = 2;
    generator->AppendTokens(padded_prompt);
    ASSERT_EQ(generator->GetSequenceData(0)[9], 2);
    generator->RewindTo(0);
    generator->AppendTokens(std::vector<int32_t>(7, 1));
    generator->AppendTokens(std::vector<int32_t>(3, 1));
    ExpectConsistentCache(*generator);
  }
}
#endif

TEST_F(LongRopeContextTest, UnsupportedBatchingFailsInsteadOfMixingCacheModes) {
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("batch_size", 2);
  EXPECT_THROW(OgaGenerator::Create(*model, *params), std::runtime_error);
  params->SetSearchOption("batch_size", 1);
  params->SetSearchOption("num_beams", 2);
  EXPECT_THROW(OgaGenerator::Create(*model, *params), std::runtime_error);
}

}  // namespace
