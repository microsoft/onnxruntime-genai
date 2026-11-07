// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
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

  void ExpectRecordedChunkSizes(OgaGenerator& generator, size_t prefix_length, size_t chunk_size) {
    const auto count = generator.GetSequenceCount(0);
    auto tensor = generator.GetOutput("present.0.value");
    const std::vector<int64_t> shape{1, 1, static_cast<int64_t>(count), 1};
    ASSERT_EQ(tensor->Shape(), shape);
    const auto* sizes = static_cast<const float*>(tensor->Data());
    for (size_t i = 0; i < count; ++i) {
      const size_t segment_length = i < prefix_length ? prefix_length : count - prefix_length;
      const size_t segment_index = i < prefix_length ? i : i - prefix_length;
      const size_t expected = chunk_size ? std::min(chunk_size, segment_length - segment_index / chunk_size * chunk_size)
                                         : segment_length;
      EXPECT_FLOAT_EQ(sizes[i], static_cast<float>(expected)) << "token=" << i;
    }
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

TEST_F(LongRopeContextTest, ReplayPreservesAppendedPromptPaddingSelection) {
  std::vector<bool> cuda_modes{false};
#if USE_CUDA
  cuda_modes.push_back(true);
#endif
  for (bool use_cuda : cuda_modes) {
    SCOPED_TRACE(testing::Message() << "use_cuda=" << use_cuda);
    for (const auto* path : {MODEL_PATH "longrope-context", MODEL_PATH "longrope-context-runtime-indices"}) {
      auto config = OgaConfig::Create(path);
      config->Overlay(R"({"model":{"pad_token_id":1,"eos_token_id":2}})");
      if (use_cuda)
        config->AppendProvider("cuda");
      model = OgaModel::Create(*config);
      for (int chunk : {0, 1, 2, 3, 4}) {
        for (bool committed_pad : {false, true}) {
          SCOPED_TRACE(testing::Message() << "path=" << path << " chunk=" << chunk << " committed_pad=" << committed_pad);
          auto generator = CreateGenerator(chunk);
          generator->AppendTokens(std::vector<int32_t>(committed_pad ? 7 : 8, 0));
          if (committed_pad) {
            std::array<float, 3> forced_logits{0, 10, -10};
            generator->SetLogits(*OgaTensor::Create(forced_logits.data(), std::array<int64_t, 2>{1, 3}));
            generator->GenerateNextToken();
            ASSERT_EQ(generator->GetSequenceData(0)[7], 1);
          }
          ASSERT_EQ(generator->GetSequenceCount(0), 8u);
          generator->AppendTokens(std::vector<int32_t>{0, 1});
          ASSERT_EQ(generator->GetSequenceCount(0), 10u);
          EXPECT_EQ(generator->GetSequenceData(0)[9], 1);
          auto tensor = generator->GetLogits();
          const auto* logits = static_cast<const float*>(tensor->Data());
          const float expected = committed_pad ? 2.0f : 0.0f;
          EXPECT_FLOAT_EQ(logits[0], expected);
          EXPECT_FLOAT_EQ(logits[1], -expected);
          EXPECT_FLOAT_EQ(logits[2], 0);
          generator->GenerateNextToken();
          EXPECT_EQ(generator->GetSequenceData(0)[10], 0);
          ExpectConsistentCache(*generator);
        }
      }
    }
  }
}

TEST_F(LongRopeContextTest, ReplayKeepsAppendedPromptWithinChunkBudget) {
  std::vector<bool> cuda_modes{false};
#if USE_CUDA
  cuda_modes.push_back(true);
#endif
  for (bool use_cuda : cuda_modes) {
    for (const auto* path : {MODEL_PATH "longrope-context-chunk-sizes",
                             MODEL_PATH "longrope-context-runtime-indices-chunk-sizes"}) {
      for (int padding : {0, 1, 7}) {
        auto config = OgaConfig::Create(path);
        if (padding)
          config->Overlay(R"({"model":{"pad_token_id":1,"eos_token_id":2}})");
        if (use_cuda)
          config->AppendProvider("cuda");
        model = OgaModel::Create(*config);
        for (int chunk : {0, 1, 2, 3, 4}) {
          for (int suffix_length : {16, 17}) {
            SCOPED_TRACE(testing::Message() << "use_cuda=" << use_cuda << " path=" << path
                                            << " padding=" << padding << " chunk=" << chunk
                                            << " suffix_length=" << suffix_length);
            auto generator = CreateGenerator(chunk);
            std::vector<int32_t> prefix(8, padding ? 0 : 1);
            std::vector<int32_t> suffix(suffix_length, padding ? 0 : 1);
            if (padding)
              std::fill(suffix.end() - padding, suffix.end(), 1);
            generator->AppendTokens(prefix);
            generator->AppendTokens(suffix);
            std::vector<int32_t> expected = prefix;
            expected.insert(expected.end(), suffix.begin(), suffix.end());
            ASSERT_EQ(generator->GetSequenceCount(0), expected.size());
            EXPECT_EQ(std::vector<int32_t>(generator->GetSequenceData(0),
                                           generator->GetSequenceData(0) + expected.size()),
                      expected);
            ExpectRecordedChunkSizes(*generator, prefix.size(), static_cast<size_t>(chunk));
            auto tensor = generator->GetLogits();
            const auto* logits = static_cast<const float*>(tensor->Data());
            const float sum = padding ? 0.0f : static_cast<float>(expected.size()) * 2.0f;
            EXPECT_FLOAT_EQ(logits[0], sum);
            EXPECT_FLOAT_EQ(logits[1], -sum);
            EXPECT_FLOAT_EQ(logits[2], 0);
            generator->GenerateNextToken();
            EXPECT_EQ(generator->GetSequenceData(0)[expected.size()], 0);
            ExpectConsistentCache(*generator);
            auto cache = generator->GetOutput("present.0.value");
            EXPECT_FLOAT_EQ(static_cast<const float*>(cache->Data())[expected.size()], 1.0f);
          }
        }
      }
    }
  }
}

TEST_F(LongRopeContextTest, ReplayRejectsPaddingOnlyPromptBeforeRebuildingCache) {
  std::vector<bool> cuda_modes{false};
#if USE_CUDA
  cuda_modes.push_back(true);
#endif
  for (bool use_cuda : cuda_modes) {
    for (const auto* path : {MODEL_PATH "longrope-context", MODEL_PATH "longrope-context-runtime-indices"}) {
      auto config = OgaConfig::Create(path);
      config->Overlay(R"({"model":{"pad_token_id":0,"eos_token_id":2}})");
      if (use_cuda)
        config->AppendProvider("cuda");
      model = OgaModel::Create(*config);
      for (int chunk : {0, 1, 4}) {
        SCOPED_TRACE(testing::Message() << "use_cuda=" << use_cuda << " path=" << path << " chunk=" << chunk);
        auto generator = CreateGenerator(chunk);
        generator->AppendTokens(std::vector<int32_t>(8, 1));
        try {
          generator->AppendTokens(std::vector<int32_t>(4, 0));
          ADD_FAILURE() << "Accepted a padding-only replayed prompt";
        } catch (const std::runtime_error& error) {
          EXPECT_NE(std::string(error.what()).find("non-padding token"), std::string::npos);
        }
        ASSERT_EQ(generator->GetSequenceCount(0), 8u);
        for (size_t i = 0; i < 8; ++i)
          EXPECT_EQ(generator->GetSequenceData(0)[i], 1);
        ExpectConsistentCache(*generator);
        auto cache = generator->GetOutput("present.0.key");
        const std::vector<int64_t> shape{1, 1, 8, 1};
        ASSERT_EQ(cache->Shape(), shape);
        const auto* values = static_cast<const float*>(cache->Data());
        for (size_t i = 0; i < 8; ++i)
          EXPECT_FLOAT_EQ(values[i], 1.0f);
        generator->AppendTokens(std::vector<int32_t>{1});
        ASSERT_EQ(generator->GetSequenceCount(0), 9u);
        ExpectConsistentCache(*generator);
      }
    }
  }
}

TEST_F(LongRopeContextTest, LegacyThresholdReplayPreservesGeneratedPadOrigin) {
  for (const auto* path : {MODEL_PATH "longrope-context-legacy", MODEL_PATH "longrope-context-legacy-runtime-indices"}) {
    auto config = OgaConfig::Create(path);
    config->Overlay(R"({"model":{"type":"phi3","context_length":8192,"pad_token_id":1,"eos_token_id":2},
                       "search":{"max_length":8192}})");
    model = OgaModel::Create(*config);
    for (int chunk : {0, 1, 4}) {
      SCOPED_TRACE(testing::Message() << "path=" << path << " chunk=" << chunk);
      auto generator = CreateGenerator(chunk);
      generator->AppendTokens(std::vector<int32_t>(4096, 0));
      std::array<float, 3> forced_logits{0, 10, -10};
      generator->SetLogits(*OgaTensor::Create(forced_logits.data(), std::array<int64_t, 2>{1, 3}));
      generator->GenerateNextToken();
      ASSERT_EQ(generator->GetSequenceCount(0), 4097u);
      ASSERT_EQ(generator->GetSequenceData(0)[4096], 1);
      ASSERT_NO_THROW(generator->GenerateNextToken());
      ASSERT_EQ(generator->GetSequenceCount(0), 4098u);
      EXPECT_EQ(generator->GetSequenceData(0)[4097], 1);
    }
  }
}

TEST_F(LongRopeContextTest, SpeculativeLegacyThresholdReplayPreservesGeneratedPadOrigin) {
  auto config = OgaConfig::Create(MODEL_PATH "longrope-context-legacy");
  config->Overlay(R"({"model":{"type":"phi3","context_length":8192,"pad_token_id":1,"eos_token_id":2,
                    "draft":{"filename":"draft.onnx","num_hidden_layers":1,"num_attention_heads":1,
                             "num_key_value_heads":1,"head_size":1,"hidden_size":1,
                             "session_options":{"intra_op_num_threads":1}}},
                    "search":{"max_length":8192},"speculative":{"max_draft_tokens":1}})");
  std::ifstream input(MODEL_PATH "longrope-context-legacy/model.onnx", std::ios::binary);
  ASSERT_TRUE(input);
  const std::string graph{std::istreambuf_iterator<char>{input}, {}};
  config->AddModelData("draft.onnx", graph.data(), graph.size());
  model = OgaModel::Create(*config);
  auto generator = CreateGenerator(4);
  generator->AppendTokens(std::vector<int32_t>(4096, 0));
  std::array<float, 3> forced_logits{0, 10, -10};
  generator->SetLogits(*OgaTensor::Create(forced_logits.data(), std::array<int64_t, 2>{1, 3}));
  generator->GenerateNextToken();
  ASSERT_EQ(generator->GetSequenceCount(0), 4097u);
  ASSERT_EQ(generator->GetSequenceData(0)[4096], 1);
  ASSERT_NO_THROW(generator->GenerateNextToken());
  ASSERT_EQ(generator->GetSequenceCount(0), 4098u);
  EXPECT_EQ(generator->GetSequenceData(0)[4097], 1);
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
      ASSERT_EQ(generator->GetSequenceCount(0), static_cast<size_t>(length));
      ExpectConsistentCache(*generator);
      generator->AppendTokens(std::vector<int32_t>{1});
      ASSERT_EQ(generator->GetSequenceCount(0), static_cast<size_t>(length + 1));
      ExpectConsistentCache(*generator);

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
