// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <string>
#include <vector>
#include "ort_genai.h"

namespace {

auto LoadMarianCacheModel(const char* fixture) {
  return OgaModel::Create((std::string(MODEL_PATH "marian-cache/") + fixture).c_str());
}

auto CreateGenerator(OgaModel& model, const std::array<int32_t, 4>& tokens) {
  auto params = OgaGeneratorParams::Create(model);
  params->SetSearchOption("batch_size", 2);
  auto generator = OgaGenerator::Create(model, *params);
  generator->AppendTokens(tokens.data(), tokens.size());
  return generator;
}

void CheckCacheTokens(OgaGenerator& generator, const std::array<int32_t, 4>& tokens,
                      int step, int count = 6, int width = 4) {
  generator.GenerateNextToken();
  for (size_t row = 0; row < 2; ++row) {
    int sum = 0;
    for (int i = 0; i < count; ++i)
      sum += width * (tokens[2 * row] + tokens[2 * row + 1] + 3 * i) * (i + 1);
    const int expected = (sum + step) % 100 + 1;
    ASSERT_EQ(generator.GetSequenceCount(row), 3U + step);
    EXPECT_EQ(generator.GetSequenceData(row)[2 + step], expected);
  }
}

TEST(MarianCacheTests, ValuesPersistAcrossStepsAndGenerators) {
  for (const char* fixture : {"dynamic", "static", "fp16", "uncached"}) {
    SCOPED_TRACE(fixture);
    auto model = LoadMarianCacheModel(fixture);
    const std::array<int32_t, 4> first{2, 3, 4, 5};
    const std::array<int32_t, 4> second{6, 7, 8, 9};
    auto generator1 = CreateGenerator(*model, first);
    auto generator2 = CreateGenerator(*model, second);
    const bool fp16 = std::string(fixture) == "fp16";
    const int count = fp16 ? 3 : 6;
    const int width = fp16 ? 2 : 4;
    for (int step = 0; step < 3; ++step) {
      CheckCacheTokens(*generator1, first, step, count, width);
      CheckCacheTokens(*generator2, second, step, count, width);
    }
    generator1.reset();
    CheckCacheTokens(*generator2, second, 3, count, width);
    auto replacement = CreateGenerator(*model, first);
    CheckCacheTokens(*replacement, first, 0, count, width);
  }
}

TEST(MarianCacheTests, PaddedRowsPreserveParityAfterEarlyEos) {
  auto cached_model = LoadMarianCacheModel("ragged");
  auto uncached_model = LoadMarianCacheModel("ragged-uncached");
  const std::array<std::array<int32_t, 4>, 2> expected{{{65, 0, 127, 127}, {21, 22, 23, 0}}};
  for (bool reversed : {false, true}) {
    SCOPED_TRACE(reversed);
    const std::array<int32_t, 6> tokens = reversed
                                              ? std::array<int32_t, 6>{4, 5, 6, 2, 127, 127}
                                              : std::array<int32_t, 6>{2, 127, 127, 4, 5, 6};
    auto cached_params = OgaGeneratorParams::Create(*cached_model);
    auto uncached_params = OgaGeneratorParams::Create(*uncached_model);
    cached_params->SetSearchOption("batch_size", 2);
    uncached_params->SetSearchOption("batch_size", 2);
    auto cached = OgaGenerator::Create(*cached_model, *cached_params);
    auto uncached = OgaGenerator::Create(*uncached_model, *uncached_params);
    cached->AppendTokens(tokens.data(), tokens.size());
    uncached->AppendTokens(tokens.data(), tokens.size());
    for (size_t step = 0; step < 4; ++step) {
      SCOPED_TRACE(step);
      ASSERT_FALSE(cached->IsDone());
      ASSERT_FALSE(uncached->IsDone());
      cached->GenerateNextToken();
      uncached->GenerateNextToken();
      const auto cached_tokens = cached->GetNextTokens();
      const auto uncached_tokens = uncached->GetNextTokens();
      ASSERT_EQ(cached_tokens.size(), 2U);
      ASSERT_EQ(uncached_tokens.size(), 2U);
      for (size_t row = 0; row < 2; ++row) {
        SCOPED_TRACE(row);
        // The final all-EOS step is not appended to the sequences.
        ASSERT_EQ(cached->GetSequenceCount(row), step == 3 ? 6U : 4U + step);
        ASSERT_EQ(uncached->GetSequenceCount(row), cached->GetSequenceCount(row));
        for (size_t i = 0; i < cached->GetSequenceCount(row); ++i)
          EXPECT_EQ(cached->GetSequenceData(row)[i], uncached->GetSequenceData(row)[i]);
        EXPECT_EQ(cached_tokens[row], expected[reversed ? 1 - row : row][step]);
        EXPECT_EQ(cached_tokens[row], uncached_tokens[row]);
      }
      EXPECT_EQ(cached->IsDone(), step == 3);
      EXPECT_EQ(uncached->IsDone(), step == 3);
    }
  }
}

TEST(MarianCacheTests, GreedyEquivalentSampling) {
  auto model = LoadMarianCacheModel("dynamic");
  for (int mode = 0; mode < 3; ++mode) {
    SCOPED_TRACE(mode);
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("batch_size", 2);
    params->SetSearchOptionBool("do_sample", mode != 0);
    params->SetSearchOption("top_k", mode == 1 ? 1 : 5);
    params->SetSearchOption("temperature", mode == 2 ? 0 : 1);
    auto generator = OgaGenerator::Create(*model, *params);
    const std::array<int32_t, 4> tokens{2, 3, 4, 5};
    generator->AppendTokens(tokens.data(), tokens.size());
    CheckCacheTokens(*generator, tokens, 0);
    CheckCacheTokens(*generator, tokens, 1);
  }
}

TEST(MarianCacheTests, RejectsSamplingAndBeams) {
  auto model = LoadMarianCacheModel("dynamic");
  for (bool sampling : {false, true}) {
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("num_beams", sampling ? 1 : 2);
    params->SetSearchOptionBool("do_sample", sampling);
    params->SetSearchOption("top_k", 5);
    EXPECT_THROW(OgaGenerator::Create(*model, *params), std::runtime_error);
  }
}

TEST(MarianCacheTests, GeneratorsHaveIndependentDynamicShapes) {
  auto model = LoadMarianCacheModel("dynamic");
  const std::array<int32_t, 4> batched_tokens{2, 3, 4, 5};
  auto batch = CreateGenerator(*model, batched_tokens);
  auto params = OgaGeneratorParams::Create(*model);
  auto scalar = OgaGenerator::Create(*model, *params);
  const int32_t token = 8;
  scalar->AppendTokens(&token, 1);
  for (int step = 0; step < 3; ++step) {
    CheckCacheTokens(*batch, batched_tokens, step);
    scalar->GenerateNextToken();
    int sum = 0;
    for (int i = 0; i < 6; ++i)
      sum += 4 * (token + 2 * i) * (i + 1);
    ASSERT_EQ(scalar->GetSequenceCount(0), 2U + step);
    EXPECT_EQ(scalar->GetSequenceData(0)[1 + step], (sum + step) % 100 + 1);
  }
}

auto CreateRecurrentGenerator(OgaModel& model, const std::vector<int32_t>& source, bool chunked = false) {
  auto params = OgaGeneratorParams::Create(model);
  params->SetSearchOption("max_length", 64);
  auto generator = OgaGenerator::Create(model, *params);
  if (chunked) {
    generator->AppendTokens(source.data(), 2);
    generator->AppendTokens(source.data() + 2, source.size() - 2);
  } else {
    generator->AppendTokens(source.data(), source.size());
  }
  return generator;
}

void GenerateSteps(OgaGenerator& generator, size_t count) {
  for (size_t step = 0; step < count; ++step)
    generator.GenerateNextToken();
}

std::vector<float> CopyMarianLogits(OgaGenerator& generator) {
  auto tensor = generator.GetLogits();
  size_t count = 1;
  for (auto dimension : tensor->Shape())
    count *= static_cast<size_t>(dimension);
  const auto* data = static_cast<const float*>(tensor->Data());
  return {data, data + count};
}

void ForceMarianToken(OgaGenerator& generator, int32_t token, int32_t fallback = 0) {
  auto tensor = generator.GetLogits();
  auto* data = static_cast<float*>(tensor->Data());
  std::fill_n(data, 128, -1000.0f);
  data[fallback] = 500.0f;
  data[token] = 1000.0f;
  generator.SetLogits(*tensor);
  generator.GenerateNextToken();
}

TEST(MarianCacheTests, BeamPrefillChunksPreserveHistoryAndLogits) {
  auto config = OgaConfig::Create(MODEL_PATH "marian-cache/uncached");
  config->ClearProviders();
#if USE_CUDA
  config->AppendProvider("cuda");
#endif
  auto model = OgaModel::Create(*config);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 12);
  params->SetSearchOption("num_beams", 2);
  params->SetSearchOption("num_return_sequences", 2);
  const std::array<int32_t, 4> source{2, 3, 4, 5};
  auto reference = OgaGenerator::Create(*model, *params);
  reference->AppendTokens(source.data(), source.size());
  auto chunked = OgaGenerator::Create(*model, *params);
  chunked->AppendTokens(source.data(), 2);
  chunked->AppendTokens(source.data() + 2, 1);
  chunked->AppendTokens(source.data() + 3, 1);
  EXPECT_EQ(CopyMarianLogits(*chunked), CopyMarianLogits(*reference));
  while (!reference->IsDone()) {
    ASSERT_FALSE(chunked->IsDone());
    EXPECT_EQ(CopyMarianLogits(*chunked), CopyMarianLogits(*reference));
    reference->GenerateNextToken();
    chunked->GenerateNextToken();
  }
  EXPECT_TRUE(chunked->IsDone());
  for (size_t beam = 0; beam < 2; ++beam) {
    const auto* expected = reference->GetSequenceData(beam);
    const std::vector<int32_t> history(expected, expected + reference->GetSequenceCount(beam));
    ASSERT_GE(history.size(), source.size());
    EXPECT_TRUE(std::equal(source.begin(), source.end(), history.begin()));
    ASSERT_EQ(chunked->GetSequenceCount(beam), history.size());
    EXPECT_TRUE(std::equal(history.begin(), history.end(), chunked->GetSequenceData(beam)));
  }
}

#if USE_CUDA
TEST(MarianCacheTests, EncoderRewindWithCudaMatchesFreshSource) {
  auto config = OgaConfig::Create(MODEL_PATH "marian-cache/uncached");
  config->ClearProviders();
  config->AppendProvider("cuda");
  auto model = OgaModel::Create(*config);
  const std::vector<int32_t> source{2, 3, 4, 5, 6, 7};
  for (size_t retained : {1U, 3U, 5U}) {
    SCOPED_TRACE(retained);
    auto generator = CreateRecurrentGenerator(*model, source);
    GenerateSteps(*generator, 8);
    generator->RewindTo(retained);
    const std::vector<int32_t> prefix(source.begin(), source.begin() + retained + 1);
    auto reference = CreateRecurrentGenerator(*model, prefix);
    EXPECT_EQ(CopyMarianLogits(*generator), CopyMarianLogits(*reference));
    GenerateSteps(*generator, 2);
    GenerateSteps(*reference, 2);
    EXPECT_EQ(generator->GetNextTokens()[0], reference->GetNextTokens()[0]);
  }
}
#endif

TEST(MarianCacheTests, CheckpointRestoresDecoderStateAndPosition) {
  auto model = LoadMarianCacheModel("recurrent");
  const std::vector<int32_t> source{2, 3, 4, 5};
  for (bool chunked : {false, true}) {
    for (size_t retained : {0U, 1U, 3U, 24U}) {
      SCOPED_TRACE(chunked);
      SCOPED_TRACE(retained);
      auto reference = CreateRecurrentGenerator(*model, source);
      auto generator = CreateRecurrentGenerator(*model, source, chunked);
      GenerateSteps(*reference, retained + 1);
      GenerateSteps(*generator, retained + 8);
      generator->RewindTo(source.size() + retained);
      EXPECT_EQ(CopyMarianLogits(*generator), CopyMarianLogits(*reference));
      for (size_t step = 0; step < 3; ++step) {
        generator->GenerateNextToken();
        reference->GenerateNextToken();
        EXPECT_EQ(generator->GetNextTokens()[0], reference->GetNextTokens()[0]);
      }
    }
  }
}

TEST(MarianCacheTests, CheckpointRepeatedRewindsDiscardChangedFuture) {
  auto model = LoadMarianCacheModel("recurrent");
  const std::vector<int32_t> source{2, 3, 4, 5};
  auto generator = CreateRecurrentGenerator(*model, source);
  GenerateSteps(*generator, 10);
  for (size_t retained : {4U, 1U, 3U, 0U}) {
    SCOPED_TRACE(retained);
    const auto* data = generator->GetSequenceData(0);
    const std::vector<int32_t> history(data, data + generator->GetSequenceCount(0));
    auto reference = CreateRecurrentGenerator(*model, source);
    for (size_t index = 0; index <= retained; ++index)
      ForceMarianToken(*reference, history[source.size() + index]);
    generator->RewindTo(source.size() + retained);
    EXPECT_EQ(CopyMarianLogits(*generator), CopyMarianLogits(*reference));
    ForceMarianToken(*generator, 42);
    ForceMarianToken(*reference, 42);
    EXPECT_EQ(CopyMarianLogits(*generator), CopyMarianLogits(*reference));
    GenerateSteps(*generator, 8);
  }
}

TEST(MarianCacheTests, CheckpointEncoderRewindRecomputesRetainedSource) {
  auto model = LoadMarianCacheModel("recurrent");
  const std::vector<int32_t> source{2, 3, 4, 5, 6, 7};
  for (bool chunked : {false, true}) {
    for (size_t retained : {1U, 3U, 5U}) {
      SCOPED_TRACE(chunked);
      SCOPED_TRACE(retained);
      auto generator = CreateRecurrentGenerator(*model, source, chunked);
      GenerateSteps(*generator, 8);
      generator->RewindTo(retained);
      const std::vector<int32_t> prefix(source.begin(), source.begin() + retained + 1);
      auto reference = CreateRecurrentGenerator(*model, prefix);
      EXPECT_EQ(CopyMarianLogits(*generator), CopyMarianLogits(*reference));
      GenerateSteps(*generator, 2);
      GenerateSteps(*reference, 2);
      EXPECT_EQ(generator->GetNextTokens()[0], reference->GetNextTokens()[0]);
    }
  }
}

TEST(MarianCacheTests, CheckpointRejectsMissingStateWithoutChangingGenerator) {
  auto model = LoadMarianCacheModel("recurrent");
  const std::vector<int32_t> source{2, 3, 4, 5};
  auto generator = CreateRecurrentGenerator(*model, source);
  generator->GenerateNextToken();
  const int32_t continuation = 42;
  generator->AppendTokens(&continuation, 1);
  const auto expected = CopyMarianLogits(*generator);
  const auto* data = generator->GetSequenceData(0);
  const std::vector<int32_t> history(data, data + generator->GetSequenceCount(0));
  EXPECT_THROW(generator->RewindTo(source.size() + 1), std::runtime_error);
  ASSERT_EQ(generator->GetSequenceCount(0), history.size());
  EXPECT_TRUE(std::equal(history.begin(), history.end(), generator->GetSequenceData(0)));
  EXPECT_EQ(CopyMarianLogits(*generator), expected);
}

TEST(MarianCacheTests, CheckpointResetNoOpAndContinuationMatchFreshGenerator) {
  auto model = LoadMarianCacheModel("recurrent");
  const std::vector<int32_t> source{2, 3, 4, 5};
  auto generator = CreateRecurrentGenerator(*model, source);
  GenerateSteps(*generator, 8);
  const auto expected = CopyMarianLogits(*generator);
  generator->RewindTo(generator->TokenCount());
  EXPECT_EQ(CopyMarianLogits(*generator), expected);
  const std::vector<int32_t> replacement{6, 7, 8};
  generator->RewindTo(0);
  generator->AppendTokens(replacement.data(), replacement.size());
  auto reference = CreateRecurrentGenerator(*model, replacement);
  GenerateSteps(*generator, 4);
  GenerateSteps(*reference, 4);
  EXPECT_EQ(CopyMarianLogits(*generator), CopyMarianLogits(*reference));
  const int32_t continuation = 42;
  generator->AppendTokens(&continuation, 1);
  reference->AppendTokens(&continuation, 1);
  GenerateSteps(*generator, 8);
  GenerateSteps(*reference, 2);
  generator->RewindTo(replacement.size() + 6);
  EXPECT_EQ(CopyMarianLogits(*generator), CopyMarianLogits(*reference));
}

TEST(MarianCacheTests, CheckpointPreservesDecoderNgramHistory) {
  auto model = LoadMarianCacheModel("recurrent");
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 32);
  params->SetSearchOption("no_repeat_ngram_size", 2);
  auto generator = OgaGenerator::Create(*model, *params);
  const std::array<int32_t, 2> source{3, 3};
  generator->AppendTokens(source.data(), source.size());
  ForceMarianToken(*generator, 3, 42);
  ForceMarianToken(*generator, 3, 42);
  std::ignore = generator->GetLogits();
  generator->RewindTo(source.size());
  std::ignore = generator->GetLogits();
  ForceMarianToken(*generator, 3, 42);
  EXPECT_EQ(generator->GetNextTokens()[0], 3);
  ForceMarianToken(*generator, 3, 42);
  EXPECT_EQ(generator->GetNextTokens()[0], 42);
}

TEST(MarianCacheTests, CheckpointBatchResetAndNoOpPreserveState) {
  auto model = LoadMarianCacheModel("recurrent");
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("batch_size", 2);
  auto generator = OgaGenerator::Create(*model, *params);
  const std::array<int32_t, 4> source{2, 3, 4, 5};
  generator->AppendTokens(source.data(), source.size());
  GenerateSteps(*generator, 3);
  const auto expected = CopyMarianLogits(*generator);
  const size_t length = generator->TokenCount();
  generator->RewindTo(length);
  EXPECT_EQ(CopyMarianLogits(*generator), expected);
  EXPECT_THROW(generator->RewindTo(length - 1), std::runtime_error);
  EXPECT_EQ(generator->TokenCount(), length);
  EXPECT_EQ(CopyMarianLogits(*generator), expected);
  generator->RewindTo(0);
  const std::array<int32_t, 6> replacement{6, 7, 8, 9, 10, 11};
  generator->AppendTokens(replacement.data(), replacement.size());
  auto reference = OgaGenerator::Create(*model, *params);
  reference->AppendTokens(replacement.data(), replacement.size());
  GenerateSteps(*generator, 3);
  GenerateSteps(*reference, 3);
  EXPECT_EQ(CopyMarianLogits(*generator), CopyMarianLogits(*reference));
}

TEST(MarianCacheTests, CheckpointAtMaximumLengthCanBeReplayed) {
  auto model = LoadMarianCacheModel("recurrent");
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 12);
  auto generator = OgaGenerator::Create(*model, *params);
  const std::array<int32_t, 4> source{2, 3, 4, 5};
  generator->AppendTokens(source.data(), source.size());
  GenerateSteps(*generator, 8);
  const auto expected = CopyMarianLogits(*generator);
  generator->RewindTo(11);
  EXPECT_EQ(CopyMarianLogits(*generator), expected);
  EXPECT_EQ(generator->TokenCount(), 12U);
  EXPECT_THROW(generator->GenerateNextToken(), std::runtime_error);
  EXPECT_EQ(CopyMarianLogits(*generator), expected);
}

class MarianCacheContractTests : public testing::TestWithParam<const char*> {};

TEST_P(MarianCacheContractTests, RejectsInvalidInterfaceAtLoad) {
  try {
    auto model = LoadMarianCacheModel(GetParam());
    FAIL() << "Accepted malformed cache interface: " << GetParam();
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("cached_source_projection_0"), std::string::npos) << message;
    if (std::string(GetParam()) == "batch" || std::string(GetParam()) == "source" ||
        std::string(GetParam()) == "width" || std::string(GetParam()) == "rank" ||
        std::string(GetParam()) == "symbols" || std::string(GetParam()) == "static-dynamic") {
      EXPECT_NE(message.find("encoder"), std::string::npos) << message;
      EXPECT_NE(message.find("decoder"), std::string::npos) << message;
      EXPECT_NE(message.find("["), std::string::npos) << message;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    MalformedModels, MarianCacheContractTests,
    testing::Values("missing-output", "missing-input", "wrong-sessions", "encoder-input", "decoder-output",
                    "type", "unsupported-type", "batch", "source", "width", "rank", "symbols", "static-dynamic"));

TEST(MarianCacheTests, ChecksStaticDimensionsBeforeAllocation) {
  auto model = LoadMarianCacheModel("static");
  auto params = OgaGeneratorParams::Create(*model);
  auto generator = OgaGenerator::Create(*model, *params);
  const std::array<int32_t, 2> tokens{2, 3};
  try {
    generator->AppendTokens(tokens.data(), tokens.size());
    generator->GenerateNextToken();
    FAIL() << "Accepted a one-row request for a two-row cache";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("cached_source_projection_0"), std::string::npos) << error.what();
  }
}

}  // namespace
