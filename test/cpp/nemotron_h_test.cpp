// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "ort_genai.h"

#ifndef MODEL_PATH
#define MODEL_PATH "../../test/models/"
#endif

namespace {

constexpr size_t kMaxLength = 24;

std::vector<int32_t> ExpectedTokens(const std::vector<int32_t>& prompt) {
  std::vector<int32_t> result = prompt;
  int32_t accumulator = 0;
  int32_t previous = 0;
  for (int32_t token : prompt) {
    accumulator = (accumulator + token + previous) % 7;
    previous = token;
  }
  while (result.size() < kMaxLength) {
    const int32_t token = accumulator + 1;
    result.push_back(token);
    accumulator = (accumulator + token + previous) % 7;
    previous = token;
  }
  return result;
}

std::unique_ptr<OgaModel> LoadModel(const char* component, const char* provider) {
  const auto path = std::filesystem::path(MODEL_PATH) / "nemotron-h" / component;
  auto config = OgaConfig::Create(path.string().c_str());
  config->ClearProviders();
  if (provider != nullptr) {
    config->AppendProvider(provider);
    config->SetProviderOption(provider, "enable_cuda_graph", "0");
  }
  return OgaModel::Create(*config);
}

void CheckDecoder(const char* provider) {
  auto model = LoadModel("decoder", provider);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", kMaxLength);
  params->SetSearchOptionBool("past_present_share_buffer", false);
  auto generator = OgaGenerator::Create(*model, *params);
  for (const auto& prompt : {std::vector<int32_t>{1, 2, 3}, std::vector<int32_t>{3, 1}}) {
    generator->AppendTokens(prompt);
    while (!generator->IsDone()) generator->GenerateNextToken();
    const auto actual = generator->GetSequence(0);
    EXPECT_EQ(std::vector<int32_t>(actual.begin(), actual.end()), ExpectedTokens(prompt));
    generator->RewindTo(0);
  }
}

void CheckMtp(const char* provider, int draft_tokens, int prefill_chunk = 0) {
  auto main = LoadModel("decoder", provider);
  auto head = LoadModel("mtp", provider);
  auto params = OgaGeneratorParams::Create(*main);
  params->SetSearchOption("max_length", kMaxLength);
  params->SetSearchOptionBool("past_present_share_buffer", false);
  params->SetSpeculativeNumber("max_draft_tokens", draft_tokens);
  if (prefill_chunk > 0) params->SetSearchOption("chunk_size", prefill_chunk);
  auto generator = OgaMtpGenerator::Create(*main, *head, *params);
  for (const auto& prompt : {std::vector<int32_t>{1, 2, 3}, std::vector<int32_t>{3, 1},
                             std::vector<int32_t>{3}}) {
    generator->AppendTokens(prompt.data(), prompt.size());
    while (!generator->IsDone()) generator->GenerateNextToken();
    const auto* data = generator->GetSequenceData();
    EXPECT_EQ(std::vector<int32_t>(data, data + generator->GetSequenceCount()), ExpectedTokens(prompt));
    EXPECT_GT(generator->GetAcceptCount(), 0u);
    EXPECT_GT(generator->GetTrialCount(), generator->GetAcceptCount());
    generator->Reset();
    EXPECT_EQ(generator->GetAcceptCount(), 0u);
    EXPECT_EQ(generator->GetTrialCount(), 0u);
  }
}

TEST(NemotronHybridTests, CpuMixedStateAndReset) {
  CheckDecoder(nullptr);
}

TEST(NemotronHybridTests, RejectsBeamSearchWithRecurrentState) {
  auto model = LoadModel("decoder", nullptr);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", kMaxLength);
  params->SetSearchOption("num_beams", 2);
  try {
    auto generator = OgaGenerator::Create(*model, *params);
    FAIL() << "Expected recurrent-state beam search to be rejected";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("beam reordering is not implemented"), std::string::npos);
  }
}

TEST(NemotronHybridTests, CpuMtpSingleDraftRollback) {
  CheckMtp(nullptr, 1);
}

TEST(NemotronHybridTests, CpuMtpMultipleDraftRollback) {
  CheckMtp(nullptr, 2);
}

TEST(NemotronHybridTests, CpuMtpChunkedPrefillRollback) {
  CheckMtp(nullptr, 1, 1);
  CheckMtp(nullptr, 2, 2);
}

#if USE_CUDA
TEST(NemotronHybridTests, CudaMixedStateAndReset) {
  CheckDecoder("cuda");
}

TEST(NemotronHybridTests, CudaMtpSingleDraftRollback) {
  CheckMtp("cuda", 1);
}

TEST(NemotronHybridTests, CudaMtpMultipleDraftRollback) {
  CheckMtp("cuda", 2);
}

TEST(NemotronHybridTests, CudaMtpChunkedPrefillRollback) {
  CheckMtp("cuda", 1, 1);
  CheckMtp("cuda", 2, 2);
}
#endif

}  // namespace
