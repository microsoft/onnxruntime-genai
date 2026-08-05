// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "generators.h"
#include "models/model.h"
#include "search.h"

namespace Generators {
namespace {

struct CpuSearchCheckpointTest : testing::Test {
  CpuSearchCheckpointTest() {
    config.model.vocab_size = 4;
    config.model.context_length = 8;
    config.model.eos_token_id = {3};
    config.model.pad_token_id = 0;
    config.search.batch_size = 2;
    config.search.max_length = 8;
    config.search.random_seed = 1234;

    params = std::make_shared<GeneratorParams>(config);
    search = std::make_unique<GreedySearch_Cpu>(*params);
  }

  DeviceSpan<int32_t> Tokens(std::initializer_list<int32_t> values) {
    auto tokens = params->p_device->Allocate<int32_t>(values.size());
    std::copy(values.begin(), values.end(), tokens.CpuSpan().begin());
    return tokens;
  }

  void SetLogits(std::initializer_list<float> values) {
    logits = params->p_device->Allocate<float>(values.size());
    std::copy(values.begin(), values.end(), logits.CpuSpan().begin());
    search->SetLogits(logits);
  }

  Config config;
  std::shared_ptr<GeneratorParams> params;
  std::unique_ptr<GreedySearch_Cpu> search;
  DeviceSpan<float> logits;
};

TEST_F(CpuSearchCheckpointTest, SamplingRollbackRestoresRandomState) {
  auto input = Tokens({1, 2});
  search->AppendTokens(input);
  SetLogits({1.0f, 1.0f, 1.0f, -100.0f,
             1.0f, 1.0f, 1.0f, -100.0f});

  search->SaveStateForTransaction();
  std::vector<int32_t> first_tokens;
  for (int i = 0; i < 4; ++i) {
    search->SampleTopK(3, 1.0f);
    auto next_tokens = search->GetNextTokens().CpuSpan();
    first_tokens.insert(first_tokens.end(), next_tokens.begin(), next_tokens.end());
  }
  EXPECT_EQ(search->GetSequenceLength(), 5);

  search->RestoreStateForTransaction();
  EXPECT_EQ(search->GetSequenceLength(), 1);
  EXPECT_EQ(search->GetNextTokens().CpuSpan()[0], 1);
  EXPECT_EQ(search->GetNextTokens().CpuSpan()[1], 2);

  std::vector<int32_t> retried_tokens;
  for (int i = 0; i < 4; ++i) {
    search->SampleTopK(3, 1.0f);
    auto next_tokens = search->GetNextTokens().CpuSpan();
    retried_tokens.insert(retried_tokens.end(), next_tokens.begin(), next_tokens.end());
  }
  EXPECT_EQ(retried_tokens, first_tokens);
}

TEST_F(CpuSearchCheckpointTest, RollbackRestoresDoneEosAndLengthState) {
  auto input = Tokens({1, 2});
  search->AppendTokens(input);
  auto sequence_lengths = search->GetSequenceLengths().CpuSpan();
  sequence_lengths[0] = 5;
  sequence_lengths[1] = 7;

  search->SaveStateForTransaction();

  SetLogits({0.0f, 10.0f, 0.0f, 0.0f,
             0.0f, 0.0f, 10.0f, 0.0f});
  search->SelectTop();
  EXPECT_EQ(search->GetSequenceLength(), 2);

  sequence_lengths[0] = 50;
  sequence_lengths[1] = 70;
  SetLogits({0.0f, 0.0f, 0.0f, 10.0f,
             0.0f, 0.0f, 0.0f, 10.0f});
  search->SelectTop();
  EXPECT_TRUE(search->IsDone());

  search->RestoreStateForTransaction();
  EXPECT_FALSE(search->IsDone());
  EXPECT_EQ(search->GetSequenceLength(), 1);
  EXPECT_EQ(search->GetSequence(0).CpuSpan()[0], 1);
  EXPECT_EQ(search->GetSequence(1).CpuSpan()[0], 2);
  EXPECT_EQ(search->GetNextTokens().CpuSpan()[0], 1);
  EXPECT_EQ(search->GetNextTokens().CpuSpan()[1], 2);
  EXPECT_EQ(search->GetSequenceLengths().CpuSpan()[0], 5);
  EXPECT_EQ(search->GetSequenceLengths().CpuSpan()[1], 7);

  SetLogits({0.0f, 0.0f, 0.0f, 10.0f,
             0.0f, 10.0f, 0.0f, 0.0f});
  search->SelectTop();
  EXPECT_FALSE(search->IsDone());
  EXPECT_EQ(search->GetSequenceLength(), 2);
  EXPECT_EQ(search->GetNextTokens().CpuSpan()[0], 3);
  EXPECT_EQ(search->GetNextTokens().CpuSpan()[1], 1);

  SetLogits({10.0f, 0.0f, 0.0f, 0.0f,
             0.0f, 0.0f, 0.0f, 10.0f});
  search->SelectTop();
  EXPECT_TRUE(search->IsDone());
  EXPECT_EQ(search->GetSequenceLength(), 2);
}

TEST_F(CpuSearchCheckpointTest, EnforcesOneActiveCheckpoint) {
  search->SaveStateForTransaction();
  EXPECT_THROW(search->SaveStateForTransaction(), std::logic_error);

  search->CommitStateForTransaction();
  EXPECT_THROW(search->CommitStateForTransaction(), std::logic_error);
  EXPECT_THROW(search->RestoreStateForTransaction(), std::logic_error);

  search->SaveStateForTransaction();
  search->RestoreStateForTransaction();
  EXPECT_THROW(search->RestoreStateForTransaction(), std::logic_error);
  EXPECT_THROW(search->CommitStateForTransaction(), std::logic_error);
}

#if USE_CUDA
struct CudaSearchCheckpointTest : testing::Test {
  CudaSearchCheckpointTest() {
    auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
    OverlayConfig(*config, R"({ "model": { "vocab_size": 4, "eos_token_id": 3, "pad_token_id": 0 } })");
    ClearProviders(*config);
    SetProviderOption(*config, "cuda", {}, {});
    model = CreateModel(GetOrtEnv(), std::move(config));

    params = CreateGeneratorParams(*model);
    params->search.batch_size = 2;
    params->search.max_length = 8;
    params->search.random_seed = 1234;
    search = CreateSearch(*params);
  }

  DeviceSpan<int32_t> Tokens(std::initializer_list<int32_t> values) {
    auto tokens = params->p_device->Allocate<int32_t>(values.size());
    std::copy(values.begin(), values.end(), tokens.CpuSpan().begin());
    tokens.CopyCpuToDevice();
    return tokens;
  }

  void SetLogits(std::initializer_list<float> values) {
    logits = params->p_device->Allocate<float>(values.size());
    std::copy(values.begin(), values.end(), logits.CpuSpan().begin());
    logits.CopyCpuToDevice();
    search->SetLogits(logits);
  }

  std::vector<int32_t> NextTokens() {
    auto next_tokens = search->GetNextTokens().CopyDeviceToCpu();
    return {next_tokens.begin(), next_tokens.end()};
  }

  std::shared_ptr<Model> model;
  std::shared_ptr<GeneratorParams> params;
  std::unique_ptr<Search> search;
  DeviceSpan<float> logits;
};

TEST_F(CudaSearchCheckpointTest, SamplingRollbackRestoresCurandStateAfterSelectTop) {
  auto input = Tokens({1, 2});
  search->AppendTokens(input);
  SetLogits({1.0f, 1.0f, 1.0f, -100.0f,
             1.0f, 1.0f, 1.0f, -100.0f});

  search->SaveStateForTransaction();
  std::vector<int32_t> first_tokens;
  for (int i = 0; i < 3; ++i) {
    search->SelectTop();
    search->SampleTopK(3, 1.0f);
    const auto next_tokens = NextTokens();
    first_tokens.insert(first_tokens.end(), next_tokens.begin(), next_tokens.end());
  }
  EXPECT_EQ(search->GetSequenceLength(), 7);

  search->RestoreStateForTransaction();
  EXPECT_EQ(search->GetSequenceLength(), 1);
  EXPECT_EQ(NextTokens(), std::vector<int32_t>({0, 0}));

  std::vector<int32_t> retried_tokens;
  for (int i = 0; i < 3; ++i) {
    search->SelectTop();
    search->SampleTopK(3, 1.0f);
    const auto next_tokens = NextTokens();
    retried_tokens.insert(retried_tokens.end(), next_tokens.begin(), next_tokens.end());
  }
  EXPECT_EQ(retried_tokens, first_tokens);
}

TEST_F(CudaSearchCheckpointTest, BatchedSamplerRollbackRestoresRequestRandomStates) {
  auto sampler = model->p_device_scoring_->CreateBatchedSampler(
      2, params->config.model.vocab_size);
  ASSERT_NE(sampler, nullptr);
  ASSERT_TRUE(sampler->SupportsTransactions());

  auto first_state = sampler->CreateState(1234);
  auto second_state = sampler->CreateState(5678);
  std::array<BatchedSamplerState*, 2> states{
      first_state.get(), second_state.get()};
  std::array<BatchedSamplingParams, 2> sampling_params{
      BatchedSamplingParams{3, 1.0f, 1.0f},
      BatchedSamplingParams{3, 1.0f, 1.0f}};

  auto scores = params->p_device->Allocate<float>(8);
  const std::array<float, 8> score_values{
      1.0f, 1.0f, 1.0f, -100.0f,
      1.0f, 1.0f, 1.0f, -100.0f};
  std::copy(score_values.begin(), score_values.end(),
            scores.CpuSpan().begin());
  scores.CopyCpuToDevice();
  std::array<DeviceSpan<float>, 2> rows{
      scores.subspan(0, 4), scores.subspan(4, 4)};

  sampler->SaveStateForTransaction(states);
  auto first_tokens = sampler->Sample(
      rows, sampling_params, states, params->config.model.vocab_size);
  const auto first = first_tokens.CopyDeviceToCpu();
  sampler->RestoreStateForTransaction();

  std::copy(score_values.begin(), score_values.end(),
            scores.CpuSpan().begin());
  scores.CopyCpuToDevice();
  auto retried_tokens = sampler->Sample(
      rows, sampling_params, states, params->config.model.vocab_size);
  const auto retried = retried_tokens.CopyDeviceToCpu();

  EXPECT_EQ(std::vector<int32_t>(retried.begin(), retried.end()),
            std::vector<int32_t>(first.begin(), first.end()));
}

TEST_F(CudaSearchCheckpointTest, RollbackRestoresDoneEosAndLengthState) {
  auto input = Tokens({1, 2});
  search->AppendTokens(input);
  auto sequence_lengths = search->GetSequenceLengths();
  sequence_lengths.CpuSpan()[0] = 5;
  sequence_lengths.CpuSpan()[1] = 7;
  sequence_lengths.CopyCpuToDevice();

  search->SaveStateForTransaction();
  SetLogits({0.0f, 10.0f, 0.0f, 0.0f,
             0.0f, 0.0f, 10.0f, 0.0f});
  search->SelectTop();
  EXPECT_EQ(search->GetSequenceLength(), 2);

  sequence_lengths.CpuSpan()[0] = 50;
  sequence_lengths.CpuSpan()[1] = 70;
  sequence_lengths.CopyCpuToDevice();
  SetLogits({0.0f, 0.0f, 0.0f, 10.0f,
             0.0f, 0.0f, 0.0f, 10.0f});
  search->SelectTop();
  EXPECT_TRUE(search->IsDone());

  search->RestoreStateForTransaction();
  EXPECT_FALSE(search->IsDone());
  EXPECT_EQ(search->GetSequenceLength(), 1);
  EXPECT_EQ(NextTokens(), std::vector<int32_t>({0, 0}));
  auto restored_sequence_lengths = search->GetSequenceLengths().CopyDeviceToCpu();
  EXPECT_EQ(restored_sequence_lengths[0], 5);
  EXPECT_EQ(restored_sequence_lengths[1], 7);

  SetLogits({0.0f, 0.0f, 0.0f, 10.0f,
             0.0f, 10.0f, 0.0f, 0.0f});
  search->SelectTop();
  EXPECT_FALSE(search->IsDone());
  EXPECT_EQ(search->GetSequenceLength(), 2);
  EXPECT_EQ(NextTokens(), std::vector<int32_t>({3, 1}));

  SetLogits({10.0f, 0.0f, 0.0f, 0.0f,
             0.0f, 0.0f, 0.0f, 10.0f});
  search->SelectTop();
  EXPECT_TRUE(search->IsDone());
  EXPECT_EQ(search->GetSequenceLength(), 2);
}
#endif

}  // namespace
}  // namespace Generators
