// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "generator/generators.h"
#include "models/model.h"
#include "models/state-checkpoint.h"
#include "test_utils.h"

namespace Generators::test {
namespace {

std::shared_ptr<Model> CheckpointModel(bool shared = true) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-qwen35-cuda");
  ClearProviders(*config);
  config->search.past_present_share_buffer = shared;
  config->search.max_length = 32;
  // This fixture advances a synthetic recurrent counter per forward, so compare identical chunk policies.
  config->search.chunk_size = 4;
  return CreateModel(GetOrtEnv(), std::move(config));
}

std::vector<float> RunSuffix(State& state, std::span<const int32_t> tokens, int total_length) {
  auto input = state.model_.p_device_->Allocate<int32_t>(tokens.size());
  std::copy(tokens.begin(), tokens.end(), input.CpuSpan().begin());
  input.CopyCpuToDevice();
  auto logits = state.Run(total_length, input, {});
  auto result = logits.CopyDeviceToCpu();
  return {result.begin(), result.end()};
}

TEST(GeneratorCheckpointTest, RestoresAnotherStateWithoutAllocatingAnotherLiveCache) {
  auto model = CheckpointModel();
  auto params = CreateGeneratorParams(*model);
  const std::array<int32_t, 5> prompt{3, 5, 7, 9, 11};
  const std::array<int32_t, 2> suffix{13, 15};
  PrefixSnapshotCache storage{{0, 4096, 0}, 1};
  auto snapshot = std::make_shared<StateCheckpointDescription::Snapshot>();
  std::vector<float> expected;
  {
    auto source = CreateGenerator(*model, *params);
    source->AppendTokens(prompt);
    auto description = source->state_->DescribeCheckpoint();
    ASSERT_NE(description, nullptr);
    ASSERT_TRUE(description->Capture(snapshot, storage, model->p_device_, prompt.size(), "generator"));
    // Four FP16 KV tensors plus conv/recurrent scalars; no mask/position copy.
    EXPECT_EQ(snapshot->bytes, 4u * 2u * prompt.size() * 16u * sizeof(uint16_t) + 2u * sizeof(uint16_t));
    expected = RunSuffix(*source->state_, suffix, 7);
  }
  auto restored = CreateGenerator(*model, *params);
  auto description = restored->state_->DescribeCheckpoint();
  ASSERT_NE(description, nullptr);
  auto* cache = restored->state_->GetInput("past_key_values.0.key");
  ASSERT_NE(cache, nullptr);
  const auto* address = cache->GetTensorRawData();
  description->Restore(*snapshot, model->p_device_);
  EXPECT_EQ(restored->state_->GetInput("past_key_values.0.key")->GetTensorRawData(), address);
  EXPECT_EQ(RunSuffix(*restored->state_, suffix, 7), expected);
  description->Restore(*snapshot, model->p_device_);
  EXPECT_EQ(restored->state_->GetInput("past_key_values.0.key")->GetTensorRawData(), address);
  EXPECT_EQ(RunSuffix(*restored->state_, suffix, 7), expected);
}

TEST(GeneratorCheckpointTest, RejectsIncompatibleCapacityBeforeChangingLiveCache) {
  auto model = CheckpointModel();
  auto params = CreateGeneratorParams(*model);
  auto source = CreateGenerator(*model, *params);
  const std::array<int32_t, 3> prompt{3, 5, 7};
  source->AppendTokens(prompt);
  PrefixSnapshotCache storage{{0, 4096, 0}, 1};
  auto snapshot = std::make_shared<StateCheckpointDescription::Snapshot>();
  auto description = source->state_->DescribeCheckpoint();
  ASSERT_NE(description, nullptr);
  ASSERT_TRUE(description->Capture(snapshot, storage, model->p_device_, prompt.size(), "generator"));
  auto smaller_params = CreateGeneratorParams(*model);
  smaller_params->search.max_length = 16;
  auto destination = CreateGenerator(*model, *smaller_params);
  auto destination_description = destination->state_->DescribeCheckpoint();
  ASSERT_NE(destination_description, nullptr);
  auto cache = ByteWrapTensor(*model->p_device_, *destination->state_->GetInput("past_key_values.0.key"));
  auto bytes = cache.CpuSpan();
  std::fill(bytes.begin(), bytes.end(), uint8_t{73});
  cache.CopyCpuToDevice();
  EXPECT_THROW(destination_description->Restore(*snapshot, model->p_device_), std::runtime_error);
  auto after = cache.CopyDeviceToCpu();
  EXPECT_TRUE(std::all_of(after.begin(), after.end(), [](uint8_t value) { return value == 73; }));
}

TEST(GeneratorCheckpointTest, DeclinesUnrepresentedStateInsteadOfSavingOnlyKv) {
  auto model = CheckpointModel(false);
  auto params = CreateGeneratorParams(*model);
  auto generator = CreateGenerator(*model, *params);
  EXPECT_EQ(generator->state_->DescribeCheckpoint(), nullptr);
  model = CheckpointModel();
  params = CreateGeneratorParams(*model);
  generator = CreateGenerator(*model, *params);
  generator->state_->adapter_names_.push_back("adapter");
  EXPECT_EQ(generator->state_->DescribeCheckpoint(), nullptr);
}

TEST(GeneratorCheckpointTest, ReusesPromptAcrossGeneratorsWithoutChangingGreedyTokens) {
  auto model = CheckpointModel();
  auto& batching = model->config_->engine.dynamic_batching.emplace();
  batching.block_size = 4;
  batching.max_snapshots = 2;
  batching.snapshot_cache.ram_bytes = 16 * 1024;
  auto params = CreateGeneratorParams(*model);
  const std::array<int32_t, 9> prompt{3, 5, 7, 9, 11, 13, 15, 17, 19};
  std::vector<int32_t> expected;
  {
    auto cold = CreateGenerator(*model, *params);
    cold->AppendTokens(prompt);
    EXPECT_EQ(cold->CachedPromptTokens(), 0u);
    for (int i = 0; i < 4; ++i) cold->GenerateNextToken();
    auto sequence = cold->GetSequence(0).CopyDeviceToCpu();
    expected.assign(sequence.begin(), sequence.end());
  }
  for (bool device_input : {false, true}) {
    SCOPED_TRACE(device_input);
    auto warm = CreateGenerator(*model, *params);
    if (device_input) {
      auto tokens = model->p_device_->Allocate<int32_t>(prompt.size());
      std::copy(prompt.begin(), prompt.end(), tokens.CpuSpan().begin());
      tokens.CopyCpuToDevice();
      warm->AppendTokens(tokens);
    } else {
      warm->AppendTokens(prompt);
    }
    EXPECT_EQ(warm->CachedPromptTokens(), 8u);
    for (int i = 0; i < 4; ++i) warm->GenerateNextToken();
    auto sequence = warm->GetSequence(0).CopyDeviceToCpu();
    EXPECT_EQ((std::vector<int32_t>{sequence.begin(), sequence.end()}), expected);
    warm->RewindToLength(0);
    warm->AppendTokens(prompt);
    EXPECT_EQ(warm->CachedPromptTokens(), 8u);
  }
  model->config_->engine.dynamic_batching->prefix_caching = false;
  auto disabled = CreateGeneratorParams(*model);
  auto reference = CreateGenerator(*model, *disabled);
  reference->AppendTokens(prompt);
  EXPECT_EQ(reference->CachedPromptTokens(), 0u);
  for (int i = 0; i < 4; ++i) reference->GenerateNextToken();
  auto sequence = reference->GetSequence(0).CopyDeviceToCpu();
  EXPECT_EQ((std::vector<int32_t>{sequence.begin(), sequence.end()}), expected);
}

TEST(GeneratorCheckpointTest, SeparatesLiveCapacitiesWhileSharingOneSnapshotBudget) {
  auto model = CheckpointModel();
  auto& batching = model->config_->engine.dynamic_batching.emplace();
  batching.block_size = 4;
  batching.max_snapshots = 1;
  batching.snapshot_cache.ram_bytes = 16 * 1024;
  const std::array<int32_t, 9> prompt{3, 5, 7, 9, 11, 13, 15, 17, 19};
  for (const auto& [capacity, cached] : {std::pair{32, 0u}, {16, 0u}, {16, 8u}, {32, 0u}}) {
    SCOPED_TRACE(capacity);
    auto params = CreateGeneratorParams(*model);
    params->search.max_length = capacity;
    auto generator = CreateGenerator(*model, *params);
    generator->AppendTokens(prompt);
    EXPECT_EQ(generator->CachedPromptTokens(), cached);
  }
}

TEST(GeneratorCheckpointTest, KeepsDifferentPrefillPoliciesInCompatibleNamespaces) {
  auto model = CheckpointModel();
  auto& batching = model->config_->engine.dynamic_batching.emplace();
  batching.block_size = 4;
  batching.max_snapshots = 2;
  batching.snapshot_cache.ram_bytes = 16 * 1024;
  const std::array<int32_t, 9> prompt{3, 5, 7, 9, 11, 13, 15, 17, 19};
  for (const auto& [chunk, cached] : {std::pair{4, 0u}, {2, 0u}, {2, 8u}, {4, 8u}}) {
    SCOPED_TRACE(chunk);
    auto params = CreateGeneratorParams(*model);
    params->search.chunk_size = chunk;
    batching.prefix_caching = true;
    auto generator = CreateGenerator(*model, *params);
    generator->AppendTokens(prompt);
    EXPECT_EQ(generator->CachedPromptTokens(), cached);
    auto actual = generator->GetLogits().CopyDeviceToCpu();
    std::vector<float> logits{actual.begin(), actual.end()};
    batching.prefix_caching = false;
    auto reference = CreateGenerator(*model, *params);
    reference->AppendTokens(prompt);
    auto expected = reference->GetLogits().CopyDeviceToCpu();
    EXPECT_EQ(logits, (std::vector<float>{expected.begin(), expected.end()}));
  }
}

TEST(GeneratorCheckpointTest, RestoresOnlyMatchingPrefixAndExecutesChangedSuffix) {
  auto model = CheckpointModel();
  auto& batching = model->config_->engine.dynamic_batching.emplace();
  batching.block_size = 4;
  batching.max_snapshots = 1;
  batching.snapshot_cache.ram_bytes = 16 * 1024;
  auto params = CreateGeneratorParams(*model);
  const std::array<int32_t, 9> first{3, 5, 7, 9, 11, 13, 15, 17, 19};
  auto source = CreateGenerator(*model, *params);
  source->AppendTokens(first);
  auto changed = first;
  changed.back() = 23;
  auto warm = CreateGenerator(*model, *params);
  warm->AppendTokens(changed);
  EXPECT_EQ(warm->CachedPromptTokens(), 8u);
  auto actual = warm->GetLogits().CopyDeviceToCpu();
  std::vector<float> logits{actual.begin(), actual.end()};
  batching.prefix_caching = false;
  auto cold = CreateGenerator(*model, *params);
  cold->AppendTokens(changed);
  auto expected = cold->GetLogits().CopyDeviceToCpu();
  EXPECT_EQ(logits, (std::vector<float>{expected.begin(), expected.end()}));
}

TEST(GeneratorCheckpointTest, PreservesLastNonPaddingLogitsAcrossPaddingOnlyCheckpointForwards) {
  auto model = CheckpointModel();
  auto& batching = model->config_->engine.dynamic_batching.emplace();
  batching.block_size = 4;
  batching.max_snapshots = 2;
  batching.snapshot_cache.ram_bytes = 16 * 1024;
  auto params = CreateGeneratorParams(*model);
  for (const std::vector<int32_t>& prompt :
       {std::vector<int32_t>{3, 5, 7, 9, 11, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 19},
        {0, 0, 0, 0, 3, 5, 7, 9, 11}}) {
    batching.prefix_caching = true;
    for (int repeat = 0; repeat < 2; ++repeat) {
      auto cached = CreateGenerator(*model, *params);
      cached->AppendTokens(cpu_span<const int32_t>{prompt});
      if (repeat != 0) EXPECT_EQ(cached->CachedPromptTokens(), prompt.front() == 0 ? 8u : 4u);
      auto actual = cached->GetLogits().CopyDeviceToCpu();
      std::vector<float> logits{actual.begin(), actual.end()};
      batching.prefix_caching = false;
      auto reference = CreateGenerator(*model, *params);
      reference->AppendTokens(cpu_span<const int32_t>{prompt});
      auto expected = reference->GetLogits().CopyDeviceToCpu();
      EXPECT_EQ(logits, (std::vector<float>{expected.begin(), expected.end()}));
      batching.prefix_caching = true;
    }
  }
}

TEST(GeneratorCheckpointTest, ReplaysLastNonPaddingRowWithoutAConfiguredChunkPolicy) {
  auto model = CheckpointModel();
  auto& batching = model->config_->engine.dynamic_batching.emplace();
  batching.block_size = 4;
  batching.max_snapshots = 1;
  batching.snapshot_cache.ram_bytes = 16 * 1024;
  auto params = CreateGeneratorParams(*model);
  params->search.chunk_size = 0;
  const std::array<int32_t, 9> prompt{3, 5, 7, 9, 11, 0, 0, 0, 0};
  batching.prefix_caching = false;
  auto reference = CreateGenerator(*model, *params);
  reference->AppendTokens(cpu_span<const int32_t>{prompt.data(), 4});
  reference->AppendTokens(cpu_span<const int32_t>{prompt.data() + 4, 5});
  auto expected = reference->GetLogits().CopyDeviceToCpu();
  std::vector<float> logits{expected.begin(), expected.end()};
  batching.prefix_caching = true;
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto generator = CreateGenerator(*model, *params);
    generator->AppendTokens(prompt);
    EXPECT_EQ(generator->CachedPromptTokens(), repeat == 0 ? 0u : 4u);
    auto actual = generator->GetLogits().CopyDeviceToCpu();
    EXPECT_EQ((std::vector<float>{actual.begin(), actual.end()}), logits);
  }
}

}  // namespace
}  // namespace Generators::test
