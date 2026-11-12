// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <gtest/gtest.h>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>

#include "dflash2_drafter.h"
#include "engine/engine.h"
#include "models/nemotron_bolt.h"
#include "search.h"

namespace Generators::test {

struct NemotronBoltTestAccess {
  static void ForceRejectedDrafts(NemotronBoltState& state) {
    state.drafts_.assign(7, 0);
  }

  static std::vector<int32_t> TargetOnly(NemotronBoltState& state, size_t count) {
    std::vector<int32_t> tokens{state.anchor_};
    for (size_t i = 1; i < count; ++i) {
      const std::vector<int32_t> block(8, tokens.back());
      const auto predictions = state.Verify(block);
      ++state.length_;
      state.state_row_ = 0;
      tokens.push_back(static_cast<int32_t>(predictions[0]));
    }
    return tokens;
  }
};

namespace {

class BoltPrefixCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto* cpu = GetDeviceInterface(DeviceType::CPU);
    for (const auto& shape : std::vector<std::vector<int64_t>>{{1, 2, 24, 2}, {8, 3, 2}, {8, 2, 2, 2}}) {
      auto tensor = std::make_unique<Tensor>(
          cpu, owned_.size() == 2 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT : ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
      tensor->CreateTensor(shape);
      caches_.push_back(tensor.get());
      owned_.push_back(std::move(tensor));
    }
    aux_ = std::make_unique<Tensor>(cpu, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
    aux_->CreateTensor(std::array<int64_t, 2>{24, 4});
    for (auto* tensor : caches_) Fill(*tensor);
    Fill(*aux_);
  }
  static void Fill(Tensor& tensor) {
    auto bytes = tensor.GetByteSpan().CpuSpan();
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<uint8_t>(i % 251 + 1);
  }
  static std::vector<uint8_t> Bytes(Tensor& tensor) {
    auto bytes = tensor.GetByteSpan().CpuSpan();
    return {bytes.begin(), bytes.end()};
  }
  std::unique_ptr<NemotronBoltPrefixCache> Cache(size_t count = 2, size_t budget = 8192) {
    return std::make_unique<NemotronBoltPrefixCache>(
        std::make_shared<PrefixSnapshotCache>(std::array<size_t, 3>{0, budget, 0}, count));
  }
  std::vector<std::unique_ptr<Tensor>> owned_;
  std::vector<Tensor*> caches_;
  std::unique_ptr<Tensor> aux_;
};

TEST_F(BoltPrefixCacheTest, RestoresBothKvHeadsSelectedMixedStateRowAndAuxiliarySuffix) {
  auto cache = Cache();
  const std::vector<int32_t> prompt(16, 5);
  std::vector<std::vector<uint8_t>> original;
  for (auto* tensor : caches_) original.push_back(Bytes(*tensor));
  const auto original_aux = Bytes(*aux_);
  cache->Capture(prompt, caches_, 7, 42, *aux_, 0, 8);
  auto saved = cache->Find(prompt);
  ASSERT_NE(saved, nullptr);
  EXPECT_EQ(saved->anchor, 42);
  for (auto* tensor : caches_) tensor->GetByteSpan().Zero();
  aux_->GetByteSpan().Zero();
  cache->Restore(*saved, caches_, *aux_, 8, 8);
  for (size_t i = 0; i < caches_.size(); ++i) {
    std::vector<uint8_t> expected(original[i].size());
    for (const auto& range : saved->ranges[i])
      std::copy_n(original[i].begin() + range.tensor_offset, range.bytes, expected.begin() + range.tensor_offset);
    EXPECT_EQ(Bytes(*caches_[i]), expected) << i;
  }
  auto restored_aux = Bytes(*aux_);
  EXPECT_TRUE(std::equal(original_aux.begin() + 64, original_aux.begin() + 128, restored_aux.begin()));
  EXPECT_TRUE(std::all_of(restored_aux.begin() + 64, restored_aux.end(), [](auto byte) { return byte == 0; }));
  EXPECT_EQ(cache->Metrics().matched_tokens, 16u);
}

TEST_F(BoltPrefixCacheTest, FindsLongestExactTokenPrefixAndEvictsUnleasedCheckpoints) {
  auto cache = Cache();
  const std::vector<int32_t> short_prompt(8, 3);
  std::vector<int32_t> long_prompt(16, 3);
  long_prompt.back() = 4;
  cache->Capture(short_prompt, caches_, 7, 11, *aux_, 0, 8);
  cache->Capture(long_prompt, caches_, 7, 12, *aux_, 0, 8);
  EXPECT_EQ(cache->Find(long_prompt)->tokens.size(), 16u);
  auto divergent = long_prompt;
  divergent.back() = 5;
  EXPECT_EQ(cache->Find(divergent)->tokens.size(), 8u);
  divergent.front() = 5;
  EXPECT_EQ(cache->Find(divergent), nullptr);
  {
    auto short_entry = cache->Find(short_prompt);
    cache->Restore(*short_entry, caches_, *aux_, 0, 8);
  }
  cache->Capture(divergent, caches_, 7, 13, *aux_, 0, 8);
  EXPECT_EQ(cache->Find(long_prompt)->tokens.size(), 8u);
}

TEST_F(BoltPrefixCacheTest, RefusedStorageAndUnalignedPublicationDoNotCreateHits) {
  for (size_t count : {size_t{0}, size_t{1}}) {
    auto cache = Cache(count, count == 0 ? 8192 : 1);
    const std::vector<int32_t> prompt(8, 3);
    cache->Capture(prompt, caches_, 7, 11, *aux_, 0, 8);
    EXPECT_EQ(cache->Find(prompt), nullptr);
    EXPECT_THROW(cache->Capture(std::vector<int32_t>(7, 3), caches_, 6, 11, *aux_, 0, 8), std::runtime_error);
  }
}

TEST_F(BoltPrefixCacheTest, RefusedPublicationPrunesAndCountsEvictionsImmediately) {
  auto storage = std::make_shared<PrefixSnapshotCache>(std::array<size_t, 3>{0, 400, 0}, 2);
  NemotronBoltPrefixCache cache{storage};
  const std::vector<int32_t> prompt(8, 3);
  cache.Capture(prompt, caches_, 7, 11, *aux_, 0, 8);
  std::weak_ptr<NemotronBoltPrefixCache::Entry> evicted = cache.Find(prompt);
  ASSERT_FALSE(evicted.expired());

  auto other = std::make_shared<PrefixSnapshotCache::Entry>();
  other->type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  other->shapes = {{128}};
  ASSERT_TRUE(storage->Allocate(other, aux_->p_device_, "other"));
  auto lease = other;
  cache.Capture(std::vector<int32_t>(16, 4), caches_, 7, 12, *aux_, 0, 8);
  EXPECT_EQ(cache.Metrics().retention_refusals, 1u);
  EXPECT_EQ(cache.Metrics().evictions, 1u);
  EXPECT_TRUE(evicted.expired());
  EXPECT_EQ(cache.Find(prompt), nullptr);
}

TEST_F(BoltPrefixCacheTest, HostStorageTiersPreservePackedStateAndEvictedRestoreFails) {
  for (size_t tier = 1; tier < 3; ++tier) {
    std::array<size_t, 3> limits{};
    limits[tier] = 8192;
    auto storage = std::make_shared<PrefixSnapshotCache>(limits, 1, std::filesystem::temp_directory_path());
    NemotronBoltPrefixCache cache{storage};
    const std::vector<int32_t> prompt(8, 3);
    Fill(*caches_[2]);
    const auto expected = Bytes(*caches_[2]);
    cache.Capture(prompt, caches_, 7, 11, *aux_, 0, 8);
    auto saved = cache.Find(prompt);
    ASSERT_NE(saved, nullptr);
    EXPECT_EQ(static_cast<size_t>(saved->tier), tier);
    caches_[2]->GetByteSpan().Zero();
    cache.Restore(*saved, caches_, *aux_, 0, 8);
    const auto actual = Bytes(*caches_[2]);
    const size_t row_begin = expected.size() / 8 * 7;
    EXPECT_TRUE(std::equal(expected.begin() + row_begin, expected.end(), actual.begin() + row_begin));
    storage->Discard(*saved);
    EXPECT_THROW(cache.Restore(*saved, caches_, *aux_, 0, 8), std::runtime_error);
  }
}

TEST(NemotronBolt, EveryAcceptanceLengthAndBudget) {
  const std::vector<int32_t> drafts{20, 21, 22, 23, 24, 25, 26};
  const std::vector<int> eos{2, 11};
  for (size_t accepted = 0; accepted <= drafts.size(); ++accepted) {
    std::vector<int64_t> predictions(8, 99);
    std::copy_n(drafts.begin(), accepted, predictions.begin());
    EXPECT_EQ(NemotronBoltAcceptedPrefix(drafts, predictions), accepted);
    for (size_t remaining = 0; remaining <= 9; ++remaining) {
      const auto tokens = NemotronBoltRoundTokens(drafts, predictions, remaining, eos);
      ASSERT_EQ(tokens.size(), std::min(accepted + 1, remaining));
      for (size_t i = 0; i < tokens.size(); ++i)
        EXPECT_EQ(tokens[i], i < accepted ? drafts[i] : 99);
    }
  }
}

TEST(NemotronBolt, StopsAtEitherEosIncludingCorrectionAndBonus) {
  for (int eos_token : {2, 11}) {
    for (size_t eos_row = 0; eos_row < 8; ++eos_row) {
      std::vector<int32_t> drafts{20, 21, 22, 23, 24, 25, 26};
      if (eos_row < 7) drafts[eos_row] = eos_token;
      std::vector<int64_t> predictions(drafts.begin(), drafts.end());
      predictions.push_back(eos_row == 7 ? eos_token : 99);
      const auto tokens = NemotronBoltRoundTokens(drafts, predictions, 8, std::vector<int>{2, 11});
      ASSERT_EQ(tokens.size(), eos_row + 1);
      EXPECT_EQ(tokens.back(), eos_token);
    }
    const auto correction = NemotronBoltRoundTokens(
        std::vector<int32_t>(7, 20), std::vector<int64_t>(8, eos_token), 8,
        std::vector<int>{2, 11});
    EXPECT_EQ(correction, std::vector<int32_t>{eos_token});
  }
}

TEST(NemotronBolt, RejectsInvalidVerificationGeometry) {
  EXPECT_THROW(NemotronBoltAcceptedPrefix(std::vector<int32_t>(6), std::vector<int64_t>(8)),
               std::runtime_error);
  EXPECT_THROW(NemotronBoltAcceptedPrefix(std::vector<int32_t>(7), std::vector<int64_t>(7)),
               std::runtime_error);
}

TEST(NemotronBolt, HonorsGreedyEquivalentSamplingOptions) {
  Config config;
  config.model.type = "nemotron_bolt";
  config.model.vocab_size = 128;
  config.model.context_length = 16;
  config.search.max_length = 16;
  config.speculative.max_draft_tokens = 7;
  struct Case {
    float temperature;
    int top_k;
    bool greedy;
  };
  for (const auto& test : std::array<Case, 8>{{
           {0.0f, 0, true},
           {0.0f, 1, true},
           {0.0f, 50, true},
           {0.7f, 1, true},
           {1.0f, 1, true},
           {0.7f, 50, false},
           {1.0f, 0, false},
           {1.0f, 50, false},
       }}) {
    SCOPED_TRACE(::testing::Message() << "temperature=" << test.temperature << ", top_k=" << test.top_k);
    auto params = std::make_shared<GeneratorParams>(config);
    params->use_graph_capture = true;
    params->search.do_sample = true;
    params->search.temperature = test.temperature;
    params->search.top_k = test.top_k;
    if (test.greedy) {
      EXPECT_NO_THROW(ValidateNemotronBoltSearch(*params));
    } else {
      EXPECT_THROW(ValidateNemotronBoltSearch(*params), std::runtime_error);
    }
  }
}

TEST(NemotronBolt, RejectsUnsupportedSearchInsteadOfIgnoringIt) {
  Config config;
  config.model.type = "nemotron_bolt";
  config.model.dflash2.sliding_window = 1024;
  config.speculative.max_draft_tokens = 7;
  auto params = std::make_shared<GeneratorParams>(config);
  params->use_graph_capture = true;
  EXPECT_NO_THROW(ValidateNemotronBoltSearch(*params));
  const std::vector<std::function<void(GeneratorParams&)>> changes{
      [](auto& p) { p.search.do_sample = true; },
      [](auto& p) { p.search.batch_size = 2; },
      [](auto& p) { p.search.num_beams = 2; },
      [](auto& p) { p.search.num_return_sequences = 2; },
      [](auto& p) { p.search.min_length = 1; },
      [](auto& p) { p.search.repetition_penalty = 1.1f; },
      [](auto& p) { p.search.no_repeat_ngram_size = 2; },
      [](auto& p) { p.search.diversity_penalty = 1; },
      [](auto& p) { p.search.length_penalty = 2; },
      [](auto& p) { p.search.blank_penalty = 1; },
      [](auto& p) { p.guidance_type = "json_schema"; },
      [](auto& p) { p.guidance_data = "{}"; },
      [](auto& p) { p.guidance_ff_tokens_enabled = true; },
      [](auto& p) { p.speculative.ngram_size = 3; },
      [](auto& p) { p.speculative.ngram_chained_lookup = true; },
      [](auto& p) { p.speculative.min_adaptive_k = 2; },
      [](auto& p) { p.speculative.cooldown = true; },
      [](auto& p) { p.speculative.max_draft_tokens = 4; },
      [](auto& p) { p.search.chunk_size = 8; },
      [](auto& p) { p.use_graph_capture = false; }};
  for (const auto& change : changes) {
    auto invalid = std::make_shared<GeneratorParams>(config);
    invalid->use_graph_capture = true;
    change(*invalid);
    EXPECT_THROW(ValidateNemotronBoltSearch(*invalid), std::runtime_error);
  }
  Config sliding = config;
  sliding.model.vocab_size = 128;
  sliding.model.context_length = 16;
  sliding.search.max_length = 16;
  auto valid_sliding_free = std::make_shared<GeneratorParams>(sliding);
  valid_sliding_free->use_graph_capture = true;
  EXPECT_NO_THROW(CreateSearch(*valid_sliding_free));
  sliding.model.decoder.sliding_window = Config::Model::Decoder::SlidingWindow{8};
  auto invalid_sliding = std::make_shared<GeneratorParams>(sliding);
  invalid_sliding->use_graph_capture = true;
  EXPECT_THROW(ValidateNemotronBoltSearch(*invalid_sliding), std::runtime_error);
  EXPECT_THROW(CreateSearch(*invalid_sliding), std::runtime_error);
}

TEST(NemotronBolt, PublicTokenCommitAndResetHaveNoLookaheadLeak) {
  Config config;
  config.model.type = "nemotron_bolt";
  config.model.vocab_size = 128;
  config.model.context_length = 16;
  config.model.eos_token_id = {2, 11};
  config.search.max_length = 11;
  auto params = std::make_shared<GeneratorParams>(config);
  auto search = CreateSearch(*params);
  auto prompt = params->p_device->Allocate<int32_t>(3);
  const std::vector<int32_t> prompt_ids{1, 8, 9};
  prompt.CopyFromCpu(prompt_ids);
  for (int request = 0; request < 2; ++request) {
    search->AppendTokens(prompt);
    const auto queued = NemotronBoltRoundTokens(
        std::vector<int32_t>{20, 21, 22, 23, 24, 25, 26},
        std::vector<int64_t>{20, 21, 22, 23, 24, 25, 26, 27}, 8,
        config.model.eos_token_id);
    EXPECT_EQ(search->GetSequenceLength(), 3);
    for (size_t i = 0; i < queued.size(); ++i) {
      EXPECT_FALSE(search->IsDone());
      search->CommitToken(queued[i]);
      EXPECT_EQ(search->GetSequenceLength(), 4 + i);
      const auto visible = search->GetSequence(0).CopyDeviceToCpu();
      EXPECT_EQ(visible.back(), queued[i]);
    }
    EXPECT_TRUE(search->IsDone());
    search->RewindTo(0);
    EXPECT_EQ(search->GetSequenceLength(), 0);
    EXPECT_FALSE(search->IsDone());
  }
}

TEST(NemotronBolt, TerminalEosIsVisibleAndStopsExactlyOnce) {
  for (int token : {2, 11}) {
    Config config;
    config.model.type = "nemotron_bolt";
    config.model.vocab_size = 128;
    config.model.context_length = 16;
    config.model.eos_token_id = {2, 11};
    config.search.max_length = 16;
    auto params = std::make_shared<GeneratorParams>(config);
    auto search = CreateSearch(*params);
    auto prompt = params->p_device->Allocate<int32_t>(1);
    prompt.CpuSpan()[0] = 1;
    search->AppendTokens(prompt);
    search->CommitToken(token);
    EXPECT_TRUE(search->IsDone());
    EXPECT_EQ(search->GetSequenceLength(), 2);
    EXPECT_EQ(search->GetSequence(0).CopyDeviceToCpu().back(), token);
    EXPECT_THROW(search->CommitToken(20), std::runtime_error);
    search->RewindTo(0);
    EXPECT_FALSE(search->IsDone());
    search->AppendTokens(prompt);
    search->CommitToken(token);
    EXPECT_EQ(search->GetSequenceLength(), 2);
  }
}

TEST(NemotronBolt, PromptAtBudgetIsImmediatelyDone) {
  Config config;
  config.model.type = "nemotron_bolt";
  config.model.vocab_size = 128;
  config.model.context_length = 16;
  config.search.max_length = 1;
  auto params = std::make_shared<GeneratorParams>(config);
  auto search = CreateSearch(*params);
  auto prompt = params->p_device->Allocate<int32_t>(1);
  prompt.CpuSpan()[0] = 20;
  search->AppendTokens(prompt);
  EXPECT_TRUE(search->IsDone());
  EXPECT_EQ(search->GetSequenceLength(), 1);
  EXPECT_THROW(search->CommitToken(21), std::runtime_error);
}

// Optional real-graph correctness coverage. No models or GPU are needed for the tests above.
TEST(NemotronBoltGPU, DestructionWaitsForSharedSessionExecution) {
  const char* path = std::getenv("ORTGENAI_BOLT_TEST_MODEL");
  if (!path) GTEST_SKIP() << "Set ORTGENAI_BOLT_TEST_MODEL to a native Bolt package";
  auto model = std::dynamic_pointer_cast<NemotronBoltModel>(CreateModel(GetOrtEnv(), path));
  ASSERT_NE(model, nullptr);
  auto params = std::make_shared<GeneratorParams>(*model);
  params->search.max_length = 16;
  const std::vector<int32_t> prompt(8, 1010);
  Generator survivor{*model, *params};
  survivor.AppendTokens(prompt);
  auto retiring = std::make_unique<Generator>(*model, *params);
  retiring->AppendTokens(prompt);
  std::promise<void> started;
  auto ready = started.get_future();
  std::unique_lock execution{model->execution_mutex_};
  auto destroyed = std::async(std::launch::async, [generator = std::move(retiring), &started]() mutable {
    started.set_value();
    generator.reset();
  });
  ready.wait();
  EXPECT_EQ(destroyed.wait_for(std::chrono::milliseconds{100}), std::future_status::timeout);
  execution.unlock();
  destroyed.get();
  survivor.GenerateNextToken();
  const auto sequence = survivor.GetSequence(0).CopyDeviceToCpu();
  const std::vector<int32_t> expected(sequence.begin(), sequence.end());
  Generator reference{*model, *params};
  reference.AppendTokens(prompt);
  reference.GenerateNextToken();
  const auto actual = reference.GetSequence(0).CopyDeviceToCpu();
  EXPECT_EQ((std::vector<int32_t>{actual.begin(), actual.end()}), expected);
}

TEST(NemotronBoltGPU, SequentialGeneratorsResetAndPublicBoundaries) {
  const char* path = std::getenv("ORTGENAI_BOLT_TEST_MODEL");
  if (!path) GTEST_SKIP() << "Set ORTGENAI_BOLT_TEST_MODEL to a native Bolt package";
  auto model = CreateModel(GetOrtEnv(), path);
  // A real, non-thinking chat asking for a palindrome function. Length 35 also exercises a
  // three-row final prefill chunk and its nonzero recurrent/anchor row.
  const std::vector<int32_t> prompt{
      10, 25708, 1010, 11, 1010, 10, 3263, 1010, 18746, 1261, 17616, 2254, 1455,
      29196, 5627, 1261, 3310, 1395, 1261, 5796, 92145, 1044, 1454, 1261, 4958,
      17984, 1046, 11, 1010, 10, 1503, 19464, 1010, 12, 13};
  std::vector<int32_t> baseline;
  for (int request = 0; request < 3; ++request) {
    SCOPED_TRACE(request);
    auto params = std::make_shared<GeneratorParams>(*model);
    params->search.max_length = static_cast<int>(prompt.size() + 32);
    params->search.do_sample = request != 0;
    params->search.temperature = request == 1 ? 0.0f : 1.0f;
    params->search.top_k = request == 2 ? 1 : 50;
    Generator generator{*model, *params};
    for (int reset = 0; reset < 2; ++reset) {
      if (reset == 1) {
        generator.AppendTokens(prompt);
        generator.GenerateNextToken();
        generator.GenerateNextToken();
        EXPECT_GT(generator.GetSpeculativeStats().tokens_buffered, 0);
        generator.RewindToLength(0);
      }
      generator.AppendTokens(prompt);
      EXPECT_THROW(generator.AppendTokens(prompt), std::runtime_error);
      EXPECT_THROW(generator.GetLogits(), std::runtime_error);
      EXPECT_THROW(generator.RewindToLength(1), std::runtime_error);
      EXPECT_EQ(generator.GetSequence(0).CopyDeviceToCpu().size(), prompt.size());
      while (!generator.IsDone()) {
        const size_t before = generator.TokenCount();
        generator.GenerateNextToken();
        EXPECT_EQ(generator.TokenCount(), before + 1);
      }
      auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
      const std::vector<int32_t> result(sequence.begin(), sequence.end());
      if (baseline.empty()) baseline = result;
      EXPECT_EQ(result, baseline);
      EXPECT_EQ(result.size(), prompt.size() + 32);
      const auto stats = generator.GetSpeculativeStats();
      EXPECT_EQ(stats.tokens_emitted, 32);
      EXPECT_EQ(stats.tokens_queued, 32);
      EXPECT_EQ(stats.tokens_buffered, 0);
      EXPECT_GT(stats.full_accept_rounds, 0);
      EXPECT_GT(stats.partial_accept_rounds, 0);
      generator.RewindToLength(0);
      EXPECT_EQ(generator.TokenCount(), 0);
    }
  }
  {
    auto params = std::make_shared<GeneratorParams>(*model);
    params->search.do_sample = true;
    params->search.temperature = 1.0f;
    params->search.top_k = 50;
    EXPECT_THROW(Generator(*model, *params), std::runtime_error);
  }
  for (int budget : {0, 1, 2, 3, 7, 8, 9}) {
    auto params = std::make_shared<GeneratorParams>(*model);
    params->search.max_length = static_cast<int>(prompt.size()) + budget;
    Generator generator{*model, *params};
    EXPECT_THROW(generator.AppendTokens(std::vector<int32_t>{-1}), std::runtime_error);
    EXPECT_EQ(generator.TokenCount(), 0);
    generator.AppendTokens(prompt);
    while (!generator.IsDone()) generator.GenerateNextToken();
    const auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
    EXPECT_EQ(sequence.size(), prompt.size() + budget);
    EXPECT_TRUE(std::equal(sequence.begin(), sequence.end(), baseline.begin()));
    EXPECT_THROW(generator.GenerateNextToken(), std::runtime_error);
  }
  auto params = std::make_shared<GeneratorParams>(*model);
  params->search.max_length = static_cast<int>(prompt.size()) + 32;
  Generator serial{*model, *params};
  serial.AppendTokens(prompt);
  const auto target_only = NemotronBoltTestAccess::TargetOnly(
      dynamic_cast<NemotronBoltState&>(*serial.state_), 32);
  EXPECT_TRUE(std::equal(target_only.begin(), target_only.end(), baseline.begin() + prompt.size()));
  {
    Generator generator{*model, *params};
    generator.AppendTokens(prompt);
    generator.GenerateNextToken();
    NemotronBoltTestAccess::ForceRejectedDrafts(dynamic_cast<NemotronBoltState&>(*generator.state_));
    generator.GenerateNextToken();
    EXPECT_EQ(generator.GetSpeculativeStats().zero_accept_rounds, 1);
    while (!generator.IsDone()) generator.GenerateNextToken();
    const auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
    EXPECT_TRUE(std::equal(sequence.begin(), sequence.end(), baseline.begin()));
  }

  // Cross multiple paged-cache blocks and wrap the drafter's 1024-position sliding-window ring.
  std::vector<int32_t> long_prompt;
  for (int i = 0; i < 32; ++i) long_prompt.insert(long_prompt.end(), prompt.begin(), prompt.end());
  auto long_params = std::make_shared<GeneratorParams>(*model);
  long_params->search.max_length = static_cast<int>(long_prompt.size() + 16);
  std::vector<int32_t> long_result;
  {
    Generator generator{*model, *long_params};
    generator.AppendTokens(long_prompt);
    while (!generator.IsDone()) generator.GenerateNextToken();
    const auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
    long_result.assign(sequence.begin() + long_prompt.size(), sequence.end());
  }
  ASSERT_FALSE(long_result.empty());
  {
    Generator generator{*model, *long_params};
    generator.AppendTokens(long_prompt);
    const auto expected = NemotronBoltTestAccess::TargetOnly(
        dynamic_cast<NemotronBoltState&>(*generator.state_), long_result.size());
    EXPECT_EQ(long_result, expected);
  }
}

TEST(NemotronBoltGPU, PrefixCacheAcrossGeneratorsAndEngineRequests) {
  const char* path = std::getenv("ORTGENAI_BOLT_TEST_MODEL");
  if (!path) GTEST_SKIP() << "Set ORTGENAI_BOLT_TEST_MODEL to a native Bolt package";
  auto config = CreateConfig(GetOrtEnv(), path);
  config->engine.dynamic_batching.emplace();
  config->engine.dynamic_batching->max_batch_size = 1;
  config->engine.dynamic_batching->prefix_caching = true;
  config->engine.dynamic_batching->max_snapshots = 2;
  config->engine.dynamic_batching->snapshot_cache.ram_bytes = size_t{4} * 1024 * 1024 * 1024;
  auto model = std::dynamic_pointer_cast<NemotronBoltModel>(CreateModel(GetOrtEnv(), std::move(config)));
  ASSERT_NE(model, nullptr);
  ASSERT_NE(model->prefix_cache_, nullptr);
  std::vector<int32_t> seed{10, 25708, 1010, 11, 1010, 10, 3263, 1010, 18746, 1261, 17616, 2254,
                            1455, 29196, 5627, 1261, 3310, 1395, 1261, 5796, 92145, 1044, 1454, 1261,
                            4958, 17984, 1046, 11, 1010, 10, 1503, 19464, 1010, 12, 13};
  const auto generate = [&](const std::vector<int32_t>& prompt) {
    auto params = std::make_shared<GeneratorParams>(*model);
    params->search.max_length = static_cast<int>(prompt.size() + 32);
    Generator generator{*model, *params};
    generator.AppendTokens(prompt);
    const auto cached = dynamic_cast<NemotronBoltState&>(*generator.state_).CachedPromptTokens();
    const auto forwards = generator.GetSpeculativeStats().target_forward_passes;
    while (!generator.IsDone()) generator.GenerateNextToken();
    const auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
    return std::tuple{std::vector<int32_t>{sequence.begin() + prompt.size(), sequence.end()}, cached, forwards};
  };
  for (size_t length : std::array<size_t, 7>{7, 8, 9, 1031, 1032, 1033, 1040}) {
    SCOPED_TRACE(length);
    std::vector<int32_t> prompt;
    while (prompt.size() < length) prompt.insert(prompt.end(), seed.begin(), seed.end());
    prompt.resize(length);
    auto policy = std::move(model->prefix_cache_);
    const auto [cold, unused, cold_forwards] = generate(prompt);
    model->prefix_cache_ = std::move(policy);
    const auto [published, first_cached, unused_forwards] = generate(prompt);
    const auto [warm, cached, warm_forwards] = generate(prompt);
    EXPECT_EQ(published, cold);
    EXPECT_EQ(warm, cold);
    EXPECT_EQ(cached, length / 8 * 8);
    EXPECT_EQ(warm_forwards + cached / 8, cold_forwards);
    // Divergence after a resident aligned checkpoint must not consume the old speculative tail.
    prompt.push_back(1010);
    auto engine = std::make_shared<Engine>(model);
    std::vector<int32_t> baseline;
    for (int repeat = 0; repeat < 2; ++repeat) {
      auto request = engine->CreateRequest();
      TurnOptions options;
      options.max_generated_tokens = 32;
      options.do_sample = false;
      request->BeginTurn(prompt, options);
      for (size_t runs = 0; !request->IsTurnComplete() && runs < 100; ++runs) {
        std::array<EngineEvent, 16> events;
        const auto count = engine->Run(events);
        for (size_t i = 0; i < count; ++i) EXPECT_EQ(events[i].flags & EngineEventFlagFailed, 0u);
      }
      ASSERT_TRUE(request->IsTurnComplete());
      const auto sequence = request->TokensCpu();
      const std::vector<int32_t> output(sequence.begin() + prompt.size(), sequence.end());
      if (repeat == 0) baseline = output;
      EXPECT_EQ(output, baseline);
      EXPECT_GE(request->TurnCachedPromptTokens(), length / 8 * 8);
      if (length == 1033 && repeat == 0) {
        const size_t old_history = request->TokensCpu().size();
        std::vector<int32_t> history(request->TokensCpu().begin(), request->TokensCpu().end());
        const std::vector<int32_t> appended(9, 1010);
        history.insert(history.end(), appended.begin(), appended.end());
        auto [reference, reused, forwards] = generate(history);
        if (!reference.empty() && std::find(model->config_->model.eos_token_id.begin(),
                                            model->config_->model.eos_token_id.end(), reference.back()) !=
                                      model->config_->model.eos_token_id.end())
          reference.pop_back();
        request->BeginTurn(appended, options);
        for (size_t runs = 0; !request->IsTurnComplete() && runs < 100; ++runs) {
          std::array<EngineEvent, 16> events;
          engine->Run(events);
        }
        ASSERT_TRUE(request->IsTurnComplete());
        EXPECT_EQ(request->TurnCachedPromptTokens(), history.size() / 8 * 8 - old_history);
        const auto continued = request->TokensCpu();
        EXPECT_EQ((std::vector<int32_t>{continued.begin() + history.size(), continued.end()}), reference);
      }
      request->Close();
    }
    const auto [expected, ignored, ignored_forwards] = generate(prompt);
    // Request deliberately excludes terminal EOS, unlike the classic Bolt Generator.
    if (!expected.empty() && std::find(model->config_->model.eos_token_id.begin(),
                                       model->config_->model.eos_token_id.end(), expected.back()) !=
                                 model->config_->model.eos_token_id.end())
      EXPECT_EQ(baseline, (std::vector<int32_t>{expected.begin(), expected.end() - 1}));
    else
      EXPECT_EQ(baseline, expected);
  }

  auto engine = std::make_shared<Engine>(model);
  auto request = engine->CreateRequest();
  TurnOptions options;
  options.max_generated_tokens = 32;
  options.do_sample = false;
  request->BeginTurn(seed, options);
  std::array<EngineEvent, 16> events;
  engine->Run(events);
  engine->Run(events);
  ASSERT_FALSE(request->IsTurnComplete());
  ASSERT_GT(engine->GetSpeculativeStats().tokens_buffered, 0u);
  EXPECT_EQ(engine->GetSpeculativeStats().effective_k, 7u);
  ASSERT_TRUE(engine->CancelRequest(request, request->CurrentTurnId()));
  for (size_t runs = 0; engine->HasPendingRequests() && runs < 10; ++runs) engine->Run(events);
  const size_t old_history = request->TokensCpu().size();
  const std::vector<int32_t> appended(9, 1010);
  std::vector<int32_t> history(request->TokensCpu().begin(), request->TokensCpu().end());
  history.insert(history.end(), appended.begin(), appended.end());
  auto [reference, reused, forwards] = generate(history);
  if (!reference.empty() && std::find(model->config_->model.eos_token_id.begin(),
                                      model->config_->model.eos_token_id.end(), reference.back()) !=
                                model->config_->model.eos_token_id.end())
    reference.pop_back();
  request->BeginTurn(appended, options);
  for (size_t runs = 0; !request->IsTurnComplete() && runs < 100; ++runs) engine->Run(events);
  ASSERT_TRUE(request->IsTurnComplete());
  EXPECT_EQ(request->TurnCachedPromptTokens(), history.size() / 8 * 8 - old_history);
  const auto continued = request->TokensCpu();
  EXPECT_EQ((std::vector<int32_t>{continued.begin() + history.size(), continued.end()}), reference);
  EXPECT_GT(engine->GetSpeculativeStats().tokens_discarded, 0u);
  EXPECT_EQ(engine->GetSpeculativeStats().tokens_emitted, 2u + reference.size());
  EXPECT_EQ(engine->GetSpeculativeStats().effective_k, 7u);
  request->Close();
  EXPECT_EQ(engine->GetSpeculativeStats().effective_k, 0u);
}

TEST(NemotronBolt, GgufInitializersConfigParsingAndPathValidation) {
  namespace fs_std = std::filesystem;
  const auto tmp_dir = fs_std::temp_directory_path() / "ortgenai_gguf_init_test";
  std::error_code ec;
  fs_std::remove_all(tmp_dir, ec);
  fs_std::create_directories(tmp_dir);

  auto write_file = [&](const fs_std::path& p, const std::string& content) {
    std::ofstream out(p, std::ios::binary);
    out << content;
  };

  // 1. String form for decoder, inline object form for dflash2
  write_file(tmp_dir / "genai_config.json", R"({
    "model": {
      "type": "nemotron_bolt",
      "vocab_size": 128,
      "context_length": 32,
      "decoder": {
        "filename": "model.onnx",
        "gguf_initializers": "gguf_initializers.json"
      },
      "dflash2": {
        "filename": "dflash2.onnx",
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "head_size": 64,
        "block_size": 8,
        "num_draft_tokens": 7,
        "selector_top_k": 8,
        "mask_token_id": 100,
        "sliding_window": 1024,
        "gguf_initializers": {
          "files": ["weights.gguf"],
          "share": true,
          "tensors": {
            "embed_w": "token_embd.weight",
            "lm_head": {
              "gguf_name": "output.weight",
              "data_type": 2,
              "shape": [128, 34],
              "share": false
            }
          }
        }
      }
    },
    "search": {}
  })");

  Config cfg(fs::path{tmp_dir.string()}, {});
  EXPECT_EQ(cfg.model.decoder.gguf_initializers.config_file, "gguf_initializers.json");
  ASSERT_EQ(cfg.model.dflash2.gguf_initializers.files.size(), 1u);
  EXPECT_EQ(cfg.model.dflash2.gguf_initializers.files[0], "weights.gguf");
  EXPECT_TRUE(cfg.model.dflash2.gguf_initializers.share);
  EXPECT_TRUE(cfg.model.dflash2.gguf_initializers.has_share);
  ASSERT_EQ(cfg.model.dflash2.gguf_initializers.tensors.size(), 2u);
  EXPECT_EQ(cfg.model.dflash2.gguf_initializers.tensors[0].onnx_name, "embed_w");
  EXPECT_EQ(cfg.model.dflash2.gguf_initializers.tensors[0].gguf_name, "token_embd.weight");
  EXPECT_FALSE(cfg.model.dflash2.gguf_initializers.tensors[0].has_share);
  EXPECT_EQ(cfg.model.dflash2.gguf_initializers.tensors[1].onnx_name, "lm_head");
  EXPECT_EQ(cfg.model.dflash2.gguf_initializers.tensors[1].gguf_name, "output.weight");
  EXPECT_EQ(cfg.model.dflash2.gguf_initializers.tensors[1].data_type, 2);
  EXPECT_EQ(cfg.model.dflash2.gguf_initializers.tensors[1].shape, (std::vector<int64_t>{128, 34}));
  EXPECT_TRUE(cfg.model.dflash2.gguf_initializers.tensors[1].has_share);
  EXPECT_FALSE(cfg.model.dflash2.gguf_initializers.tensors[1].share);

  auto dflash2_cfg = CreateDflash2Config(cfg);
  EXPECT_EQ(dflash2_cfg->model.decoder.gguf_initializers.files, cfg.model.dflash2.gguf_initializers.files);
  EXPECT_EQ(dflash2_cfg->model.decoder.gguf_initializers.tensors.size(), 2u);

  // Per-tensor file and data type overrides for mtp
  Config mtp_cfg;
  OverlayConfig(mtp_cfg, R"({
    "model": {
      "mtp": {
        "filename": "mtp.onnx",
        "gguf_initializers": {
          "files": ["mtp.gguf"],
          "tensors": {
            "w0": {
              "gguf_name": "blk.0.weight",
              "file": "shard0.gguf",
              "data_type": 1
            }
          }
        }
      }
    }
  })");
  ASSERT_EQ(mtp_cfg.model.mtp.gguf_initializers.files.size(), 1u);
  EXPECT_EQ(mtp_cfg.model.mtp.gguf_initializers.files[0], "mtp.gguf");
  ASSERT_EQ(mtp_cfg.model.mtp.gguf_initializers.tensors.size(), 1u);
  EXPECT_EQ(mtp_cfg.model.mtp.gguf_initializers.tensors[0].onnx_name, "w0");
  EXPECT_EQ(mtp_cfg.model.mtp.gguf_initializers.tensors[0].gguf_name, "blk.0.weight");
  EXPECT_EQ(mtp_cfg.model.mtp.gguf_initializers.tensors[0].file, "shard0.gguf");
  EXPECT_EQ(mtp_cfg.model.mtp.gguf_initializers.tensors[0].data_type, 1);

  // 2. LoadGgufInitializersConfig from external JSON file and reject path traversal
  write_file(tmp_dir / "gguf_initializers.json", R"({
    "files": ["model.gguf"],
    "tensors": {
      "embed": {
        "gguf_name": "token_embd.weight",
        "share": true
      }
    }
  })");
  Config::Model::GgufInitializers ext_loaded;
  LoadGgufInitializersConfig(fs::path{(tmp_dir / "gguf_initializers.json").string()}, ext_loaded);
  ASSERT_EQ(ext_loaded.files.size(), 1u);
  EXPECT_EQ(ext_loaded.files[0], "model.gguf");
  ASSERT_EQ(ext_loaded.tensors.size(), 1u);
  EXPECT_EQ(ext_loaded.tensors[0].onnx_name, "embed");
  EXPECT_TRUE(ext_loaded.tensors[0].share);

  // Path traversal in external gguf_initializers.json must be rejected
  write_file(tmp_dir / "bad_gguf_initializers.json", R"({
    "files": ["../outside.gguf"],
    "tensors": {}
  })");
  Config::Model::GgufInitializers bad_loaded;
  EXPECT_THROW(LoadGgufInitializersConfig(fs::path{(tmp_dir / "bad_gguf_initializers.json").string()}, bad_loaded),
               std::runtime_error);

  // Path traversal in inline genai_config.json gguf_initializers must be rejected
  write_file(tmp_dir / "genai_config.json", R"({
    "model": {
      "type": "nemotron_bolt",
      "vocab_size": 128,
      "context_length": 32,
      "decoder": {
        "filename": "model.onnx",
        "gguf_initializers": "../outside.json"
      }
    },
    "search": {}
  })");
  EXPECT_THROW(Config(fs::path{tmp_dir.string()}, {}), std::runtime_error);

  // 3. Synthetic GGUF v3 binary header parsing and AddGgufInitializers validation
  auto write_u32 = [](std::string& buf, uint32_t v) {
    buf.append(reinterpret_cast<const char*>(&v), 4);
  };
  auto write_u64 = [](std::string& buf, uint64_t v) {
    buf.append(reinterpret_cast<const char*>(&v), 8);
  };
  auto write_gguf_str = [&](std::string& buf, std::string_view s) {
    write_u64(buf, s.size());
    buf.append(s.data(), s.size());
  };

  std::string gguf_bytes;
  gguf_bytes.append("GGUF", 4);
  write_u32(gguf_bytes, 3);  // version 3
  write_u64(gguf_bytes, 2);  // tensor_count = 2
  write_u64(gguf_bytes, 1);  // kv_count = 1
  write_gguf_str(gguf_bytes, "general.alignment");
  write_u32(gguf_bytes, 4);  // UINT32
  write_u32(gguf_bytes, 32);

  // Tensor 0: "weight_f32", dims=[4, 2] -> ONNX default shape [2, 4] FLOAT (32 bytes), offset 0
  write_gguf_str(gguf_bytes, "weight_f32");
  write_u32(gguf_bytes, 2);
  write_u64(gguf_bytes, 4);
  write_u64(gguf_bytes, 2);
  write_u32(gguf_bytes, 0);  // F32
  write_u64(gguf_bytes, 0);

  // Tensor 1: "weight_q8", dims=[32, 2] -> ONNX default shape [2, 34] UINT8 (68 bytes), offset 32
  write_gguf_str(gguf_bytes, "weight_q8");
  write_u32(gguf_bytes, 2);
  write_u64(gguf_bytes, 32);
  write_u64(gguf_bytes, 2);
  write_u32(gguf_bytes, 8);  // Q8_0
  write_u64(gguf_bytes, 32);

  while (gguf_bytes.size() % 32 != 0) gguf_bytes.push_back('\0');
  gguf_bytes.append(32 + 68, '\x01');
  write_file(tmp_dir / "synthetic.gguf", gguf_bytes);

  // Missing GGUF tensor name throws from AddGgufInitializers before CreateSession
  write_file(tmp_dir / "genai_config.json", R"({
    "model": {
      "type": "gpt2",
      "vocab_size": 128,
      "context_length": 32,
      "decoder": {
        "filename": "nonexistent_model.onnx",
        "gguf_initializers": {
          "files": ["synthetic.gguf"],
          "tensors": {
            "w_missing": "does_not_exist"
          }
        }
      }
    },
    "search": {}
  })");
  try {
    CreateModel(GetOrtEnv(), tmp_dir.string().c_str());
    FAIL() << "Expected missing GGUF tensor to throw";
  } catch (const std::exception& e) {
    EXPECT_NE(std::string(e.what()).find("does_not_exist"), std::string::npos) << e.what();
  }

  // Byte size mismatch throws from AddGgufInitializers
  write_file(tmp_dir / "genai_config.json", R"({
    "model": {
      "type": "gpt2",
      "vocab_size": 128,
      "context_length": 32,
      "decoder": {
        "filename": "nonexistent_model.onnx",
        "gguf_initializers": {
          "files": ["synthetic.gguf"],
          "tensors": {
            "w0": {
              "gguf_name": "weight_f32",
              "shape": [2, 5]
            }
          }
        }
      }
    },
    "search": {}
  })");
  try {
    CreateModel(GetOrtEnv(), tmp_dir.string().c_str());
    FAIL() << "Expected byte size mismatch to throw";
  } catch (const std::exception& e) {
    EXPECT_NE(std::string(e.what()).find("Byte size mismatch"), std::string::npos) << e.what();
  }

  // Valid GGUF tensor mappings succeed in AddGgufInitializers and reach CreateSession(nonexistent_model.onnx)
  write_file(tmp_dir / "genai_config.json", R"({
    "model": {
      "type": "gpt2",
      "vocab_size": 128,
      "context_length": 32,
      "decoder": {
        "filename": "nonexistent_model.onnx",
        "gguf_initializers": {
          "files": ["synthetic.gguf"],
          "tensors": {
            "w0": "weight_f32",
            "w1": "weight_q8"
          }
        }
      }
    },
    "search": {}
  })");
  try {
    CreateModel(GetOrtEnv(), tmp_dir.string().c_str());
    FAIL() << "Expected nonexistent_model.onnx to throw after AddGgufInitializers succeeds";
  } catch (const std::exception& e) {
    EXPECT_NE(std::string(e.what()).find("nonexistent_model.onnx"), std::string::npos) << e.what();
  }

  fs_std::remove_all(tmp_dir, ec);
}

}  // namespace
}  // namespace Generators::test
