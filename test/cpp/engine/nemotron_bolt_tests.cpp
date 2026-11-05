// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <gtest/gtest.h>
#include <array>
#include <cstdlib>
#include <functional>

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

TEST(NemotronBolt, PrefillAuxiliaryRowsPreserveOrderAndIgnorePadding) {
  auto* device = GetDeviceInterface(DeviceType::CPU);
  Tensor chunk(device, Ort::TypeToTensorType<float>);
  Tensor prompt(device, Ort::TypeToTensorType<float>);
  chunk.CreateTensor(std::vector<int64_t>{8, 3});
  prompt.CreateTensor(std::vector<int64_t>{19, 3});
  for (size_t start = 0; start < 19; start += 8) {
    auto values = chunk.GetDeviceSpan<float>().CpuSpan();
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = static_cast<float>(start * 3 + i);
    CopyNemotronBoltPrefillAuxRows(prompt, chunk, start, std::min(size_t{8}, 19 - start));
  }
  const auto result = prompt.GetDeviceSpan<float>().CopyDeviceToCpu();
  for (size_t i = 0; i < result.size(); ++i)
    EXPECT_EQ(result[i], static_cast<float>(i));
  EXPECT_THROW(CopyNemotronBoltPrefillAuxRows(prompt, chunk, 18, 2), std::runtime_error);
  EXPECT_THROW(CopyNemotronBoltPrefillAuxRows(prompt, chunk, 0, 9), std::runtime_error);
  EXPECT_THROW(CopyNemotronBoltPrefillAuxRows(prompt, chunk, 0, 0), std::runtime_error);
}

TEST(NemotronBolt, BoundedAuxiliaryMatchesExactFullPromptSuffixAcrossBoundaries) {
  auto* device = GetDeviceInterface(DeviceType::CPU);
  for (size_t count : {size_t{1}, size_t{7}, size_t{8}, size_t{9}, size_t{57},
                       size_t{1031}, size_t{1032}, size_t{1033}, size_t{4099}, size_t{32768}}) {
    const size_t retained = NemotronBoltPrefillAuxRowsToRetain(count, 1024, 8);
    const size_t omitted = count - retained;
    EXPECT_EQ(retained, std::min(count, size_t{1032}));
    Tensor chunk(device, Ort::TypeToTensorType<float>);
    Tensor full(device, Ort::TypeToTensorType<float>);
    Tensor bounded(device, Ort::TypeToTensorType<float>);
    chunk.CreateTensor(std::vector<int64_t>{8, 3});
    full.CreateTensor(std::vector<int64_t>{static_cast<int64_t>(count), 3});
    bounded.CreateTensor(std::vector<int64_t>{static_cast<int64_t>(retained), 3});
    std::fill(bounded.GetDeviceSpan<float>().CpuSpan().begin(),
              bounded.GetDeviceSpan<float>().CpuSpan().end(), -1.0f);
    for (size_t position = 0; position < count; position += 8) {
      auto input = chunk.GetDeviceSpan<float>().CpuSpan();
      for (size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(position * 3 + i);
      const size_t valid = std::min(size_t{8}, count - position);
      CopyNemotronBoltPrefillAuxRows(full, chunk, position, valid);
      CopyNemotronBoltPrefillAuxRows(bounded, chunk, position, valid, omitted);
    }
    const auto expected = full.GetDeviceSpan<float>().CopyDeviceToCpu().subspan(omitted * 3);
    const auto actual = bounded.GetDeviceSpan<float>().CopyDeviceToCpu();
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
      EXPECT_EQ(actual[i], expected[i]) << "N=" << count << ", physical row=" << i / 3;
  }
}

TEST(NemotronBolt, BoundedAuxiliaryKeepsFullAttentionAndRejectsInvalidRanges) {
  EXPECT_EQ(NemotronBoltPrefillAuxRowsToRetain(4099, 0, 8), 4099u);
  EXPECT_THROW(NemotronBoltPrefillAuxRowsToRetain(0, 1024, 8), std::runtime_error);
  EXPECT_THROW(NemotronBoltPrefillAuxRowsToRetain(4099, 1024, 0), std::runtime_error);
  auto* device = GetDeviceInterface(DeviceType::CPU);
  Tensor chunk(device, Ort::TypeToTensorType<float>);
  Tensor bounded(device, Ort::TypeToTensorType<float>);
  chunk.CreateTensor(std::vector<int64_t>{8, 3});
  bounded.CreateTensor(std::vector<int64_t>{1032, 3});
  EXPECT_THROW(CopyNemotronBoltPrefillAuxRows(bounded, chunk, 1033, 1, 1), std::runtime_error);
  EXPECT_THROW(CopyNemotronBoltPrefillAuxRows(bounded, chunk, 0, 9, 1), std::runtime_error);
  EXPECT_THROW(CopyNemotronBoltPrefillAuxRows(bounded, chunk, 0, 0, 1), std::runtime_error);
}

TEST(NemotronBolt, TargetRowContractAcceptsFixedR8AndSymbolicRAndRejectsUnsupportedWidths) {
  const auto fixed = ValidateNemotronBoltTargetRowContract(
      {1, 8}, {"", ""}, {1, 8}, {"", ""}, {1, 8, 131072}, {"", "", ""},
      {1, 8, 24576}, {"", "", ""}, 131072);
  EXPECT_FALSE(fixed.dynamic_r);
  EXPECT_EQ(fixed.aux_width, 24576u);

  const auto dynamic = ValidateNemotronBoltTargetRowContract(
      {1, -1}, {"", "R"}, {1, -1}, {"", "R"}, {1, -1, 131072}, {"", "R", ""},
      {1, -1, 24576}, {"", "R", ""}, 131072);
  EXPECT_TRUE(dynamic.dynamic_r);
  EXPECT_EQ(dynamic.aux_width, 24576u);

  // Reject static 16 without symbolic R
  EXPECT_THROW(ValidateNemotronBoltTargetRowContract(
                   {1, 16}, {"", ""}, {1, 16}, {"", ""}, {1, 16, 131072}, {"", "", ""},
                   {1, 16, 24576}, {"", "", ""}, 131072),
               std::runtime_error);
  // Reject mismatched symbolic / static
  EXPECT_THROW(ValidateNemotronBoltTargetRowContract(
                   {1, -1}, {"", "R"}, {1, 8}, {"", ""}, {1, -1, 131072}, {"", "R", ""},
                   {1, -1, 24576}, {"", "R", ""}, 131072),
               std::runtime_error);
  // Reject wrong symbolic dimension name
  EXPECT_THROW(ValidateNemotronBoltTargetRowContract(
                   {1, -1}, {"", "seq"}, {1, -1}, {"", "seq"}, {1, -1, 131072}, {"", "seq", ""},
                   {1, -1, 24576}, {"", "seq", ""}, 131072),
               std::runtime_error);
  // Reject wrong vocab size
  EXPECT_THROW(ValidateNemotronBoltTargetRowContract(
                   {1, -1}, {"", "R"}, {1, -1}, {"", "R"}, {1, -1, 1000}, {"", "R", ""},
                   {1, -1, 24576}, {"", "R", ""}, 131072),
               std::runtime_error);
}

TEST(NemotronBolt, PlanPrefillChunksCoversR16AndR8TailsAndCapacityGuards) {
  std::vector<size_t> lengths{1, 7, 8, 9, 15, 16, 32, 33, 57, 1031, 1032, 1033, 4099, 32768};
  for (size_t rem = 1; rem <= 15; ++rem) lengths.push_back(16 + rem);
  for (size_t n : lengths) {
    const auto steps = PlanNemotronBoltPrefillChunks(n, 135168, true, 0, 0);
    ASSERT_FALSE(steps.empty());
    size_t expected_r16 = n / 16;
    size_t rem = n % 16;
    size_t expected_r8 = (rem == 0) ? 0 : (rem <= 8 ? 1 : 2);
    size_t actual_r16 = 0, actual_r8 = 0;
    size_t pos = 0;
    int32_t expected_state_in = 0;
    for (size_t i = 0; i < steps.size(); ++i) {
      const auto& s = steps[i];
      EXPECT_EQ(s.pos, pos);
      EXPECT_EQ(s.state_row_in, expected_state_in);
      if (s.profile_rows == 16) {
        ++actual_r16;
        EXPECT_EQ(s.valid_rows, 16u);
        EXPECT_EQ(s.seqlens_k, static_cast<int32_t>(pos + 15));
        EXPECT_EQ(s.state_row_out, 7);
      } else {
        ++actual_r8;
        EXPECT_EQ(s.profile_rows, 8u);
        EXPECT_GE(s.valid_rows, 1u);
        EXPECT_LE(s.valid_rows, 8u);
        EXPECT_EQ(s.seqlens_k, static_cast<int32_t>(pos + 7));
        EXPECT_EQ(s.state_row_out, static_cast<int32_t>(s.valid_rows - 1));
      }
      pos += s.valid_rows;
      EXPECT_EQ(s.length_after, pos);
      expected_state_in = s.state_row_out;
    }
    EXPECT_EQ(pos, n);
    EXPECT_EQ(actual_r16, expected_r16) << "n=" << n;
    EXPECT_EQ(actual_r8, expected_r8) << "n=" << n;
  }
  // Capacity guards
  EXPECT_THROW(PlanNemotronBoltPrefillChunks(16, 15, true, 0, 0), std::runtime_error);
  EXPECT_THROW(PlanNemotronBoltPrefillChunks(17, 23, true, 0, 0), std::runtime_error);
  EXPECT_NO_THROW(PlanNemotronBoltPrefillChunks(17, 24, true, 0, 0));
}

TEST(NemotronBolt, PlanPrefillChunksSupportsNonzeroPrefixSplitAndFinalStateMatch) {
  struct PrefixCase {
    size_t prefix_r8;
    int32_t expected_mid_state_row;
    size_t total_rows;
  };
  const std::vector<PrefixCase> cases{{9, 0, 25}, {12, 3, 28}, {8, 7, 24}};
  for (const auto& tc : cases) {
    const auto r16_steps = PlanNemotronBoltPrefillChunks(tc.total_rows, 135168, true, 0, 0, tc.prefix_r8);
    const auto r8_steps = PlanNemotronBoltPrefillChunks(tc.total_rows, 135168, false, 0, 0, tc.prefix_r8);
    ASSERT_GE(r16_steps.size(), 2u);
    EXPECT_EQ(r16_steps.back().profile_rows, 16u);
    EXPECT_EQ(r16_steps.back().pos, tc.prefix_r8);
    EXPECT_EQ(r16_steps.back().state_row_in, tc.expected_mid_state_row);
    EXPECT_EQ(r16_steps.back().state_row_out, 7);
    EXPECT_EQ(r16_steps.back().length_after, tc.total_rows);
    EXPECT_EQ(r8_steps.back().state_row_out, r16_steps.back().state_row_out);
    EXPECT_EQ(r8_steps.back().length_after, r16_steps.back().length_after);
  }
  EXPECT_THROW(PlanNemotronBoltPrefillChunks(16, 135168, true, 0, 0, 16), std::runtime_error);
}

TEST(NemotronBolt, MixedR16AndR8BoundedAuxiliaryMatchesExactFullPromptSuffixAcrossAllRemainders) {
  auto* device = GetDeviceInterface(DeviceType::CPU);
  std::vector<size_t> lengths{1, 7, 8, 9, 15, 16, 32, 33, 57, 1031, 1032, 1033, 4099, 32768};
  for (size_t rem = 1; rem <= 15; ++rem) lengths.push_back(16 + rem);
  for (size_t count : lengths) {
    const size_t retained = NemotronBoltPrefillAuxRowsToRetain(count, 1024, 8);
    const size_t omitted = count - retained;
    Tensor chunk8(device, Ort::TypeToTensorType<float>);
    Tensor chunk16(device, Ort::TypeToTensorType<float>);
    Tensor full(device, Ort::TypeToTensorType<float>);
    Tensor bounded(device, Ort::TypeToTensorType<float>);
    chunk8.CreateTensor(std::vector<int64_t>{8, 3});
    chunk16.CreateTensor(std::vector<int64_t>{16, 3});
    full.CreateTensor(std::vector<int64_t>{static_cast<int64_t>(count), 3});
    bounded.CreateTensor(std::vector<int64_t>{static_cast<int64_t>(retained), 3});
    std::fill(bounded.GetDeviceSpan<float>().CpuSpan().begin(),
              bounded.GetDeviceSpan<float>().CpuSpan().end(), -1.0f);
    const auto steps = PlanNemotronBoltPrefillChunks(count, 135168, true, 0, 0);
    for (const auto& step : steps) {
      Tensor& chunk = (step.profile_rows == 16) ? chunk16 : chunk8;
      auto input = chunk.GetDeviceSpan<float>().CpuSpan();
      for (size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(step.pos * 3 + i);
      CopyNemotronBoltPrefillAuxRows(full, chunk, step.pos, step.valid_rows);
      CopyNemotronBoltPrefillAuxRows(bounded, chunk, step.pos, step.valid_rows, omitted);
    }
    const auto expected = full.GetDeviceSpan<float>().CopyDeviceToCpu().subspan(omitted * 3);
    const auto actual = bounded.GetDeviceSpan<float>().CopyDeviceToCpu();
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
      EXPECT_EQ(actual[i], expected[i]) << "N=" << count << ", physical row=" << i / 3;
  }
}

// Optional real-graph correctness coverage. No models or GPU are needed for the tests above.
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
  for (int request = 0; request < 2; ++request) {
    auto params = std::make_shared<GeneratorParams>(*model);
    params->search.max_length = static_cast<int>(prompt.size() + 32);
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

}  // namespace
}  // namespace Generators::test
