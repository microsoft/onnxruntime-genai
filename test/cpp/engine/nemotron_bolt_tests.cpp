// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <gtest/gtest.h>
#include <array>
#include <cstdlib>
#include <fstream>
#include <functional>

#include <nlohmann/json.hpp>

#include "models/nemotron_bolt.h"
#include "search.h"

namespace Generators::test {

struct NemotronBoltTestAccess {
  static std::vector<uint8_t> SelectedLogits(NemotronBoltState& state) {
    const size_t row_bytes = static_cast<size_t>(state.bolt_.config_->model.vocab_size) * sizeof(uint16_t);
    // Keep the wrapped buffer's host mirror alive until the vector has been copied.
    auto selected = state.tensors_.at("logits")->GetByteSpan().subspan(static_cast<size_t>(state.state_row_) * row_bytes, row_bytes);
    const auto bytes = selected.CopyDeviceToCpu();
    return {bytes.begin(), bytes.end()};
  }

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

TEST(NemotronBolt, TargetOnlySearchDoesNotRequireCapture) {
  Config config;
  config.model.type = "nemotron_bolt";
  config.speculative.max_draft_tokens = 0;
  auto params = std::make_shared<GeneratorParams>(config);
  for (bool capture : {false, true}) {
    params->use_graph_capture = capture;
    EXPECT_NO_THROW(ValidateNemotronBoltSearch(*params));
  }
  params->speculative.max_draft_tokens = 7;
  params->use_graph_capture = false;
  EXPECT_THROW(ValidateNemotronBoltSearch(*params), std::runtime_error);
}

TEST(NemotronBolt, RejectsUnsupportedPackageDraftCountBeforeLoadingSessions) {
  for (int draft_tokens : {Config{}.speculative.max_draft_tokens, -1, 1, 6, 8}) {
    auto config = std::make_unique<Config>();
    config->model.type = "nemotron_bolt";
    config->model.decoder.filename = "must-not-load.onnx";
    config->speculative.max_draft_tokens = draft_tokens;
    try {
      NemotronBoltModel model{std::move(config), GetOrtEnv()};
      FAIL() << "Unsupported package draft count " << draft_tokens << " was accepted";
    } catch (const std::runtime_error& error) {
      EXPECT_STREQ(error.what(), "nemotron_bolt: speculative.max_draft_tokens must be 0 or 7");
    }
  }
}

TEST(NemotronBolt, ParsesTargetOnlyRegardlessOfJsonMemberOrder) {
  for (const auto document : {
           R"({"speculative":{"max_draft_tokens":0},"model":{"type":"nemotron_bolt","context_length":16,"vocab_size":128}})",
           R"({"model":{"type":"nemotron_bolt","context_length":16,"vocab_size":128},"speculative":{"max_draft_tokens":0}})"}) {
    Config config{fs::path{}, {}, document};
    EXPECT_EQ(config.speculative.max_draft_tokens, 0);
    EXPECT_NO_THROW(OverlayConfig(config, R"({"speculative":{"max_draft_tokens":0}})"));
  }
  EXPECT_THROW((Config{fs::path{}, {}, R"({"speculative":{"max_draft_tokens":0},"model":{"type":"llama","context_length":16}})"}),
               std::runtime_error);
}

TEST(NemotronBolt, PublicSpeculativeSetterAllowsZeroOnlyForBolt) {
  for (const char* type : {"nemotron_bolt", "llama"}) {
    Config config;
    config.model.type = type;
    GeneratorParams params{config};
    ASSERT_NO_THROW(params.SetSpeculativeNumber("max_draft_tokens", 7));
    if (config.model.type == "nemotron_bolt") {
      EXPECT_NO_THROW(params.SetSpeculativeNumber("max_draft_tokens", 0));
      EXPECT_EQ(params.GetSpeculativeNumber("max_draft_tokens"), 0);
    } else {
      EXPECT_THROW(params.SetSpeculativeNumber("max_draft_tokens", 0), std::exception);
      EXPECT_EQ(params.GetSpeculativeNumber("max_draft_tokens"), 7);
    }
    for (double invalid : {-1.0, 0.5, 17.0}) {
      EXPECT_THROW(params.SetSpeculativeNumber("max_draft_tokens", invalid), std::exception);
    }
  }
}

TEST(NemotronBolt, PreservesDrafterOptimizerExclusions) {
  for (const std::string existing : {"", "OtherFusion", "OtherFusion,GatherSliceToSplitFusion",
                                     "GatherSliceToSplitFusionSuffix"}) {
    Config config;
    config.model.vocab_size = 128;
    auto& drafter = config.model.dflash2;
    drafter.filename = "drafter.onnx";
    drafter.num_hidden_layers = drafter.num_key_value_heads = drafter.head_size = 1;
    drafter.block_size = 8;
    drafter.num_draft_tokens = 7;
    drafter.selector_top_k = 1;
    drafter.sliding_window = 8;
    drafter.session_options.emplace();
    auto& entries = drafter.session_options->config_entries;
    entries.emplace_back("optimization.disable_specified_optimizers", "SupersededFusion");
    entries.emplace_back("optimization.disable_specified_optimizers", existing);
    const auto projected = CreateNemotronBoltDrafterConfig(config);
    const auto& merged = projected->model.decoder.session_options.config_entries;
    ASSERT_EQ(merged.size(), entries.size());
    EXPECT_EQ(merged.front().second, "SupersededFusion");
    EXPECT_EQ(merged.back().second,
              existing.empty() ? "GatherSliceToSplitFusion"
                               : existing + (existing == "OtherFusion,GatherSliceToSplitFusion"
                                                 ? ""
                                                 : ",GatherSliceToSplitFusion"));
  }
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

std::unique_ptr<Config> BoltTestConfig(const char* path) {
  auto config = std::make_unique<Config>(fs::path{path}, "{}");
  for (auto* session : {&config->model.decoder.session_options,
                        config->model.dflash2.session_options
                            ? &*config->model.dflash2.session_options
                            : nullptr}) {
    if (!session) continue;
    session->intra_op_num_threads = 1;
    session->inter_op_num_threads = 1;
    for (auto& provider : session->provider_options) {
      if (provider.name != "cuda") continue;
      auto warmup = std::find_if(provider.options.begin(), provider.options.end(),
                                 [](const auto& option) {
                                   return option.first == "min_num_runs_before_cuda_graph_capture";
                                 });
      if (warmup != provider.options.end()) {
        if (warmup->second != "2")
          throw std::runtime_error("Bolt diagnostic requires exactly two CUDA graph warmups");
        // The built-in EP fixes this count at two; only the plugin EP accepts the option.
        provider.options.erase(warmup);
      }
    }
    if (std::getenv("ORTGENAI_BOLT_BUILTIN")) {
      session->custom_ops_library.reset();
      session->custom_ops_libraries.clear();
    }
  }
  return config;
}

// Optional real-graph correctness coverage. No models or GPU are needed for the tests above.
TEST(NemotronBoltGPU, TargetOnlyCaptureAndReset) {
  const char* path = std::getenv("ORTGENAI_BOLT_TEST_MODEL");
  if (!path) GTEST_SKIP() << "Set ORTGENAI_BOLT_TEST_MODEL to a native Bolt package";
  auto config = BoltTestConfig(path);
  config->speculative.max_draft_tokens = 0;
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  auto& bolt = dynamic_cast<NemotronBoltModel&>(*model);
  ASSERT_FALSE(bolt.drafter_model_);
  const std::vector<int32_t> prompt{
      10, 25708, 1010, 11, 1010, 10, 3263, 1010, 18746, 1261, 17616, 2254, 1455,
      29196, 5627, 1261, 3310, 1395, 1261, 5796, 92145, 1044, 1454, 1261, 4958,
      17984, 1046, 11, 1010, 10, 1503, 19464, 1010, 12, 13};
  std::vector<int32_t> expected;
  std::vector<uint8_t> expected_logits;
  for (bool capture : {false, true}) {
    auto params = std::make_shared<GeneratorParams>(*model);
    params->use_graph_capture = capture;
    params->search.max_length = static_cast<int>(prompt.size()) + 32;
    Generator generator{*model, *params};
    for (int request = 0; request < 2; ++request) {
      std::vector<uint8_t> logits;
      generator.AppendTokens(prompt);
      while (!generator.IsDone()) {
        generator.GenerateNextToken();
        const auto row = NemotronBoltTestAccess::SelectedLogits(
            dynamic_cast<NemotronBoltState&>(*generator.state_));
        logits.insert(logits.end(), row.begin(), row.end());
      }
      const auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
      const std::vector<int32_t> result(sequence.begin(), sequence.end());
      if (expected.empty()) {
        expected = result;
        expected_logits = logits;
      }
      EXPECT_EQ(result, expected);
      EXPECT_EQ(logits.size(), expected_logits.size());
      EXPECT_TRUE(logits == expected_logits);
      const auto stats = generator.GetSpeculativeStats();
      EXPECT_EQ(stats.draft_forward_passes, 0);
      EXPECT_EQ(stats.rounds, 0);
      EXPECT_EQ(stats.effective_k, 0);
      EXPECT_EQ(stats.tokens_emitted, result.size() - prompt.size());
      generator.RewindToLength(0);
      EXPECT_EQ(generator.TokenCount(), 0);
    }
  }
  if (const char* output = std::getenv("ORTGENAI_BOLT_OUTPUT")) {
    std::ofstream tokens{std::string{output} + ".tokens"};
    for (int32_t token : expected) tokens << token << "\n";
    std::ofstream logits{std::string{output} + ".logits", std::ios::binary};
    logits.write(reinterpret_cast<const char*>(expected_logits.data()),
                 static_cast<std::streamsize>(expected_logits.size()));
    ASSERT_TRUE(tokens && logits);
  }
#if defined(__linux__)
  if (std::getenv("ORTGENAI_BOLT_BUILTIN")) {
    std::ifstream maps{"/proc/self/maps"};
    ASSERT_TRUE(maps);
    const std::string libraries{std::istreambuf_iterator<char>{maps}, {}};
    EXPECT_EQ(libraries.find("libbolt"), std::string::npos);
  }
#endif
}

TEST(NemotronBoltGPU, SequentialGeneratorsResetAndPublicBoundaries) {
  const char* path = std::getenv("ORTGENAI_BOLT_TEST_MODEL");
  if (!path) GTEST_SKIP() << "Set ORTGENAI_BOLT_TEST_MODEL to a native Bolt package";
  auto model = CreateModel(GetOrtEnv(), BoltTestConfig(path));
  auto& bolt = dynamic_cast<NemotronBoltModel&>(*model);
  ASSERT_TRUE(bolt.drafter_model_);
  ASSERT_EQ(bolt.shared_initializer_entries_.size(), 2);
  ASSERT_EQ(bolt.drafter_model_->shared_initializer_entries_.size(), 2);
  for (size_t i = 0; i < bolt.shared_initializer_entries_.size(); ++i)
    EXPECT_EQ(bolt.shared_initializer_entries_[i].get(),
              bolt.drafter_model_->shared_initializer_entries_[i].get());
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
      EXPECT_EQ(stats.zero_accept_rounds + stats.partial_accept_rounds + stats.full_accept_rounds,
                stats.rounds);
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
  for (bool capture : {false, true}) {
    auto ordinary_params = std::make_shared<GeneratorParams>(*model);
    ordinary_params->speculative.max_draft_tokens = 0;
    ordinary_params->use_graph_capture = capture;
    ordinary_params->search.max_length = static_cast<int>(prompt.size()) + 32;
    Generator ordinary{*model, *ordinary_params};
    for (int request = 0; request < 2; ++request) {
      ordinary.AppendTokens(prompt);
      while (!ordinary.IsDone()) ordinary.GenerateNextToken();
      const auto sequence = ordinary.GetSequence(0).CopyDeviceToCpu();
      EXPECT_TRUE(std::equal(sequence.begin(), sequence.end(), baseline.begin(), baseline.end()));
      const auto stats = ordinary.GetSpeculativeStats();
      EXPECT_EQ(stats.draft_forward_passes, 0);
      EXPECT_EQ(stats.rounds, 0);
      EXPECT_EQ(stats.effective_k, 0);
      EXPECT_EQ(stats.tokens_emitted, 32);
      EXPECT_EQ(stats.tokens_buffered, 0);
      ordinary.RewindToLength(0);
    }
  }
  {
    Generator generator{*model, *params};
    generator.AppendTokens(prompt);
    generator.GenerateNextToken();
    NemotronBoltTestAccess::ForceRejectedDrafts(dynamic_cast<NemotronBoltState&>(*generator.state_));
    generator.GenerateNextToken();
    const auto stats = generator.GetSpeculativeStats();
    EXPECT_EQ(stats.zero_accept_rounds, 1);
    EXPECT_EQ(stats.partial_accept_rounds, 0);
    EXPECT_EQ(stats.full_accept_rounds, 0);
    EXPECT_EQ(stats.rounds, 1);
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
#if defined(__linux__)
  if (std::getenv("ORTGENAI_BOLT_BUILTIN")) {
    std::ifstream maps{"/proc/self/maps"};
    ASSERT_TRUE(maps);
    const std::string libraries{std::istreambuf_iterator<char>{maps}, {}};
    EXPECT_EQ(libraries.find("libbolt"), std::string::npos);
  }
#endif
}

TEST(NemotronBoltGPU, RealPromptOrdinaryAndSpeculative) {
  const char* path = std::getenv("ORTGENAI_BOLT_TEST_MODEL");
  const char* cases_path = std::getenv("ORTGENAI_BOLT_TEST_CASES");
  if (!path || !cases_path)
    GTEST_SKIP() << "Set ORTGENAI_BOLT_TEST_MODEL and ORTGENAI_BOLT_TEST_CASES";
  const char* output = std::getenv("ORTGENAI_BOLT_OUTPUT");
  ASSERT_NE(output, nullptr);
  std::ifstream input{cases_path};
  ASSERT_TRUE(input);
  nlohmann::json fixtures;
  input >> fixtures;
  ASSERT_FALSE(fixtures.at("cases").empty());
  std::ofstream report{std::string{output} + ".jsonl"};
  ASSERT_TRUE(report);
  auto model = CreateModel(GetOrtEnv(), BoltTestConfig(path));
  for (const auto& fixture : fixtures.at("cases")) {
    const auto case_id = fixture.at("case_id").get<std::string>();
    SCOPED_TRACE(case_id);
    const auto prompt = fixture.at("prompt_ids").get<std::vector<int32_t>>();
    const auto budget = fixture.at("max_new_tokens").get<size_t>();
    ASSERT_FALSE(prompt.empty());
    ASSERT_GT(budget, 0);
    ASSERT_LE(budget, 512);
    std::array<std::vector<int32_t>, 2> generated;
    nlohmann::json modes = nlohmann::json::array();
    for (size_t mode = 0; mode < generated.size(); ++mode) {
      auto params = std::make_shared<GeneratorParams>(*model);
      params->speculative.max_draft_tokens = mode == 0 ? 0 : 7;
      params->use_graph_capture = true;
      params->search.max_length = static_cast<int>(prompt.size() + budget);
      Generator generator{*model, *params};
      generator.AppendTokens(prompt);
      while (!generator.IsDone()) generator.GenerateNextToken();
      const auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
      ASSERT_GE(sequence.size(), prompt.size());
      EXPECT_TRUE(std::equal(prompt.begin(), prompt.end(), sequence.begin()));
      generated[mode].assign(sequence.begin() + prompt.size(), sequence.end());
      ASSERT_FALSE(generated[mode].empty());
      EXPECT_LE(generated[mode].size(), budget);
      const auto& eos = model->config_->model.eos_token_id;
      const bool ended_with_eos = std::find(eos.begin(), eos.end(), generated[mode].back()) != eos.end();
      EXPECT_TRUE(ended_with_eos || generated[mode].size() == budget);
      const auto stats = generator.GetSpeculativeStats();
      EXPECT_EQ(stats.tokens_emitted, generated[mode].size());
      EXPECT_EQ(stats.dflash2_failures, 0);
      EXPECT_EQ(stats.dflash2_disables, 0);
      if (mode == 0) {
        EXPECT_EQ(stats.rounds, 0);
        EXPECT_EQ(stats.draft_forward_passes, 0);
      } else if (generated[mode].size() > 1) {
        EXPECT_GT(stats.rounds, 0);
        EXPECT_GT(stats.draft_forward_passes, 0);
      }
      modes.push_back({
          {"k", params->speculative.max_draft_tokens},
          {"generated_ids", generated[mode]},
          {"ended_with_eos", ended_with_eos},
          {"reached_output_cap", generated[mode].size() == budget},
          {"rounds", stats.rounds},
          {"draft_forward_passes", stats.draft_forward_passes},
          {"acceptance_length_histogram", stats.acceptance_length_histogram},
          {"dflash2_failures", stats.dflash2_failures},
          {"dflash2_disables", stats.dflash2_disables},
          {"tokens_emitted", stats.tokens_emitted},
          {"tokens_buffered", stats.tokens_buffered},
      });
    }
    report << nlohmann::json{
                  {"case_id", case_id},
                  {"prompt_tokens", prompt.size()},
                  {"same_generated_ids", generated[0] == generated[1]},
                  {"modes", modes}}
                  .dump()
           << "\n";
    report.flush();
    ASSERT_TRUE(report);
    EXPECT_EQ(generated[0], generated[1]);
  }
#if defined(__linux__)
  if (std::getenv("ORTGENAI_BOLT_BUILTIN")) {
    std::ifstream maps{"/proc/self/maps"};
    ASSERT_TRUE(maps);
    const std::string libraries{std::istreambuf_iterator<char>{maps}, {}};
    EXPECT_EQ(libraries.find("libbolt"), std::string::npos);
  }
#endif
}

}  // namespace
}  // namespace Generators::test
