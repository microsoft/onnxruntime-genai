// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "config.h"
#include "dflash2_drafter.h"
#include "generator/generators.h"
#include "models/io/kv_cache.h"
#include "ort_genai.h"

namespace Generators::test {
namespace {

struct TurnResult {
  size_t cached{};
  uint64_t proposed{};
  uint64_t accepted{};
  std::vector<int32_t> output;
};

TurnResult RunGraphRestoreTurn(OgaEngine& engine, const OgaSequences& prompt) {
  auto request_options = OgaRequestOptions::Create();
  request_options->SetMaxSessionTokens(8192);
  auto request = engine.CreateRequest(request_options.get());
  auto options = request->CreateTurnOptions();
  options->SetDoSample(false);
  options->SetMaxGeneratedTokens(64);
  const auto before = engine.GetSpeculativeStats();
  const uint64_t proposed_before = before->GetCount("draft_tokens_proposed");
  const uint64_t accepted_before = before->GetCount("draft_tokens_accepted");
  const uint64_t turn_id = request->BeginTurn(prompt.SequenceData(0), prompt.SequenceCount(0), options.get());
  auto events = engine.CreateEventBuffer(128);
  TurnResult result;
  bool finished = false;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
  while (!finished && std::chrono::steady_clock::now() < deadline) {
    engine.Run(*events);
    for (size_t i = 0; i < events->Count(); ++i) {
      const auto* event = events->Get(i);
      if (event->Flags() & OgaEngineEventFlag_Failed) {
        throw std::runtime_error("Graph restore turn failed.");
      }
      if (event->TurnId() != turn_id) {
        continue;
      }
      if (event->Flags() & OgaEngineEventFlag_Token) {
        result.output.push_back(event->Token());
      }
      if (event->Flags() & OgaEngineEventFlag_TurnFinished) {
        result.cached = event->Usage().CachedPromptTokens();
        finished = true;
      }
    }
  }
  if (!finished) {
    throw std::runtime_error("Graph restore turn timed out.");
  }
  const auto after = engine.GetSpeculativeStats();
  result.proposed = after->GetCount("draft_tokens_proposed") - proposed_before;
  result.accepted = after->GetCount("draft_tokens_accepted") - accepted_before;
  request->Close();
  engine.Run(*events);
  return result;
}

class Dflash2GraphRestoreTest : public ::testing::TestWithParam<const char*> {};

TEST_P(Dflash2GraphRestoreTest, ExtendedCachedPrefixReplaysWithoutStaleGraphOrLostDrafts) {
  const char* model_path = std::getenv("D_FLASH2_GRAPH_TEST_MODEL");
  if (!model_path || !*model_path) {
    ASSERT_EQ(std::getenv("D_FLASH2_GRAPH_TEST_REQUIRED"), nullptr)
        << "CUDA qualification requires D_FLASH2_GRAPH_TEST_MODEL.";
    GTEST_SKIP() << "Set D_FLASH2_GRAPH_TEST_MODEL to a graph-enabled Qwen model with 256-token "
                    "blocks and 512-token prefill, and pass --ep_dir with its CUDA plugin.";
  }
  auto config = CreateConfig(GetOrtEnv(), model_path);
  ASSERT_TRUE(IsGraphCaptureEnabled(config->model.decoder.session_options));
  ASSERT_TRUE(IsGraphCaptureEnabled(CreateDflash2Config(*config)->model.decoder.session_options));
  auto oga_config = OgaConfig::Create(model_path);
  oga_config->Overlay(R"({"engine":{"dynamic_batching":{"prefix_caching":true,"max_scheduled_tokens":512,"max_batch_size":4,"dflash2_max_snapshots":3,"max_snapshots":3}}})");
  {
    auto draft_model = std::make_shared<Dflash2Model>(CreateDflash2Config(*config), GetOrtEnv());
    const auto type = draft_model->session_info_.GetInputDataType(
        ComposeKeyValueName(config->model.dflash2.inputs.past_key_names, 0));
    const size_t bytes = Dflash2Drafter::PrefixCheckpointBytes(
        *config, config->engine.dynamic_batching->block_size, type);
    ASSERT_GT(bytes, 0u);
    const std::string policy = std::string{GetParam()} == "Vram"
                                   ? "\"vram_bytes\":" + std::to_string(3 * bytes)
                               : std::string{GetParam()} == "Ram"
                                   ? "\"ram_bytes\":" + std::to_string(3 * bytes)
                               : std::string{GetParam()} == "Disk"
                                   ? "\"disk_bytes\":" + std::to_string(3 * bytes)
                                   : "\"vram_bytes\":" + std::to_string(bytes) +
                                         ",\"ram_bytes\":" + std::to_string(bytes) +
                                         ",\"disk_bytes\":" + std::to_string(bytes);
    oga_config->Overlay(("{\"engine\":{\"dynamic_batching\":{\"snapshot_cache\":{" + policy + "}}}}").c_str());
  }
  auto model = OgaModel::Create(*oga_config);
  auto tokenizer = OgaTokenizer::Create(*model);

  const std::string patch =
      "diff --git a/math.py b/math.py\n@@ -1,2 +1,3 @@\n"
      "-def safe_divide(a, b): return a / b\n"
      "+def safe_divide(a, b):\n+    return a / b\n";
  std::string prompt = "<|im_start|>user\n";
  for (int i = 0; i < 10; ++i) {
    prompt += patch;
  }
  const std::string ending =
      "Propose a minimal patch that handles division by zero and explain the tradeoffs."
      "<|im_end|>\n<|im_start|>assistant\n";
  auto initial = OgaSequences::Create();
  tokenizer->Encode((prompt + ending).c_str(), *initial);
  for (int i = 0; i < 12; ++i) {
    prompt += patch;
  }
  auto extended = OgaSequences::Create();
  tokenizer->Encode((prompt + ending).c_str(), *extended);
  if (initial->SequenceCount(0) <= 512 || initial->SequenceCount(0) >= 768 ||
      extended->SequenceCount(0) <= 1024 || extended->SequenceCount(0) >= 1280) {
    FAIL() << "The tokenizer must place the initial prompt between 512 and 768 tokens and "
              "the extended prompt between 1024 and 1280 tokens; got "
           << initial->SequenceCount(0) << " and " << extended->SequenceCount(0) << ".";
  }

  auto engine = OgaEngine::Create(*model);
  const auto cold = RunGraphRestoreTurn(*engine, *initial);
  const auto extension = RunGraphRestoreTurn(*engine, *extended);
  const auto replay = RunGraphRestoreTurn(*engine, *extended);
  const auto repeated_replay = RunGraphRestoreTurn(*engine, *extended);
  EXPECT_EQ(cold.cached, 0u);
  EXPECT_EQ(extension.cached, 512u);
  EXPECT_EQ(replay.cached, 1024u);
  EXPECT_EQ(repeated_replay.cached, 1024u);
  EXPECT_GT(extension.accepted, 0u);
  EXPECT_GT(replay.proposed, 0u);
  EXPECT_GT(replay.accepted, 0u);
  EXPECT_EQ(replay.proposed, extension.proposed);
  EXPECT_EQ(replay.accepted, extension.accepted);
  EXPECT_EQ(repeated_replay.proposed, replay.proposed);
  EXPECT_EQ(repeated_replay.accepted, replay.accepted);
  EXPECT_EQ(replay.output, extension.output);
  EXPECT_EQ(repeated_replay.output, extension.output);

  // A third boundary forces the two earlier snapshots into RAM and disk in the mixed policy.
  for (int i = 0; i < 12; ++i) {
    prompt += patch;
  }
  auto longest = OgaSequences::Create();
  tokenizer->Encode((prompt + ending).c_str(), *longest);
  ASSERT_GT(longest->SequenceCount(0), 1536u);
  ASSERT_LT(longest->SequenceCount(0), 1792u);
  const auto growth = RunGraphRestoreTurn(*engine, *longest);
  EXPECT_EQ(growth.cached, 1024u);
  const auto ram_replay = RunGraphRestoreTurn(*engine, *extended);
  const auto disk_replay = RunGraphRestoreTurn(*engine, *initial);
  EXPECT_EQ(ram_replay.cached, 1024u);
  EXPECT_EQ(disk_replay.cached, 512u);
  EXPECT_GT(ram_replay.proposed, 0u);
  EXPECT_GT(disk_replay.proposed, 0u);
  EXPECT_EQ(ram_replay.output, extension.output);
  EXPECT_EQ(disk_replay.output, cold.output);
}

INSTANTIATE_TEST_SUITE_P(
    SnapshotTiers, Dflash2GraphRestoreTest,
    ::testing::Values("Vram", "Ram", "Disk", "Mixed"),
    [](const ::testing::TestParamInfo<const char*>& info) { return info.param; });

}  // namespace
}  // namespace Generators::test
