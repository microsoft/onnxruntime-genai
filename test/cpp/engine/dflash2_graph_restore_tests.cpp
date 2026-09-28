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

TEST(Dflash2GraphRestoreTest, ExtendedCachedPrefixReplaysWithoutStaleGraphOrLostDrafts) {
  const char* model_path = std::getenv("D_FLASH2_GRAPH_TEST_MODEL");
  if (!model_path) {
    GTEST_SKIP() << "Set D_FLASH2_GRAPH_TEST_MODEL to a graph-enabled Qwen model with 256-token "
                    "blocks and 512-token prefill, and pass --ep_dir with its CUDA plugin.";
  }
  auto config = CreateConfig(GetOrtEnv(), model_path);
  ASSERT_TRUE(IsGraphCaptureEnabled(config->model.decoder.session_options));
  ASSERT_TRUE(IsGraphCaptureEnabled(CreateDflash2Config(*config)->model.decoder.session_options));
  auto model = OgaModel::Create(model_path);
  auto tokenizer = OgaTokenizer::Create(*model);
  auto engine = OgaEngine::Create(*model);

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
  ASSERT_EQ(initial->SequenceCount(0), 534u);
  ASSERT_EQ(extended->SequenceCount(0), 1146u);

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
  EXPECT_GT(repeated_replay.accepted, 0u);
  EXPECT_EQ(replay.output, extension.output);
  EXPECT_EQ(repeated_replay.output, extension.output);
}

}  // namespace
}  // namespace Generators::test
