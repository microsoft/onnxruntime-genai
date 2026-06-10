// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Portions of this file consist of AI generated content.

// Tests for the controller-plugin escape hatch (issue #2114 §8, PR-E) -- the hook that lets a custom
// plugin OWN the per-step decode loop (bucket C: stateful controllers like Lookahead's Jacobi n-gram
// pool, nested cascades) that cannot be expressed as a static DAG or the `speculative` strategy.
//
// What is exercised here, and the honesty boundary:
//   * SCHEMA (always runnable): `pipeline.controller {library, entry_point, config}` parses into
//     Config::Pipeline::Controller. No ONNX load.
//   * PRIMITIVE SURFACE + HOST DISPATCH (in-tree stub): a tiny in-tree stub controller -- compiled
//     INTO this test binary, NOT a real .so -- drives the runtime through the exposed C-ABI step
//     primitives (get/append tokens, run a forward step, read logits + hidden states, rewind, query
//     EOS/length) and reproduces plain greedy decoding token-for-token, proving the primitive surface
//     is sufficient and the host adaptation + GenerateNextToken delegation are correct.
//   * DEFERRED / build-gated: loading a controller from a real external .so requires
//     USE_GENAI_PLUGINS=ON (dlopen). In the default OFF build, LoadDecodeController must fail loudly
//     with a clear "not enabled" message. We assert exactly that here; the .so round-trip itself is
//     deferred to a plugins-ON build (it cannot be built in-tree under USE_GENAI_PLUGINS=OFF).

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "span.h"
#include "generators.h"
#include "models/model.h"
#include "models/controller_host.h"
#include "models/plugin_loader.h"

#include "test_utils.h"

namespace {

// ---- In-tree greedy stub controller (NOT a real .so) -------------------------------------------
// Reproduces plain greedy decoding using only the exposed step primitives: read the next-position
// logits, take the argmax, and append it as the accepted token. One outer step == one token.

int StubGreedyStep(OgaDecodeController* /*self*/, OgaDecodeStepContext* ctx, int* tokens_emitted_out) {
  const float* logits = nullptr;
  size_t vocab = 0;
  if (ctx->GetLogits(ctx->host, &logits, &vocab) != 0 || vocab == 0)
    return 1;

  int32_t best = 0;
  float best_value = logits[0];
  for (size_t i = 1; i < vocab; ++i) {
    if (logits[i] > best_value) {
      best_value = logits[i];
      best = static_cast<int32_t>(i);
    }
  }

  if (ctx->AppendTokens(ctx->host, &best, 1) != 0)
    return 1;

  if (tokens_emitted_out) *tokens_emitted_out = 1;
  return 0;
}

void StubDestroy(OgaDecodeController* /*self*/) {}

// A dummy non-null controller handle; the greedy stub is stateless so it carries no real state.
OgaDecodeController* const kStubHandle = reinterpret_cast<OgaDecodeController*>(0x1);

int StubCreateController(void* /*config*/, OgaDecodeController** controller_out,
                         OgaControllerStepFn* step_out, OgaDecodeControllerDestroyFn* destroy_out) {
  *controller_out = kStubHandle;
  *step_out = &StubGreedyStep;
  *destroy_out = &StubDestroy;
  return 0;
}

std::shared_ptr<Generators::Model> LoadModel(const std::string& relative_path) {
  return Generators::CreateModel(Generators::GetOrtEnv(), (MODEL_PATH + relative_path).c_str());
}

// Plain greedy decoding on the model -- the reference the controller must reproduce.
std::vector<int32_t> GreedyBaseline(const std::shared_ptr<Generators::Model>& model,
                                    const std::vector<int32_t>& prompt, int max_length) {
  auto params = Generators::CreateGeneratorParams(*model);
  params->search.max_length = max_length;
  auto generator = Generators::CreateGenerator(*model, *params);
  generator->AppendTokens(Generators::cpu_span<const int32_t>(prompt.data(), prompt.size()));
  while (!generator->IsDone())
    generator->GenerateNextToken();
  auto seq = generator->GetSequence(0).CopyDeviceToCpu();
  return std::vector<int32_t>(seq.begin(), seq.end());
}

}  // namespace

// SCHEMA: `pipeline.controller` parses into Config::Pipeline::Controller (library + entry_point +
// opaque config). Injected via a JSON overlay onto an existing fixture so the test is self-contained.
TEST(ControllerHookTests, SchemaParsesControllerBlock) {
  const std::string overlay =
      R"({"pipeline":{"controller":{"library":"libmy_controller.so","entry_point":"OgaCreateDecodeController","config":"{\"pool\":3}"}}})";
  Generators::Config config(fs::path(MODEL_PATH "speculative-tiny/target"), overlay);

  ASSERT_TRUE(config.pipeline.present);
  ASSERT_TRUE(config.pipeline.controller.has_value());
  EXPECT_EQ(config.pipeline.controller->library, "libmy_controller.so");
  EXPECT_EQ(config.pipeline.controller->entry_point, "OgaCreateDecodeController");
  EXPECT_EQ(config.pipeline.controller->config, "{\"pool\":3}");
}

// Absent controller block leaves Config::Pipeline::controller unset (back-compat: no behavior change).
TEST(ControllerHookTests, AbsentControllerBlockLeavesNoController) {
  Generators::Config config(fs::path(MODEL_PATH "speculative-tiny/target"), /*json_overlay*/ "");
  EXPECT_FALSE(config.pipeline.controller.has_value());
}

#if !USE_DML

// PRIMITIVE SURFACE: an in-tree stub controller, driven through the host step-context primitives via
// ControllerHook::Step in a manual loop, reproduces plain greedy decoding token-for-token. This
// proves the exposed primitive surface (GetLogits + AppendTokens, plus IsDone) is sufficient to drive
// a custom decode loop, without any .so loading.
TEST(ControllerHookTests, InTreeStubReproducesGreedyViaPrimitives) {
  const std::vector<int32_t> prompt{1, 2, 3, 4};
  const int max_length = 24;

  auto model = LoadModel("speculative-tiny/target");
  const auto baseline = GreedyBaseline(model, prompt, max_length);

  auto params = Generators::CreateGeneratorParams(*model);
  params->search.max_length = max_length;
  auto generator = Generators::CreateGenerator(*model, *params);
  generator->AppendTokens(Generators::cpu_span<const int32_t>(prompt.data(), prompt.size()));

  auto hook = Generators::CreateControllerHook(&StubCreateController, /*config*/ "", /*keepalive*/ nullptr);
  ASSERT_NE(hook, nullptr);
  while (!generator->IsDone()) {
    const int emitted = hook->Step(*generator);
    EXPECT_EQ(emitted, 1);
  }

  auto seq = generator->GetSequence(0).CopyDeviceToCpu();
  const std::vector<int32_t> controller_output(seq.begin(), seq.end());
  EXPECT_EQ(controller_output, baseline);
}

// HOST DISPATCH: installing the same in-tree stub as the Generator's controller_ makes
// Generator::GenerateNextToken delegate the whole step to the controller. Running the standard
// `while (!IsDone()) GenerateNextToken()` outer loop must still reproduce greedy decoding, proving the
// delegation seam is wired correctly and leaves loop termination intact.
TEST(ControllerHookTests, GenerateNextTokenDelegatesToController) {
  const std::vector<int32_t> prompt{5, 6, 7};
  const int max_length = 24;

  auto model = LoadModel("speculative-tiny/target");
  const auto baseline = GreedyBaseline(model, prompt, max_length);

  auto params = Generators::CreateGeneratorParams(*model);
  params->search.max_length = max_length;
  auto generator = Generators::CreateGenerator(*model, *params);
  generator->AppendTokens(Generators::cpu_span<const int32_t>(prompt.data(), prompt.size()));
  generator->controller_ = Generators::CreateControllerHook(&StubCreateController, "", nullptr);

  while (!generator->IsDone())
    generator->GenerateNextToken();

  auto seq = generator->GetSequence(0).CopyDeviceToCpu();
  const std::vector<int32_t> controller_output(seq.begin(), seq.end());
  EXPECT_EQ(controller_output, baseline);
}

// PRIMITIVE SURFACE SMOKE: exercise the remaining primitives directly through the host step-context --
// sequence length, EOS id, token readback, hidden states, and rewind/rollback -- and assert they
// return sane, self-consistent values on a real decoder-pipeline model.
TEST(ControllerHookTests, StepContextPrimitivesAreConsistent) {
  const std::vector<int32_t> prompt{1, 2, 3, 4};
  const int max_length = 24;

  auto model = LoadModel("speculative-tiny/target");
  auto params = Generators::CreateGeneratorParams(*model);
  params->search.max_length = max_length;
  auto generator = Generators::CreateGenerator(*model, *params);
  generator->AppendTokens(Generators::cpu_span<const int32_t>(prompt.data(), prompt.size()));

  Generators::DecodeStepHost host{*generator};
  OgaDecodeStepContext ctx = host.Context();

  // Length + EOS (the fixture sets eos_token_id = 15).
  EXPECT_EQ(ctx.GetSequenceLength(ctx.host), prompt.size());
  EXPECT_EQ(ctx.GetEosTokenId(ctx.host), 15);

  // Token readback must equal the prompt.
  std::vector<int32_t> tokens(prompt.size());
  size_t count = 0;
  ASSERT_EQ(ctx.GetTokens(ctx.host, tokens.data(), tokens.size(), &count), 0);
  ASSERT_EQ(count, prompt.size());
  EXPECT_EQ(tokens, prompt);

  // Too-small buffer reports the required count and a non-zero status.
  size_t needed = 0;
  EXPECT_NE(ctx.GetTokens(ctx.host, tokens.data(), 0, &needed), 0);
  EXPECT_EQ(needed, prompt.size());

  // Logits are exposed with the model's vocab size (16).
  const float* logits = nullptr;
  size_t vocab = 0;
  ASSERT_EQ(ctx.GetLogits(ctx.host, &logits, &vocab), 0);
  EXPECT_EQ(vocab, 16u);
  ASSERT_NE(logits, nullptr);

  // Hidden states are exposed (this decoder-pipeline fixture declares a hidden_states output).
  const float* hidden = nullptr;
  size_t hidden_size = 0;
  EXPECT_EQ(ctx.GetHiddenStates(ctx.host, &hidden, &hidden_size), 0);
  EXPECT_GT(hidden_size, 0u);
  EXPECT_NE(hidden, nullptr);

  // Advance one token via the append primitive, sync the KV cache with a forward step, then roll
  // back to the prompt length.
  ASSERT_EQ(ctx.GetLogits(ctx.host, &logits, &vocab), 0);
  int32_t t0 = 0;
  for (size_t i = 1; i < vocab; ++i)
    if (logits[i] > logits[t0]) t0 = static_cast<int32_t>(i);
  ASSERT_EQ(ctx.AppendTokens(ctx.host, &t0, 1), 0);
  EXPECT_EQ(ctx.GetSequenceLength(ctx.host), prompt.size() + 1);

  ASSERT_EQ(ctx.GetLogits(ctx.host, &logits, &vocab), 0);  // forward step syncs the KV cache to the new length
  ASSERT_EQ(ctx.RewindTo(ctx.host, prompt.size()), 0);
  EXPECT_EQ(ctx.GetSequenceLength(ctx.host), prompt.size());
}

#endif  // !USE_DML

// DEFERRED / BUILD-GATED: loading a controller from a real external .so requires USE_GENAI_PLUGINS.
// In the default OFF build LoadDecodeController must throw a clear, actionable error; the real .so
// round-trip is deferred to a plugins-ON build. (In an ON build this still throws because the bogus
// library path cannot be loaded, so the assertion holds in both modes.)
TEST(ControllerHookTests, LoadDecodeControllerGatedByPlugins) {
  Generators::Config::Pipeline::Controller controller;
  controller.library = "libgenai_nonexistent_controller.so";
  controller.entry_point = "OgaCreateDecodeController";
  EXPECT_THROW({ Generators::LoadDecodeController(controller); }, std::runtime_error);
}

#if !USE_GENAI_PLUGINS
TEST(ControllerHookTests, DisabledBuildReportsClearControllerMessage) {
  Generators::Config::Pipeline::Controller controller;
  controller.library = "libgenai_nonexistent_controller.so";
  controller.entry_point = "OgaCreateDecodeController";
  try {
    Generators::LoadDecodeController(controller);
    FAIL() << "Expected LoadDecodeController to throw in a USE_GENAI_PLUGINS=OFF build.";
  } catch (const std::runtime_error& e) {
    const std::string message = e.what();
    EXPECT_NE(message.find("not enabled"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("USE_GENAI_PLUGINS=ON"), std::string::npos) << "message: " << message;
  }
}
#endif
