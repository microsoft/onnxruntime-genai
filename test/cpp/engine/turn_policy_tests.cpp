// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for the per-turn generation policy: how each turn resolves its own policy from model
// defaults plus explicit overrides, which combinations admission rejects before mutating anything,
// and how the turn seed drives the Request's random streams across turns and rollbacks.

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engine_test_doubles.h"
#include "engine_test_helpers.h"
#include "engine/turn_policy.h"

namespace Generators {
namespace test {
namespace {

std::vector<int32_t> Prompt() { return {2, 3, 4}; }

// Three equally likely candidates, so a sampled turn's output is decided entirely by the random
// stream while a greedy turn deterministically takes the lowest-indexed maximum.
constexpr std::array<int32_t, 3> kSamplingCandidates{20, 21, 22};

// Reports a different scoring device type while delegating every real operation to the model's own
// device, so a capability keyed by device type can be exercised on a CPU-only machine.
class ScoringDeviceTypeOverride final : public DeviceInterface {
 public:
  ScoringDeviceTypeOverride(DeviceInterface& inner, DeviceType type)
      : inner_{inner}, type_{type} {}

  DeviceType GetType() const override { return type_; }
  void InitOrt(const OrtApi& api, Ort::Allocator& allocator) override {
    inner_.InitOrt(api, allocator);
  }
  Ort::Allocator& GetAllocator() override { return inner_.GetAllocator(); }
  std::unique_ptr<OrtMemoryInfo> GetMemoryInfo() const override {
    return inner_.GetMemoryInfo();
  }
  std::string GetExecutionProviderName() const override {
    return inner_.GetExecutionProviderName();
  }
  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return inner_.AllocateBase(size);
  }
  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* memory, size_t size) override {
    return inner_.WrapMemoryBase(memory, size);
  }
  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override {
    return inner_.CreateGreedy(params);
  }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override {
    return inner_.CreateBeam(params);
  }
  std::unique_ptr<KeyValueCache> CreateKeyValueCache(State& state) override {
    return inner_.CreateKeyValueCache(state);
  }
  void Synchronize() override { inner_.Synchronize(); }

 private:
  DeviceInterface& inner_;
  DeviceType type_;
};

TurnOptions SampledTurn(std::optional<uint64_t> seed, size_t max_generated_tokens) {
  TurnOptions options;
  options.do_sample = true;
  options.top_k = 3;
  options.temperature = 1.0f;
  options.seed = seed;
  options.max_generated_tokens = max_generated_tokens;
  return options;
}

class TurnPolicyTest : public ::testing::Test {
 protected:
  void SetUp() override { model_ = LoadDummyDecoderModel(); }

  // Runs one whole turn and returns the tokens it generated. The step bound keeps a regression
  // that stops completing turns from hanging the suite.
  std::vector<int32_t> RunTurn(DoublesEngine& engine,
                               const std::shared_ptr<Request>& request,
                               std::span<const int32_t> tokens,
                               const TurnOptions& options) {
    request->BeginTurn(tokens, options);
    std::vector<int32_t> generated;
    std::array<EngineEvent, 8> storage;
    for (int step = 0; step < 64 && !request->IsTurnComplete(); ++step) {
      const size_t count = engine.engine->Run(storage);
      for (size_t i = 0; i < count; ++i) {
        if (storage[i].flags & EngineEventFlagToken) {
          generated.push_back(storage[i].token);
        }
      }
    }
    EXPECT_TRUE(request->IsTurnComplete());
    return generated;
  }

  std::shared_ptr<Model> model_;
};

// The capability predicate exists so an unsupported no_repeat_ngram_size is refused at admission
// instead of throwing after the model has already run. Verifying that means exercising both sides:
// the CPU search really does suppress a repeated n-gram, and a scoring device whose search does not
// implement it never gets the chance to try.
TEST_F(TurnPolicyTest, NoRepeatNgramSuppressesRepeatsOnCpuAndIsRejectedElsewhere) {
  constexpr int32_t kForcedToken = 5;
  ASSERT_NE(kForcedToken, EosToken(*model_));

  const auto run_with_ngram_size = [&](std::optional<int> no_repeat_ngram_size) {
    auto engine = MakeDoublesEngine(model_, /*capacity=*/8, kForcedToken);
    auto request = CreateEngineRequest(engine.engine);
    TurnOptions options;
    options.max_generated_tokens = 3;
    options.no_repeat_ngram_size = no_repeat_ngram_size;
    return RunTurn(engine, request, Prompt(), options);
  };

  // Without it the scripted logits force the same token every step.
  EXPECT_EQ(run_with_ngram_size(std::nullopt),
            std::vector<int32_t>(3, kForcedToken));

  // With it the CPU search bans the token that would repeat the trailing bigram, so the third step
  // has to select something else even though the forced token still has the highest raw logit.
  const auto suppressed = run_with_ngram_size(2);
  ASSERT_EQ(suppressed.size(), 3u);
  EXPECT_EQ(suppressed[0], kForcedToken);
  EXPECT_EQ(suppressed[1], kForcedToken);
  EXPECT_NE(suppressed[2], kForcedToken);

  // A scoring device whose Search leaves ApplyNoRepeatNgram unimplemented is rejected at admission,
  // before the Request is mutated, so the same Request is still usable without the option.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, kForcedToken);
  auto request = CreateEngineRequest(engine.engine);
  ScoringDeviceTypeOverride device{*model_->p_device_scoring_, DeviceType::CUDA};
  ScopedScoringDevice scoped_device{*model_, device};
  ASSERT_FALSE(SupportsNoRepeatNgram(DeviceType::CUDA));

  TurnOptions options;
  options.no_repeat_ngram_size = 2;
  options.max_generated_tokens = 3;
  try {
    request->BeginTurn(Prompt(), options);
    FAIL() << "Expected no_repeat_ngram_size to be rejected on this scoring device.";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("no_repeat_ngram_size"), std::string::npos) << message;
    EXPECT_NE(message.find("scoring device"), std::string::npos) << message;
  }
  EXPECT_EQ(request->Status(), RequestStatus::Unassigned);
  EXPECT_EQ(request->CurrentSequenceLength(), 0);

  options.no_repeat_ngram_size.reset();
  EXPECT_EQ(request->BeginTurn(Prompt(), options), 1u);
}

// Under any greedy resolution, a scalar that is itself a request for greedy selection, or that
// restricts nothing, is honored exactly as written rather than rejected.
TEST_F(TurnPolicyTest, GreedyResolutionAcceptsConsistentAndNeutralScalars) {
  const auto run_greedy_turn = [&](const TurnOptions& turn_options) {
    auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
    engine.executor->SetSamplingCandidateTokens(
        {kSamplingCandidates.begin(), kSamplingCandidates.end()});
    auto request = CreateEngineRequest(engine.engine);
    TurnOptions options = turn_options;
    options.max_generated_tokens = 4;
    const auto generated = RunTurn(engine, request, Prompt(), options);
    // Every candidate has the same logit, so only top-logit selection produces the lowest index
    // four times in a row.
    EXPECT_EQ(generated, std::vector<int32_t>(4, kSamplingCandidates.front()));
  };

  {
    // top_k == 1 on its own asks for greedy selection.
    TurnOptions options;
    options.top_k = 1;
    run_greedy_turn(options);
  }
  {
    // temperature == 0 on its own asks for greedy selection.
    TurnOptions options;
    options.temperature = 0.0f;
    run_greedy_turn(options);
  }
  {
    // Two consistent ways of saying the same thing.
    TurnOptions options;
    options.do_sample = false;
    options.top_k = 1;
    run_greedy_turn(options);
  }
  {
    TurnOptions options;
    options.do_sample = false;
    options.temperature = 0.0f;
    run_greedy_turn(options);
  }
  {
    // top_p == 1 and top_k == 0 restrict nothing, so neither contradicts top-logit selection.
    TurnOptions options;
    options.do_sample = false;
    options.top_p = 1.0f;
    options.top_k = 0;
    run_greedy_turn(options);
  }
  {
    // An explicit sampling request that its own scalars override is still coherent.
    TurnOptions options;
    options.do_sample = true;
    options.top_k = 1;
    options.temperature = 0.0f;
    run_greedy_turn(options);
  }
}

// A scalar that only makes sense for a distribution the turn will never draw from is a caller
// mistake, and it is rejected before the Request is touched.
TEST_F(TurnPolicyTest, GreedyResolutionRejectsContradictoryStochasticScalars) {
  const auto expect_rejected = [&](const TurnOptions& turn_options,
                                   std::string_view named_option) {
    auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
    auto request = CreateEngineRequest(engine.engine);
    try {
      request->BeginTurn(Prompt(), turn_options);
      FAIL() << "Expected a contradictory greedy policy to be rejected.";
    } catch (const std::runtime_error& error) {
      const std::string message = error.what();
      EXPECT_NE(message.find("top logit"), std::string::npos) << message;
      EXPECT_NE(message.find(named_option), std::string::npos) << message;
    }
    EXPECT_EQ(request->Status(), RequestStatus::Unassigned);
    EXPECT_EQ(request->CurrentSequenceLength(), 0);

    // The rejection mutated nothing, so an ordinary turn still admits on the same Request.
    EXPECT_EQ(request->BeginTurn(Prompt(), TurnOptions{}), 1u);
  };

  {
    TurnOptions options;
    options.do_sample = false;
    options.temperature = 0.7f;
    expect_rejected(options, "temperature");
  }
  {
    TurnOptions options;
    options.do_sample = true;
    options.top_k = 1;
    options.top_p = 0.9f;
    expect_rejected(options, "top_p");
  }
  {
    TurnOptions options;
    options.do_sample = false;
    options.top_k = 40;
    expect_rejected(options, "top_k");
  }
  {
    TurnOptions options;
    options.do_sample = true;
    options.temperature = 0.0f;
    options.top_k = 40;
    expect_rejected(options, "top_k");
  }
}

// A model whose own search defaults resolve to greedy silently overrides an explicit
// do_sample=true. The caller cannot see those defaults through TurnOptions, so admission rejects
// the turn and names the model-supplied cause rather than quietly selecting the top logit.
TEST_F(TurnPolicyTest, ExplicitDoSampleIsRejectedWhenAModelDefaultForcesGreedy) {
  const auto load_model = [](int top_k, float temperature) {
    auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
    config->search.top_k = top_k;
    config->search.temperature = temperature;
    return CreateModel(GetOrtEnv(), std::move(config));
  };

  const auto expect_rejected = [&](const std::shared_ptr<Model>& model,
                                   const std::vector<std::string>& named_defaults) {
    auto engine = MakeDoublesEngine(model, /*capacity=*/8, EosToken(*model));
    auto request = CreateEngineRequest(engine.engine);
    TurnOptions options;
    options.do_sample = true;
    try {
      request->BeginTurn(Prompt(), options);
      FAIL() << "Expected an explicit do_sample=true to be rejected.";
    } catch (const std::runtime_error& error) {
      const std::string message = error.what();
      EXPECT_NE(message.find("do_sample=true"), std::string::npos) << message;
      for (const auto& named_default : named_defaults) {
        EXPECT_NE(message.find(named_default), std::string::npos) << message;
      }
    }
    // Nothing was mutated, so the same Request still admits an ordinary turn.
    EXPECT_EQ(request->Status(), RequestStatus::Unassigned);
    EXPECT_EQ(request->CurrentSequenceLength(), 0);
    EXPECT_EQ(request->BeginTurn(Prompt(), TurnOptions{}), 1u);
  };

  // Either model default is enough on its own, and a model that supplies both is named for both.
  expect_rejected(load_model(/*top_k=*/1, /*temperature=*/1.0f), {"search.top_k = 1"});
  expect_rejected(load_model(/*top_k=*/50, /*temperature=*/0.0f), {"search.temperature = 0"});
  expect_rejected(load_model(/*top_k=*/1, /*temperature=*/0.0f),
                  {"search.top_k = 1", "search.temperature = 0"});
}

// The way out is to override the field the model set, which is exactly what the rejection asks for.
TEST_F(TurnPolicyTest, OverridingTheModelGreedyDefaultSamplesTheTurn) {
  const auto run_sampled_turn = [&](int model_top_k, float model_temperature,
                                    const TurnOptions& overrides) {
    auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
    config->search.top_k = model_top_k;
    config->search.temperature = model_temperature;
    auto model = CreateModel(GetOrtEnv(), std::move(config));
    auto engine = MakeDoublesEngine(model, /*capacity=*/8, EosToken(*model));
    engine.executor->SetSamplingCandidateTokens(
        {kSamplingCandidates.begin(), kSamplingCandidates.end()});
    auto request = CreateEngineRequest(engine.engine);
    TurnOptions options = overrides;
    options.do_sample = true;
    options.seed = uint64_t{7};
    options.max_generated_tokens = 8;
    const auto generated = RunTurn(engine, request, Prompt(), options);
    // Every candidate has the same logit, so only a sampled turn deviates from the lowest index.
    EXPECT_EQ(generated.size(), 8u);
    EXPECT_NE(generated, std::vector<int32_t>(8, kSamplingCandidates.front()));
  };

  {
    // The model pins top_k to 1; the turn has to name top_k itself to sample.
    TurnOptions overrides;
    overrides.top_k = 3;
    run_sampled_turn(/*model_top_k=*/1, /*model_temperature=*/1.0f, overrides);
  }
  {
    // The model pins temperature to 0; the turn has to name temperature itself.
    TurnOptions overrides;
    overrides.temperature = 1.0f;
    run_sampled_turn(/*model_top_k=*/3, /*model_temperature=*/0.0f, overrides);
  }
}

// A caller who spells greedy out on the turn has chosen it, so do_sample=true alongside it is the
// same coherent combination it is on a model with ordinary sampling defaults.
TEST_F(TurnPolicyTest, ExplicitGreedySpellingIsAcceptedOnAGreedyModel) {
  const auto run_greedy_turn = [&](int model_top_k, float model_temperature,
                                   const TurnOptions& turn_options) {
    auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
    config->search.top_k = model_top_k;
    config->search.temperature = model_temperature;
    auto model = CreateModel(GetOrtEnv(), std::move(config));
    auto engine = MakeDoublesEngine(model, /*capacity=*/8, EosToken(*model));
    engine.executor->SetSamplingCandidateTokens(
        {kSamplingCandidates.begin(), kSamplingCandidates.end()});
    auto request = CreateEngineRequest(engine.engine);
    TurnOptions options = turn_options;
    options.max_generated_tokens = 4;
    const auto generated = RunTurn(engine, request, Prompt(), options);
    EXPECT_EQ(generated, std::vector<int32_t>(4, kSamplingCandidates.front()));
  };

  {
    // do_sample=true with the caller's own top_k of 1: the model default is not what decided this.
    TurnOptions options;
    options.do_sample = true;
    options.top_k = 1;
    run_greedy_turn(/*model_top_k=*/1, /*model_temperature=*/1.0f, options);
  }
  {
    TurnOptions options;
    options.do_sample = true;
    options.temperature = 0.0f;
    run_greedy_turn(/*model_top_k=*/50, /*model_temperature=*/0.0f, options);
  }
  {
    // Not asking to sample at all leaves the model's greedy defaults entirely uncontroversial.
    run_greedy_turn(/*model_top_k=*/1, /*model_temperature=*/1.0f, TurnOptions{});
    TurnOptions options;
    options.do_sample = false;
    run_greedy_turn(/*model_top_k=*/50, /*model_temperature=*/0.0f, options);
  }
}

// Temperature 1 rescales nothing, exactly like top_p 1 and top_k 0, so it is neutral under a greedy
// resolution rather than a request for a distribution the turn will never draw from.
TEST_F(TurnPolicyTest, GreedyResolutionAcceptsNeutralTemperature) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  engine.executor->SetSamplingCandidateTokens(
      {kSamplingCandidates.begin(), kSamplingCandidates.end()});
  auto request = CreateEngineRequest(engine.engine);

  TurnOptions options;
  options.do_sample = false;
  options.temperature = 1.0f;
  options.max_generated_tokens = 4;
  const auto generated = RunTurn(engine, request, Prompt(), options);
  EXPECT_EQ(generated, std::vector<int32_t>(4, kSamplingCandidates.front()));
}

TEST_F(TurnPolicyTest, UnsetOptionsResolveToModelDefaults) {
  const auto& defaults = model_->config_->search;
  const auto policy = ResolveTurnPolicy(defaults, TurnOptions{});

  EXPECT_EQ(policy.do_sample, defaults.do_sample);
  EXPECT_EQ(policy.temperature, defaults.temperature);
  EXPECT_EQ(policy.top_p, defaults.top_p);
  EXPECT_EQ(policy.top_k, defaults.top_k);
  EXPECT_EQ(policy.repetition_penalty, defaults.repetition_penalty);
  EXPECT_EQ(policy.no_repeat_ngram_size, defaults.no_repeat_ngram_size);
  // Turn-relative limits have no model default at all.
  EXPECT_EQ(policy.min_generated_tokens, 0u);
  EXPECT_FALSE(policy.max_generated_tokens.has_value());
}

TEST_F(TurnPolicyTest, ResetRestoresEveryOptionToUnset) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto request = CreateEngineRequest(engine.engine);

  TurnOptions options;
  options.request = request;
  options.do_sample = true;
  options.temperature = 0.5f;
  options.top_p = 0.9f;
  options.top_k = 7;
  options.repetition_penalty = 1.5f;
  options.no_repeat_ngram_size = 3;
  options.min_generated_tokens = 2;
  options.max_generated_tokens = 5;
  options.seed = 0u;
  options.stop_strings = {"STOP"};
  options.guidance_type = "regex";
  options.guidance_data = "[0-9]+";

  options.Reset();

  EXPECT_EQ(options.request.lock(), request);
  EXPECT_FALSE(options.do_sample.has_value());
  EXPECT_FALSE(options.temperature.has_value());
  EXPECT_FALSE(options.top_p.has_value());
  EXPECT_FALSE(options.top_k.has_value());
  EXPECT_FALSE(options.repetition_penalty.has_value());
  EXPECT_FALSE(options.no_repeat_ngram_size.has_value());
  EXPECT_FALSE(options.min_generated_tokens.has_value());
  EXPECT_FALSE(options.max_generated_tokens.has_value());
  EXPECT_FALSE(options.seed.has_value());
  EXPECT_TRUE(options.stop_strings.empty());
  EXPECT_TRUE(options.guidance_type.empty());
  EXPECT_TRUE(options.guidance_data.empty());
}

// Policy never carries over: a turn that overrides nothing decodes with the model's own defaults
// again, even directly after a turn that overrode everything.
TEST_F(TurnPolicyTest, PolicyIsResolvedAfreshForEveryTurn) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  engine.executor->SetSamplingCandidateTokens(
      {kSamplingCandidates.begin(), kSamplingCandidates.end()});
  auto request = CreateEngineRequest(engine.engine);

  const auto sampled = RunTurn(engine, request, Prompt(),
                               SampledTurn(/*seed=*/uint64_t{7}, /*max_generated_tokens=*/6));
  ASSERT_EQ(sampled.size(), 6u);
  EXPECT_NE(sampled, std::vector<int32_t>(6, kSamplingCandidates.front()));

  TurnOptions defaults;
  defaults.max_generated_tokens = 6;
  const auto greedy = RunTurn(engine, request, std::array<int32_t, 1>{5}, defaults);

  // The model's search defaults are greedy, so every step takes the same lowest-indexed maximum.
  EXPECT_EQ(greedy, std::vector<int32_t>(6, kSamplingCandidates.front()));
}

TEST_F(TurnPolicyTest, SameSeedReproducesTheSameSampledTurn) {
  const auto sample_with_seed = [&](uint64_t seed) {
    auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
    engine.executor->SetSamplingCandidateTokens(
        {kSamplingCandidates.begin(), kSamplingCandidates.end()});
    auto request = CreateEngineRequest(engine.engine);
    return RunTurn(engine, request, Prompt(),
                   SampledTurn(seed, /*max_generated_tokens=*/8));
  };

  // Zero is an ordinary deterministic seed, and a seed with a nonzero high half -- which the
  // classic int seed could never express -- is too.
  for (const uint64_t seed : {uint64_t{0}, uint64_t{1234},
                              uint64_t{0x1234'5678'9abc'def0}}) {
    const auto first = sample_with_seed(seed);
    ASSERT_EQ(first.size(), 8u);
    EXPECT_EQ(sample_with_seed(seed), first);
  }
}

TEST_F(TurnPolicyTest, FullWidthSeedsAreDistinctFromTheirLowHalf) {
  const auto sample_with_seed = [&](uint64_t seed) {
    auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
    engine.executor->SetSamplingCandidateTokens(
        {kSamplingCandidates.begin(), kSamplingCandidates.end()});
    auto request = CreateEngineRequest(engine.engine);
    return RunTurn(engine, request, Prompt(),
                   SampledTurn(seed, /*max_generated_tokens=*/12));
  };

  constexpr uint64_t low_half = 0x9abc'def0;
  EXPECT_NE(sample_with_seed(low_half | (uint64_t{0x1234'5678} << 32)),
            sample_with_seed(low_half));
}

// An omitted seed continues the stream the previous turn left off at; reusing an explicitly seeded
// options object deliberately restarts it.
TEST_F(TurnPolicyTest, OmittedSeedContinuesTheStreamAndAnExplicitSeedRestartsIt) {
  const auto run_two_turns = [&](std::optional<uint64_t> second_turn_seed) {
    auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
    engine.executor->SetSamplingCandidateTokens(
        {kSamplingCandidates.begin(), kSamplingCandidates.end()});
    auto request = CreateEngineRequest(engine.engine);
    auto first = RunTurn(engine, request, Prompt(),
                         SampledTurn(uint64_t{99}, /*max_generated_tokens=*/8));
    auto second = RunTurn(engine, request, std::array<int32_t, 1>{5},
                          SampledTurn(second_turn_seed, /*max_generated_tokens=*/8));
    return std::pair{std::move(first), std::move(second)};
  };

  const auto [seeded_first, continued] = run_two_turns(std::nullopt);
  const auto [reseeded_first, reseeded] = run_two_turns(uint64_t{99});

  ASSERT_EQ(seeded_first.size(), 8u);
  // Both runs start identically, which is what makes the second turns comparable.
  EXPECT_EQ(reseeded_first, seeded_first);
  // Reseeding with the first turn's seed restarts that exact stream, so the reseeded second turn
  // reproduces the first turn's draws while the continued one does not.
  EXPECT_EQ(reseeded, seeded_first);
  EXPECT_NE(continued, seeded_first);
}

// A step that is rolled back after the reseed was applied must reseed its retry identically, so the
// turn's output does not depend on how many times the batch was aborted.
TEST_F(TurnPolicyTest, RolledBackStepReseedsTheRetryIdentically) {
  const auto run_with_injected_abort = [&](bool inject_abort) {
    auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
    engine.executor->SetSamplingCandidateTokens(
        {kSamplingCandidates.begin(), kSamplingCandidates.end()});
    auto request = CreateEngineRequest(engine.engine);
    auto options = SampledTurn(uint64_t{31337}, /*max_generated_tokens=*/8);
    options.request = request;
    request->BeginTurn(Prompt(), options);

    if (inject_abort) {
      // Fails inside the transaction's post-processing, after the pending reseed has already been
      // applied to the Request's stream and before anything is staged.
      engine.executor->SetNextFailure(ScriptedExecutionFailure::PostProcessing);
      const auto aborted = RunOne(*engine.engine);
      EXPECT_EQ(aborted.flags, EngineEventFlagRetryable);
      EXPECT_EQ(request->TurnGeneratedTokens(), 0u);
      // Rollback rewound the stream the reseed had already restarted, so the reseed is still
      // pending and the retry applies it again.
      EXPECT_TRUE(request->HasPendingTurnSeed());
    }

    std::vector<int32_t> generated;
    std::array<EngineEvent, 8> storage;
    for (int step = 0; step < 64 && !request->IsTurnComplete(); ++step) {
      const size_t count = engine.engine->Run(storage);
      for (size_t i = 0; i < count; ++i) {
        if (storage[i].flags & EngineEventFlagToken) {
          generated.push_back(storage[i].token);
        }
      }
    }
    EXPECT_TRUE(request->IsTurnComplete());
    return generated;
  };

  const auto without_abort = run_with_injected_abort(false);
  ASSERT_EQ(without_abort.size(), 8u);
  EXPECT_EQ(run_with_injected_abort(true), without_abort);
}

// A turn that ends before a sampling step commits takes its reseed with it: the durable basis is
// untouched, so the Request continues exactly the stream it was on.
TEST_F(TurnPolicyTest, CanceledTurnDiscardsItsPendingReseed) {
  // The model fixes the seed basis so two independent Requests start on the same stream; an
  // unseeded Request draws a 64-bit basis instead, which is deliberately not reproducible.
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
  config->search.random_seed = 4321;
  auto model = CreateModel(GetOrtEnv(), std::move(config));

  const auto run_after_canceled_turn = [&](std::optional<uint64_t> canceled_turn_seed) {
    auto engine = MakeDoublesEngine(model, /*capacity=*/8, EosToken(*model));
    engine.executor->SetSamplingCandidateTokens(
        {kSamplingCandidates.begin(), kSamplingCandidates.end()});
    auto request = CreateEngineRequest(engine.engine);

    auto canceled = SampledTurn(canceled_turn_seed, /*max_generated_tokens=*/8);
    canceled.request = request;
    const uint64_t turn_id = request->BeginTurn(Prompt(), canceled);
    EXPECT_TRUE(request->Cancel(turn_id));
    // Drain the cancellation so the Request is ready for another turn. Nothing was sampled.
    while (engine.engine->HasPendingRequests()) {
      static_cast<void>(RunOne(*engine.engine));
    }
    EXPECT_EQ(request->FinishReason(), GenerationFinishReason::Canceled);
    EXPECT_EQ(request->TurnGeneratedTokens(), 0u);
    // The canceled turn took its reseed with it instead of leaving it to be applied, or promoted,
    // by whatever runs next.
    EXPECT_FALSE(request->HasPendingTurnSeed());

    return RunTurn(engine, request, std::array<int32_t, 1>{5},
                   SampledTurn(std::nullopt, /*max_generated_tokens=*/8));
  };

  const auto after_seeded_cancel = run_after_canceled_turn(uint64_t{99});
  ASSERT_EQ(after_seeded_cancel.size(), 8u);
  // Identical to the run where the canceled turn never asked for a seed at all: the discarded
  // reseed neither advanced nor replaced the Request's durable basis.
  EXPECT_EQ(after_seeded_cancel, run_after_canceled_turn(std::nullopt));

  // And the Request is still perfectly seedable afterwards.
  auto engine = MakeDoublesEngine(model, /*capacity=*/8, EosToken(*model));
  engine.executor->SetSamplingCandidateTokens(
      {kSamplingCandidates.begin(), kSamplingCandidates.end()});
  auto request = CreateEngineRequest(engine.engine);
  const auto seeded = RunTurn(engine, request, Prompt(),
                              SampledTurn(uint64_t{99}, /*max_generated_tokens=*/8));
  EXPECT_NE(seeded, after_seeded_cancel);
}

TEST_F(TurnPolicyTest, MinGeneratedTokensMasksEndOfSequence) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto request = CreateEngineRequest(engine.engine);

  TurnOptions options;
  options.min_generated_tokens = 3;
  options.max_generated_tokens = 3;
  // The executor forces the end-of-stream token every step; the floor must keep the turn running
  // until it has generated three tokens anyway.
  const auto generated = RunTurn(engine, request, Prompt(), options);

  EXPECT_EQ(generated.size(), 3u);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::TurnLimit);
  for (const int32_t token : generated) {
    EXPECT_NE(token, EosToken(*model_));
  }

  // The floor is turn-scoped: the next turn stops on the very first forced end-of-stream token.
  const auto next = RunTurn(engine, request, std::array<int32_t, 1>{5}, TurnOptions{});
  EXPECT_TRUE(next.empty());
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::EosToken);
}

TEST_F(TurnPolicyTest, UnsatisfiableMinimumIsRejectedBeforeMutation) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  const auto prompt = Prompt();
  auto request =
      CreateEngineRequest(engine.engine, /*max_session_tokens=*/prompt.size() + 2);

  TurnOptions options;
  options.min_generated_tokens = 3;
  EXPECT_THROW(request->BeginTurn(prompt, options), std::runtime_error);
  EXPECT_EQ(request->Status(), RequestStatus::Unassigned);
  EXPECT_EQ(request->CurrentSequenceLength(), 0);

  // The Request is untouched, so a satisfiable minimum still admits normally.
  options.min_generated_tokens = 2;
  EXPECT_EQ(request->BeginTurn(prompt, options), 1u);
}

TEST_F(TurnPolicyTest, StaticEngineRejectsATurnSeedBeforeMutation) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
  config->engine.dynamic_batching.reset();
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  auto cache = std::make_shared<RecordingCacheManager>(
      model, /*capacity=*/8, /*trace=*/nullptr, /*supports_dynamic_batching=*/false);
  EngineDependencies dependencies{
      cache, Scheduler::Create(model, cache),
      std::make_unique<RecordingModelExecutor>(model, cache, EosToken(*model))};
  auto engine = std::make_shared<Engine>(model, std::move(dependencies));
  auto request = engine->CreateRequest();

  TurnOptions options;
  options.seed = 1234u;
  try {
    request->BeginTurn(Prompt(), options);
    FAIL() << "Expected a per-turn seed to be rejected on a static Engine.";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("dynamic batching"), std::string::npos);
  }
  EXPECT_EQ(request->Status(), RequestStatus::Unassigned);
  EXPECT_EQ(request->CurrentSequenceLength(), 0);

  // Everything the static path does support still admits on the same untouched Request.
  options.seed.reset();
  options.max_generated_tokens = 1;
  EXPECT_EQ(request->BeginTurn(Prompt(), options), 1u);
}

}  // namespace
}  // namespace test
}  // namespace Generators
