// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "config.h"
#include "decoding/whisper_timestamp_logits_processor.h"
#include "ep_registration.h"
#include "generator/generators.h"
#include "models/model.h"
#include "models/preprocessing/genai_tokenizer.h"
#include "ort_genai.h"
#include "search.h"
#include "telemetry_test_environment.h"

namespace Generators {

DeviceType g_test_device = DeviceType::CPU;

namespace {

struct TestWhisperState;

struct TestWhisperModel final : Model {
  explicit TestWhisperModel(std::unique_ptr<Config> config)
      : Model{std::move(config)} {}

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t>,
                                     const GeneratorParams& params) const override;

  std::vector<std::vector<float>> scripted_logits;
};

struct TestWhisperState final : State {
  TestWhisperState(const TestWhisperModel& model, const GeneratorParams& params)
      : State{params, model},
        model_{model},
        logits_{params.p_device->Allocate<float>(
            static_cast<size_t>(params.BatchBeamSize()) * params.config.model.vocab_size)} {
    logits_.Zero();
  }

  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) override {
    if (run_index_ < model_.scripted_logits.size()) {
      const auto& scripted = model_.scripted_logits[run_index_++];
      if (scripted.size() != logits_.size())
        throw std::runtime_error("Scripted logits size does not match the test model");
      std::copy(scripted.begin(), scripted.end(), logits_.CpuSpan().begin());
      logits_.CopyCpuToDevice();
    }
    return logits_;
  }

  const TestWhisperModel& model_;
  DeviceSpan<float> logits_;
  size_t run_index_{};
};

std::unique_ptr<State> TestWhisperModel::CreateState(
    DeviceSpan<int32_t>, const GeneratorParams& params) const {
  return std::make_unique<TestWhisperState>(*this, params);
}

WhisperTimestampLogitsProcessor CreateProcessor(std::optional<int> max_initial_timestamp_index = {}) {
  return WhisperTimestampLogitsProcessor{
      {.timestamp_begin = 5,
       .eot_token = 2,
       .no_timestamps_token = 4,
       .max_initial_timestamp_index = max_initial_timestamp_index}};
}

std::shared_ptr<TestWhisperModel> CreateTestModel(bool whisper_timestamps = true,
                                                  int batch_size = 1) {
  auto config = std::make_unique<Config>();
  config->model.type = "whisper";
  config->model.vocab_size = 8;
  config->model.context_length = 16;
  config->model.pad_token_id = 3;
  config->model.eos_token_id = {2};
  config->model.timestamp_begin_token_id = 5;
  config->model.no_timestamps_token_id = 4;
  config->search.max_length = 16;
  config->search.batch_size = batch_size;
  config->search.whisper_timestamps = whisper_timestamps;
  config->search.whisper_max_initial_timestamp_index = 2;
  auto model = std::make_shared<TestWhisperModel>(std::move(config));
  model->p_device_scoring_ = GetDeviceInterface(g_test_device);
  return model;
}

DeviceSpan<float> SetLogits(Generator& generator,
                            const GeneratorParams& params,
                            std::initializer_list<float> values) {
  auto logits = params.p_device->Allocate<float>(values.size());
  std::copy(values.begin(), values.end(), logits.CpuSpan().begin());
  logits.CopyCpuToDevice();
  generator.SetLogits(logits);
  return logits;
}

TEST(WhisperTimestampConfigTests, ParsesModelMetadataAndSearchOptions) {
  Config config;
  OverlayConfig(config, R"({
    "model": {
      "timestamp_begin_token_id": 50364,
      "no_timestamps_token_id": 50363
    },
    "search": {
      "whisper_timestamps": true,
      "whisper_max_initial_timestamp_index": 50
    }
  })");

  EXPECT_EQ(config.model.timestamp_begin_token_id, 50364);
  EXPECT_EQ(config.model.no_timestamps_token_id, 50363);
  EXPECT_TRUE(config.search.whisper_timestamps);
  EXPECT_EQ(config.search.whisper_max_initial_timestamp_index, 50);
}

TEST(WhisperTimestampConfigTests, RejectsFractionalInitialTimestampIndex) {
  Config config;

  EXPECT_THROW(
      OverlayConfig(config, R"({
        "search": {
          "whisper_max_initial_timestamp_index": 1.5
        }
      })"),
      std::runtime_error);
}

TEST(WhisperTimestampConfigTests, UsesCanonicalInitialTimestampDefault) {
  Config config;

  EXPECT_EQ(config.search.whisper_max_initial_timestamp_index, 50);
}

TEST(WhisperTimestampLogitsTests, RequiresTimestampAtInitialBoundary) {
  auto processor = CreateProcessor(1);
  std::array<float, 8> logits{5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f, -1.0f, -2.0f};

  processor.Apply(logits, std::array<int32_t, 2>{0, 1}, 2);

  EXPECT_TRUE(std::isinf(logits[0]));
  EXPECT_TRUE(std::isinf(logits[4]));
  EXPECT_FLOAT_EQ(logits[5], 0.0f);
  EXPECT_FLOAT_EQ(logits[6], -1.0f);
  EXPECT_TRUE(std::isinf(logits[7]));
}

TEST(WhisperTimestampLogitsTests, EnforcesTimestampPairsAndMonotonicity) {
  auto processor = CreateProcessor();
  std::array<float, 8> logits{5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f, -1.0f, -2.0f};

  processor.Apply(logits, std::array<int32_t, 3>{0, 1, 5}, 2);

  EXPECT_FLOAT_EQ(logits[0], 5.0f);
  EXPECT_FLOAT_EQ(logits[1], 4.0f);
  EXPECT_FLOAT_EQ(logits[2], 3.0f);
  EXPECT_EQ(logits[3], -std::numeric_limits<float>::infinity());
  EXPECT_EQ(logits[5], -std::numeric_limits<float>::infinity());
  EXPECT_TRUE(std::isinf(logits[6]));
  EXPECT_TRUE(std::isinf(logits[7]));
}

TEST(WhisperTimestampLogitsTests, AllowsEotAfterClosingTimestamp) {
  auto processor = CreateProcessor();
  std::array<float, 8> logits{5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f, -1.0f, -2.0f};

  processor.Apply(logits, std::array<int32_t, 4>{0, 5, 0, 6}, 1);

  EXPECT_FLOAT_EQ(logits[2], 3.0f);
  EXPECT_EQ(logits[3], -std::numeric_limits<float>::infinity());
  EXPECT_EQ(logits[5], -std::numeric_limits<float>::infinity());
  EXPECT_FLOAT_EQ(logits[6], -1.0f);
}

TEST(WhisperTimestampLogitsTests, SuppressesControlTokensBetweenEotAndTimestamps) {
  auto processor = CreateProcessor();
  std::array<float, 8> logits{0.0f, -1.0f, -2.0f, 100.0f, 90.0f, -3.0f, -4.0f, -5.0f};

  processor.Apply(logits, std::array<int32_t, 3>{0, 5, 1}, 1);

  EXPECT_EQ(logits[3], -std::numeric_limits<float>::infinity());
  EXPECT_EQ(logits[4], -std::numeric_limits<float>::infinity());
}

TEST(WhisperTimestampLogitsTests, RequiresStrictlyIncreasingTimestampAfterText) {
  auto processor = CreateProcessor();
  std::array<float, 8> logits{10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 0.0f, -1.0f, -2.0f};

  processor.Apply(logits, std::array<int32_t, 3>{0, 5, 1}, 1);

  EXPECT_EQ(logits[5], -std::numeric_limits<float>::infinity());
  EXPECT_FLOAT_EQ(logits[6], -1.0f);
}

TEST(WhisperTimestampLogitsTests, PreservesEotAfterTimestampFollowingText) {
  auto processor = CreateProcessor();
  std::array<float, 8> logits{10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 0.0f, -1.0f, -2.0f};

  processor.Apply(logits, std::array<int32_t, 4>{0, 5, 1, 6}, 1);

  EXPECT_EQ(logits[0], -std::numeric_limits<float>::infinity());
  EXPECT_EQ(logits[1], -std::numeric_limits<float>::infinity());
  EXPECT_FLOAT_EQ(logits[2], 8.0f);
}

TEST(WhisperTimestampLogitsTests, SuppressesNoTimestampsTokenIndependently) {
  auto processor = CreateProcessor();
  std::array<float, 8> logits{8.0f, 9.0f, 7.0f, 6.0f, 10.0f, -10.0f, -11.0f, -12.0f};

  processor.Apply(logits, std::array<int32_t, 2>{0, 1}, 1);

  EXPECT_FLOAT_EQ(logits[1], 9.0f);
  EXPECT_EQ(logits[4], -std::numeric_limits<float>::infinity());
}

TEST(WhisperTimestampLogitsTests, ForcesTimestampsWhenTheirCombinedMassWins) {
  auto processor = CreateProcessor();
  std::array<float, 8> logits{0.5f, -1.0f, -2.0f, -3.0f, 0.0f, 0.0f, 0.0f, -10.0f};

  processor.Apply(logits, std::array<int32_t, 1>{0}, 0);

  EXPECT_TRUE(std::isinf(logits[0]));
  EXPECT_TRUE(std::isinf(logits[1]));
  EXPECT_TRUE(std::isinf(logits[2]));
  EXPECT_TRUE(std::isinf(logits[3]));
  EXPECT_TRUE(std::isinf(logits[4]));
  EXPECT_FLOAT_EQ(logits[5], 0.0f);
}

TEST(WhisperTimestampIntegrationTests, AppliesRulesToInitialSuppliedLogits) {
  auto model = CreateTestModel();
  auto params = CreateGeneratorParams(*model);
  Generator generator{*model, *params};
  auto logits = SetLogits(generator, *params, {10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 1.0f, 0.0f, -1.0f});

  generator.GenerateNextToken();

  ASSERT_EQ(generator.TokenCount(), 1u);
  EXPECT_EQ(generator.GetSequence(0).CopyDeviceToCpu()[0], 5);
}

TEST(WhisperTimestampIntegrationTests, RejectsUndersizedSuppliedLogits) {
  auto model = CreateTestModel();
  auto params = CreateGeneratorParams(*model);
  Generator generator{*model, *params};
  auto logits = SetLogits(generator, *params, {1.0f, 2.0f});

  EXPECT_THROW(generator.GenerateNextToken(), std::runtime_error);
}

TEST(WhisperTimestampIntegrationTests, AppliesRulesToModelProducedLogits) {
  auto model = CreateTestModel();
  model->scripted_logits = {
      {10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 1.0f, 0.0f, -1.0f}};
  auto params = CreateGeneratorParams(*model);
  Generator generator{*model, *params};
  const std::array<int32_t, 1> prompt{0};

  generator.AppendTokens(cpu_span<const int32_t>{prompt});
  generator.GenerateNextToken();

  auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
  ASSERT_EQ(sequence.size(), 2u);
  EXPECT_EQ(sequence[1], 5);
}

TEST(WhisperTimestampIntegrationTests, RetainsPromptBoundaryAcrossSteps) {
  auto model = CreateTestModel();
  auto params = CreateGeneratorParams(*model);
  Generator generator{*model, *params};
  const std::array<int32_t, 2> prompt{0, 1};
  generator.AppendTokens(cpu_span<const int32_t>{prompt});

  auto first_logits =
      SetLogits(generator, *params, {10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 1.0f, 0.0f, -1.0f});
  generator.GenerateNextToken();
  auto second_logits =
      SetLogits(generator, *params, {0.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f});
  generator.GenerateNextToken();

  auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
  ASSERT_EQ(sequence.size(), 4u);
  EXPECT_EQ(sequence[2], 5);
  EXPECT_EQ(sequence[3], 1);
}

TEST(WhisperTimestampIntegrationTests, KeepsBatchRowsIndependent) {
  auto model = CreateTestModel(true, 2);
  auto params = CreateGeneratorParams(*model);
  Generator generator{*model, *params};
  const std::array<int32_t, 2> prompt{0, 0};
  generator.AppendTokens(cpu_span<const int32_t>{prompt});

  auto first_logits = SetLogits(
      generator, *params,
      {10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 3.0f, 2.0f, 1.0f,
       10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 1.0f, 3.0f, 2.0f});
  generator.GenerateNextToken();
  auto second_logits = SetLogits(
      generator, *params,
      {0.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f,
       10.0f, 0.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f});
  generator.GenerateNextToken();
  auto third_logits = SetLogits(
      generator, *params,
      {0.0f, -1.0f, -2.0f, -3.0f, -4.0f, 10.0f, 9.0f, 8.0f,
       0.0f, -1.0f, -2.0f, -3.0f, -4.0f, 10.0f, 9.0f, 8.0f});
  generator.GenerateNextToken();

  auto first_sequence = generator.GetSequence(0).CopyDeviceToCpu();
  auto second_sequence = generator.GetSequence(1).CopyDeviceToCpu();
  ASSERT_EQ(first_sequence.size(), 4u);
  ASSERT_EQ(second_sequence.size(), 4u);
  EXPECT_EQ(first_sequence[1], 5);
  EXPECT_EQ(first_sequence[2], 1);
  EXPECT_EQ(first_sequence[3], 6);
  EXPECT_EQ(second_sequence[1], 6);
  EXPECT_EQ(second_sequence[2], 0);
  EXPECT_EQ(second_sequence[3], 7);
}

TEST(WhisperTimestampIntegrationTests, AppliesRulesThroughGeneratorBeamSearch) {
  auto model = CreateTestModel();
  model->config_->search.num_beams = 2;
  auto params = CreateGeneratorParams(*model);
  Generator generator{*model, *params};
  const std::array<int32_t, 1> prompt{0};
  generator.AppendTokens(cpu_span<const int32_t>{prompt});
  auto logits = SetLogits(
      generator, *params,
      {10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 3.0f, 2.0f, 1.0f,
       10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 1.0f, 3.0f, 2.0f});

  EXPECT_NO_THROW(generator.GenerateNextToken());
  EXPECT_FALSE(generator.IsDone());
}

TEST(WhisperTimestampIntegrationTests, AppliesRulesBeforeSampling) {
  struct SamplingOptions {
    int top_k;
    float top_p;
  };
  constexpr std::array<SamplingOptions, 3> sampling_options{{
      {2, 1.0f},
      {0, 0.5f},
      {2, 0.5f},
  }};

  for (const auto& options : sampling_options) {
    auto model = CreateTestModel();
    model->config_->search.do_sample = true;
    model->config_->search.top_k = options.top_k;
    model->config_->search.top_p = options.top_p;
    auto params = CreateGeneratorParams(*model);
    Generator generator{*model, *params};
    const std::array<int32_t, 1> prompt{0};
    generator.AppendTokens(cpu_span<const int32_t>{prompt});
    auto logits = SetLogits(
        generator, *params,
        {10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 1.0f,
         -std::numeric_limits<float>::infinity(),
         -std::numeric_limits<float>::infinity()});

    generator.GenerateNextToken();

    auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
    ASSERT_EQ(sequence.size(), 2u);
    EXPECT_EQ(sequence[1], 5);
  }
}

TEST(WhisperTimestampIntegrationTests, UsesReorderedBeamHistories) {
  Config config;
  config.model.vocab_size = 8;
  config.model.context_length = 16;
  config.model.pad_token_id = 3;
  config.model.eos_token_id = {2};
  config.search.max_length = 16;
  config.search.num_beams = 2;
  auto params = std::make_shared<GeneratorParams>(config);
  params->p_device = GetDeviceInterface(g_test_device);
  auto search = params->p_device->CreateBeam(*params);
  WhisperTimestampLogitsProcessor processor{
      {.timestamp_begin = 5, .eot_token = 2, .no_timestamps_token = 4}};

  auto prompt = params->p_device->Allocate<int32_t>(1);
  prompt.CpuSpan()[0] = 0;
  prompt.CopyCpuToDevice();
  search->AppendTokens(prompt);

  auto first_logits = params->p_device->Allocate<float>(16);
  std::fill(first_logits.CpuSpan().begin(), first_logits.CpuSpan().end(),
            -std::numeric_limits<float>::infinity());
  first_logits.CpuSpan()[5] = 10.0f;
  first_logits.CpuSpan()[6] = 9.0f;
  first_logits.CpuSpan()[13] = 10.0f;
  first_logits.CpuSpan()[14] = 9.0f;
  first_logits.CopyCpuToDevice();
  search->SetLogits(first_logits);
  ApplyWhisperTimestampRulesToSearch(*search, processor, 1);
  search->SelectTop();

  auto second_logits = params->p_device->Allocate<float>(16);
  std::fill(second_logits.CpuSpan().begin(), second_logits.CpuSpan().end(),
            -std::numeric_limits<float>::infinity());
  second_logits.CpuSpan()[1] = 9.0f;
  second_logits.CpuSpan()[2] = 10.0f;
  second_logits.CpuSpan()[8] = 10.0f;
  second_logits.CopyCpuToDevice();
  search->SetLogits(second_logits);
  ApplyWhisperTimestampRulesToSearch(*search, processor, 1);
  search->SelectTop();

  const auto first_history = search->sequences_.GetSequence(0).CopyDeviceToCpu();
  const auto second_history = search->sequences_.GetSequence(1).CopyDeviceToCpu();
  ASSERT_EQ(first_history.size(), 3u);
  ASSERT_EQ(second_history.size(), 3u);
  EXPECT_EQ(first_history[1], 6);
  EXPECT_EQ(first_history[2], 0);
  EXPECT_EQ(second_history[1], 5);
  EXPECT_EQ(second_history[2], 1);

  auto third_logits = params->p_device->Allocate<float>(16);
  std::fill(third_logits.CpuSpan().begin(), third_logits.CpuSpan().end(),
            -std::numeric_limits<float>::infinity());
  third_logits.CpuSpan()[6] = 10.0f;
  third_logits.CpuSpan()[7] = 9.0f;
  third_logits.CpuSpan()[13] = 8.0f;
  third_logits.CpuSpan()[14] = 10.0f;
  third_logits.CpuSpan()[15] = 9.0f;
  third_logits.CopyCpuToDevice();
  search->SetLogits(third_logits);
  ApplyWhisperTimestampRulesToSearch(*search, processor, 1);

  auto processed = search->GetLogits().CopyDeviceToCpu();
  EXPECT_EQ(processed[6], -std::numeric_limits<float>::infinity());
  EXPECT_FLOAT_EQ(processed[7], 9.0f);
  EXPECT_EQ(processed[13], -std::numeric_limits<float>::infinity());
  EXPECT_FLOAT_EQ(processed[14], 10.0f);
}

TEST(WhisperTimestampIntegrationTests, SkipsCompletedBeamBatches) {
  Config config;
  config.model.vocab_size = 8;
  config.model.context_length = 16;
  config.model.pad_token_id = 3;
  config.model.eos_token_id = {2};
  config.search.max_length = 16;
  config.search.batch_size = 2;
  config.search.num_beams = 2;
  config.search.early_stopping = true;
  auto params = std::make_shared<GeneratorParams>(config);
  params->p_device = GetDeviceInterface(g_test_device);
  auto search = params->p_device->CreateBeam(*params);
  WhisperTimestampLogitsProcessor processor{
      {.timestamp_begin = 5, .eot_token = 2, .no_timestamps_token = 4}};

  auto prompt = params->p_device->Allocate<int32_t>(2);
  prompt.CpuSpan()[0] = 0;
  prompt.CpuSpan()[1] = 0;
  prompt.CopyCpuToDevice();
  search->AppendTokens(prompt);

  auto first_logits = params->p_device->Allocate<float>(32);
  std::fill(first_logits.CpuSpan().begin(), first_logits.CpuSpan().end(),
            -std::numeric_limits<float>::infinity());
  for (size_t row = 0; row < 4; ++row) {
    first_logits.CpuSpan()[row * 8 + 5] = 10.0f;
    first_logits.CpuSpan()[row * 8 + 6] = 9.0f;
  }
  first_logits.CopyCpuToDevice();
  search->SetLogits(first_logits);
  ApplyWhisperTimestampRulesToSearch(*search, processor, 1);
  search->SelectTop();

  auto second_logits = params->p_device->Allocate<float>(32);
  std::fill(second_logits.CpuSpan().begin(), second_logits.CpuSpan().end(),
            -std::numeric_limits<float>::infinity());
  second_logits.CpuSpan()[2] = 10.0f;
  second_logits.CpuSpan()[1] = -10.0f;
  second_logits.CpuSpan()[10] = 10.0f;
  second_logits.CpuSpan()[9] = -10.0f;
  second_logits.CpuSpan()[16] = 10.0f;
  second_logits.CpuSpan()[17] = 9.0f;
  second_logits.CpuSpan()[24] = 10.0f;
  second_logits.CpuSpan()[25] = 9.0f;
  second_logits.CopyCpuToDevice();
  search->SetLogits(second_logits);
  ApplyWhisperTimestampRulesToSearch(*search, processor, 1);
  search->SelectTop();

  if (g_test_device == DeviceType::CPU) {
    ASSERT_TRUE(search->IsSequenceDone(0));
    ASSERT_TRUE(search->IsSequenceDone(1));
    ASSERT_FALSE(search->IsSequenceDone(2));
    ASSERT_FALSE(search->IsSequenceDone(3));
  }

  auto third_logits = params->p_device->Allocate<float>(32);
  std::fill(third_logits.CpuSpan().begin(), third_logits.CpuSpan().end(),
            -std::numeric_limits<float>::infinity());
  third_logits.CpuSpan()[20] = 20.0f;
  third_logits.CpuSpan()[16] = 10.0f;
  third_logits.CpuSpan()[17] = 9.0f;
  third_logits.CpuSpan()[28] = 20.0f;
  third_logits.CpuSpan()[24] = 10.0f;
  third_logits.CpuSpan()[25] = 9.0f;
  third_logits.CopyCpuToDevice();
  search->SetLogits(third_logits);

  EXPECT_NO_THROW(ApplyWhisperTimestampRulesToSearch(*search, processor, 1));
  const auto processed = search->GetLogits().CopyDeviceToCpu();
  for (size_t index = 0; index < 16; ++index)
    EXPECT_EQ(processed[index], -std::numeric_limits<float>::infinity());
  EXPECT_EQ(processed[20], -std::numeric_limits<float>::infinity());
  EXPECT_EQ(processed[28], -std::numeric_limits<float>::infinity());
}

TEST(WhisperTimestampIntegrationTests, SkipsCompletedBatchRows) {
  auto model = CreateTestModel(true, 2);
  auto params = CreateGeneratorParams(*model);
  Generator generator{*model, *params};
  const std::array<int32_t, 2> prompt{0, 0};
  generator.AppendTokens(cpu_span<const int32_t>{prompt});

  auto first_logits = SetLogits(
      generator, *params,
      {0.0f, -1.0f, -2.0f, -3.0f, -4.0f, 10.0f, 9.0f, 8.0f,
       0.0f, -1.0f, -2.0f, -3.0f, -4.0f, 10.0f, 9.0f, 8.0f});
  generator.GenerateNextToken();
  auto second_logits = SetLogits(
      generator, *params,
      {0.0f, 1.0f, 10.0f, -1.0f, -2.0f, -10.0f, -11.0f, -12.0f,
       0.0f, 10.0f, 9.0f, -1.0f, -2.0f, -10.0f, -11.0f, -12.0f});
  generator.GenerateNextToken();
  auto third_logits = SetLogits(
      generator, *params,
      {-std::numeric_limits<float>::infinity(),
       -std::numeric_limits<float>::infinity(),
       -std::numeric_limits<float>::infinity(),
       -std::numeric_limits<float>::infinity(),
       -std::numeric_limits<float>::infinity(),
       -std::numeric_limits<float>::infinity(),
       -std::numeric_limits<float>::infinity(),
       -std::numeric_limits<float>::infinity(),
       0.0f, -1.0f, -2.0f, -3.0f, -4.0f, 1.0f, 10.0f, 9.0f});

  EXPECT_NO_THROW(generator.GenerateNextToken());
  auto second_sequence = generator.GetSequence(1).CopyDeviceToCpu();
  ASSERT_EQ(second_sequence.size(), 4u);
  EXPECT_EQ(second_sequence[3], 6);
}

TEST(WhisperTimestampIntegrationTests, LeavesDisabledGenerationUnchanged) {
  auto model = CreateTestModel(false);
  auto params = CreateGeneratorParams(*model);
  Generator generator{*model, *params};
  auto logits = SetLogits(generator, *params, {10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 1.0f, 0.0f, -1.0f});

  generator.GenerateNextToken();

  ASSERT_EQ(generator.TokenCount(), 1u);
  EXPECT_EQ(generator.GetSequence(0).CopyDeviceToCpu()[0], 0);
}

TEST(WhisperTimestampIntegrationTests, RejectsMissingMetadata) {
  auto model = CreateTestModel();
  model->config_->model.no_timestamps_token_id.reset();
  auto params = CreateGeneratorParams(*model);

  EXPECT_THROW(Generator(*model, *params), std::runtime_error);
}

TEST(WhisperTimestampIntegrationTests, RejectsInvalidMetadataOrdering) {
  auto model = CreateTestModel();
  model->config_->model.no_timestamps_token_id =
      model->config_->model.timestamp_begin_token_id;
  auto params = CreateGeneratorParams(*model);

  EXPECT_THROW(Generator(*model, *params), std::runtime_error);
}

TEST(WhisperTimestampIntegrationTests, RejectsInvalidInitialTimestampIndex) {
  auto model = CreateTestModel();
  model->config_->search.whisper_max_initial_timestamp_index = 3;
  auto params = CreateGeneratorParams(*model);

  EXPECT_THROW(Generator(*model, *params), std::runtime_error);
}

TEST(WhisperTimestampIntegrationTests, RejectsBypassModelTypes) {
  auto model = CreateTestModel();
  model->config_->model.type = "nemotron_speech";
  auto params = CreateGeneratorParams(*model);

  EXPECT_THROW(Generator(*model, *params), std::runtime_error);
}

TEST(WhisperTimestampIntegrationTests, RejectsIncompatibleGenerationModes) {
  {
    auto model = CreateTestModel();
    auto params = CreateGeneratorParams(*model);
    params->speculative.ngram_size = 1;
    EXPECT_THROW(Generator(*model, *params), std::runtime_error);
  }
  {
    auto model = CreateTestModel();
    auto params = CreateGeneratorParams(*model);
    params->guidance_type = "regex";
    params->guidance_data = ".*";
    EXPECT_THROW(Generator(*model, *params), std::runtime_error);
  }
  {
    auto model = CreateTestModel();
    model->config_->model.eos_token_id.push_back(3);
    auto params = CreateGeneratorParams(*model);
    EXPECT_THROW(Generator(*model, *params), std::runtime_error);
  }
}

TEST(WhisperTimestampIntegrationTests, RejectsPromptWithNoTimestampsToken) {
  auto model = CreateTestModel();
  auto params = CreateGeneratorParams(*model);
  Generator generator{*model, *params};
  const std::array<int32_t, 1> prompt{4};
  generator.AppendTokens(cpu_span<const int32_t>{prompt});
  auto logits = SetLogits(generator, *params, {10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 1.0f, 0.0f, -1.0f});

  EXPECT_THROW(generator.GenerateNextToken(), std::runtime_error);
}

TEST(WhisperTimestampIntegrationTests, RejectsAnotherWindowOnTheSameGenerator) {
  auto model = CreateTestModel();
  auto params = CreateGeneratorParams(*model);
  Generator generator{*model, *params};
  const std::array<int32_t, 1> prompt{0};
  generator.AppendTokens(cpu_span<const int32_t>{prompt});
  auto logits = SetLogits(generator, *params,
                          {10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 1.0f, 0.0f, -1.0f});
  generator.GenerateNextToken();

  EXPECT_THROW(generator.AppendTokens(cpu_span<const int32_t>{prompt}), std::runtime_error);
}

TEST(WhisperTimestampTokenizerTests, HidesTimestampTokensFromDecodedText) {
  Config config;
  config.config_path = fs::path{MODEL_PATH "whisper"};
  config.model.vocab_size = 51865;
  config.model.timestamp_begin_token_id = 50364;
  auto tokenizer = std::make_shared<Tokenizer>(config);
  auto text_tokens = tokenizer->Encode(" hello");
  std::vector<int32_t> timestamped_tokens{50364};
  timestamped_tokens.insert(timestamped_tokens.end(), text_tokens.begin(), text_tokens.end());
  timestamped_tokens.push_back(50414);

  EXPECT_EQ(tokenizer->Decode(timestamped_tokens), tokenizer->Decode(text_tokens));
  EXPECT_TRUE(tokenizer->HasTimestampTokens());
  EXPECT_EQ(tokenizer->GetTimestampBeginTokenId(), 50364);
  EXPECT_FALSE(tokenizer->IsTimestampToken(50363));
  EXPECT_TRUE(tokenizer->IsTimestampToken(50364));
  EXPECT_TRUE(tokenizer->IsTimestampToken(51864));
  EXPECT_FALSE(tokenizer->IsTimestampToken(51865));
  EXPECT_DOUBLE_EQ(tokenizer->TimestampToSeconds(50414), 1.0);
  EXPECT_DOUBLE_EQ(tokenizer->TimestampToSeconds(50399), 0.7);
  EXPECT_THROW(tokenizer->TimestampToSeconds(50363), std::invalid_argument);
  EXPECT_THROW(tokenizer->TimestampToSeconds(51865), std::invalid_argument);

  auto stream = tokenizer->CreateStream();
  EXPECT_TRUE(stream->Decode(50364).empty());

  auto stream_tokens = tokenizer->Encode(" hello world");
  auto timestamped_stream = tokenizer->CreateStream();
  auto plain_stream = tokenizer->CreateStream();
  std::string timestamped_text;
  std::string plain_text;
  for (size_t index = 0; index < stream_tokens.size(); ++index) {
    plain_text += plain_stream->Decode(stream_tokens[index]);
    timestamped_text += timestamped_stream->Decode(stream_tokens[index]);
    if (index == 0)
      timestamped_text += timestamped_stream->Decode(50364);
  }
  EXPECT_EQ(timestamped_text, plain_text);
}

TEST(WhisperTimestampTokenizerTests, RejectsInvalidTimestampMetadata) {
  Config config;
  config.config_path = fs::path{MODEL_PATH "whisper"};
  config.model.vocab_size = 51865;
  config.model.timestamp_begin_token_id = -1;

  EXPECT_THROW((void)Tokenizer{config}, std::runtime_error);
}

TEST(WhisperTimestampTokenizerTests, ReportsUnavailableTimestampMetadata) {
  Config config;
  config.config_path = fs::path{MODEL_PATH "whisper"};
  config.model.vocab_size = 51865;
  auto tokenizer = std::make_shared<Tokenizer>(config);

  EXPECT_FALSE(tokenizer->HasTimestampTokens());
  EXPECT_FALSE(tokenizer->IsTimestampToken(50364));
  EXPECT_THROW(tokenizer->GetTimestampBeginTokenId(), std::runtime_error);
  EXPECT_THROW(tokenizer->TimestampToSeconds(50364), std::invalid_argument);
}

TEST(WhisperTimestampLogitsTests, DoesNotForceTimestampsOnAnExactTie) {
  auto processor = CreateProcessor();
  std::array<float, 8> logits{0.0f, -10.0f, -10.0f, -10.0f, 0.0f, 0.0f,
                              -std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()};

  processor.Apply(logits, std::array<int32_t, 1>{0}, 0);

  EXPECT_FLOAT_EQ(logits[0], 0.0f);
  EXPECT_FLOAT_EQ(logits[5], 0.0f);
}

TEST(WhisperTimestampLogitsTests, ProbabilityMassComparisonIsShiftInvariant) {
  auto processor = CreateProcessor();
  for (float offset : {std::ldexp(1.0f, 60), -std::ldexp(1.0f, 60)}) {
    SCOPED_TRACE(offset);
    std::array<float, 8> logits{offset, -std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity(),
                                offset, offset,
                                -std::numeric_limits<float>::infinity()};

    processor.Apply(logits, std::array<int32_t, 1>{0}, 0);

    EXPECT_EQ(logits[0], -std::numeric_limits<float>::infinity());
    EXPECT_FLOAT_EQ(logits[5], offset);
    EXPECT_FLOAT_EQ(logits[6], offset);
  }
}

TEST(WhisperTimestampLogitsTests, UsesDoublePrecisionForNearTieMass) {
  WhisperTimestampLogitsProcessor processor{
      {.timestamp_begin = 5, .eot_token = 2, .no_timestamps_token = 4}};
  std::vector<float> logits(22, -std::numeric_limits<float>::infinity());
  logits[0] = 0.0f;
  const float timestamp_logit = static_cast<float>(std::log(1.0 / 17.0));
  std::fill(logits.begin() + 5, logits.end(), timestamp_logit);

  processor.Apply(logits, std::array<int32_t, 1>{0}, 0);

  EXPECT_EQ(logits[0], -std::numeric_limits<float>::infinity());
}

TEST(WhisperTimestampLogitsTests, RejectsAnAllMaskedRow) {
  auto processor = CreateProcessor(0);
  std::array<float, 8> logits;
  logits.fill(-std::numeric_limits<float>::infinity());

  EXPECT_THROW(processor.Apply(logits, std::array<int32_t, 1>{0}, 1),
               std::runtime_error);
}

}  // namespace
}  // namespace Generators

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);

  std::filesystem::path ep_dir;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--ep_dir" && i + 1 < argc) {
      ep_dir = argv[++i];
    } else if (arg == "--device" && i + 1 < argc) {
      const std::string device = argv[++i];
      if (device == "cpu")
        Generators::g_test_device = Generators::DeviceType::CPU;
      else if (device == "cuda")
        Generators::g_test_device = Generators::DeviceType::CUDA;
      else
        throw std::runtime_error("Unsupported test device: " + device);
    }
  }

  test_ep::EpRegistrar ep_registrar;
  ep_registrar.DiscoverFromDirectory(ep_dir);
  ep_registrar.RegisterAll();

  std::cout << "Whisper timestamp test device: "
            << Generators::to_string(Generators::g_test_device) << std::endl;
  const int result = RUN_ALL_TESTS();
  OgaShutdown();
  return result;
}
