// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <string>
#include <utility>

#include "models/nemotron_speech.h"
#include "models/parakeet.h"
#include "models/preprocessing/genai_tokenizer.h"
#include "models/preprocessing/token_post_processor.h"
#include "models/preprocessing/nemotron_streaming_processor.h"
#include "models/whisper.h"
#include "ort_genai.h"

namespace {
std::string GetExceptionMessage(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const std::exception& ex) {
    return ex.what();
  }

  return {};
}
}  // namespace

TEST(AudioSpeechValidationTests, WhisperAudioFeaturesRankValidation) {
  EXPECT_NO_THROW(Generators::ValidateWhisperAudioFeaturesShape({1, 80, 3000}, 3000));

  const std::string rank_error = GetExceptionMessage([] {
    Generators::ValidateWhisperAudioFeaturesShape({1, 3000}, 3000);
  });
  EXPECT_NE(rank_error.find("rank 3"), std::string::npos);
}

TEST(AudioSpeechValidationTests, NemotronMelTensorRankValidation) {
  const std::string rank_error = GetExceptionMessage([] {
    Generators::GetValidatedNemotronMelFrameCount({304, 128}, 128);
  });
  EXPECT_NE(rank_error.find("rank 3"), std::string::npos);
}

TEST(AudioSpeechValidationTests, NemotronMelShapeValidationReturnsFrameCountFromMiddleDimension) {
  EXPECT_EQ(Generators::GetValidatedNemotronMelFrameCount({1, 304, 128}, 128), 304);
}

TEST(AudioSpeechValidationTests, NemotronMelDimensionValidation) {
  const std::string mels_error = GetExceptionMessage([] {
    Generators::GetValidatedNemotronMelFrameCount({1, 128, 304}, 128);
  });
  EXPECT_NE(mels_error.find("expected num_mels"), std::string::npos);
  EXPECT_NE(mels_error.find("304"), std::string::npos);
  EXPECT_NE(mels_error.find("128"), std::string::npos);
}

TEST(AudioSpeechValidationTests, NemotronEncoderOutputRankValidation) {
  EXPECT_NO_THROW(Generators::ValidateNemotronEncoderOutputRank({1, 64, 512}));

  const std::string rank_error = GetExceptionMessage([] {
    Generators::ValidateNemotronEncoderOutputRank({1, 64});
  });
  EXPECT_NE(rank_error.find("rank 3"), std::string::npos);
}

TEST(AudioSpeechValidationTests, NemotronTimestampConfiguration) {
  Generators::Config config;
  config.model.type = "nemotron_speech";
  config.model.timestamp_level = Generators::Config::TimestampLevel::All;
  config.model.sample_rate = 16000;
  config.model.hop_length = 160;
  config.model.subsampling_factor = 8;
  config.model.segment_separators = {".", "!"};
  config.model.segment_gap_threshold_seconds = 1.0;

  Generators::NemotronConfig nemotron_config;
  EXPECT_NO_THROW(nemotron_config.PopulateFromConfig(config));
  EXPECT_EQ(nemotron_config.timestamp_level, Generators::Config::TimestampLevel::All);
  EXPECT_EQ(Generators::GetSegmentGapThresholdFrames(config.model), 13);
}

TEST(AudioSpeechValidationTests, NemotronTimestampGapRoundsToNearestFrame) {
  Generators::Config config;
  config.model.type = "nemotron_speech";
  config.model.timestamp_level = Generators::Config::TimestampLevel::Segment;
  config.model.sample_rate = 16000;
  config.model.hop_length = 160;
  config.model.subsampling_factor = 8;
  Generators::NemotronConfig nemotron_config;

  config.model.segment_gap_threshold_seconds = 0.99;
  nemotron_config.PopulateFromConfig(config);
  EXPECT_EQ(Generators::GetSegmentGapThresholdFrames(config.model), 12);

  config.model.segment_gap_threshold_seconds = 1.0;
  nemotron_config.PopulateFromConfig(config);
  EXPECT_EQ(Generators::GetSegmentGapThresholdFrames(config.model), 13);
}

TEST(AudioSpeechValidationTests, NemotronGlobalFrameUsesAbsoluteSampleOrigin) {
  EXPECT_EQ(Generators::GetNemotronGlobalFrame(0, 3, 160, 8), 3);
  EXPECT_EQ(Generators::GetNemotronGlobalFrame(25600, 3, 160, 8), 23);
  EXPECT_EQ(Generators::GetNemotronGlobalFrame(16000, 3, 160, 8), 15);
  EXPECT_THROW(Generators::GetNemotronGlobalFrame(-1, 0, 160, 8), std::runtime_error);
}

TEST(AudioSpeechValidationTests, NemotronChunkOriginKeyMatchesConsumer) {
  auto& allocator = Generators::GetDeviceInterface(Generators::DeviceType::CPU)->GetAllocator();
  auto features = std::make_shared<Generators::Tensor>(
      OrtValue::CreateTensor<float>(allocator, std::array<int64_t, 1>{1}));
  Generators::NamedTensors tensors;
  tensors.emplace(std::string(Generators::Config::Defaults::AudioFeaturesName), features);
  Generators::AddTimestampChunkOrigin(tensors, 8960);
  ASSERT_EQ(tensors.size(), 2U);
  ASSERT_NE(tensors.find("chunk_start_sample"), tensors.end());

  const auto as_extra_inputs = [](const Generators::NamedTensors& named_tensors) {
    std::vector<Generators::ExtraInput> inputs;
    for (const auto& [name, tensor] : named_tensors) inputs.push_back({name, tensor});
    return inputs;
  };
  EXPECT_EQ(Generators::GetNemotronChunkStartSample(as_extra_inputs(tensors)), 8960);

  auto renamed = tensors;
  auto origin = renamed.at("chunk_start_sample");
  renamed.erase("chunk_start_sample");
  renamed.emplace("renamed_chunk_start_sample", origin);
  const auto error = GetExceptionMessage([&] {
    (void)Generators::GetNemotronChunkStartSample(as_extra_inputs(renamed));
  });
  EXPECT_NE(error.find("missing chunk_start_sample"), std::string::npos);
  renamed.erase("renamed_chunk_start_sample");
  EXPECT_THROW(Generators::GetNemotronChunkStartSample(as_extra_inputs(renamed)), std::runtime_error);

  auto wrong_type = std::make_shared<Generators::Tensor>(
      OrtValue::CreateTensor<float>(allocator, std::array<int64_t, 1>{1}));
  tensors.at("chunk_start_sample") = wrong_type;
  EXPECT_THROW(Generators::GetNemotronChunkStartSample(as_extra_inputs(tensors)), std::runtime_error);

  auto wrong_count = std::make_shared<Generators::Tensor>(
      OrtValue::CreateTensor<int64_t>(allocator, std::array<int64_t, 1>{2}));
  tensors.at("chunk_start_sample") = wrong_count;
  EXPECT_THROW(Generators::GetNemotronChunkStartSample(as_extra_inputs(tensors)), std::runtime_error);

  auto negative = OrtValue::CreateTensor<int64_t>(allocator, std::array<int64_t, 1>{1});
  *negative->GetTensorMutableData<int64_t>() = -1;
  tensors.at("chunk_start_sample") = std::make_shared<Generators::Tensor>(std::move(negative));
  EXPECT_THROW(Generators::GetNemotronChunkStartSample(as_extra_inputs(tensors)), std::runtime_error);
}

TEST(AudioSpeechValidationTests, NemotronProcessorEmitsChunkStartSample) {
  const char* model_override = std::getenv("NEMOTRON_STREAMING_MODEL_PATH");
  const auto model_path = model_override ? fs::path{model_override} : fs::path{MODEL_PATH "nemotron-speech-streaming"};
  if (!fs::exists(model_path))
    GTEST_SKIP() << "Streaming ASR model not found at " << model_path.string();

  auto config = std::make_unique<Generators::Config>(model_path, "");
  config->model.timestamp_level = Generators::Config::TimestampLevel::All;
  const auto chunk_samples = static_cast<size_t>(config->model.chunk_samples);
  ASSERT_GT(chunk_samples, 1U);
  // The downloaded CUDA export uses standard ONNX operators; run this metadata test on CPU too.
  config->model.encoder.session_options->provider_options.clear();
  config->model.encoder.session_options->providers.clear();
  config->model.decoder.session_options.provider_options.clear();
  config->model.decoder.session_options.providers.clear();
  config->model.joiner.session_options->provider_options.clear();
  config->model.joiner.session_options->providers.clear();
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), std::move(config));
  Generators::NemotronStreamingProcessor processor{*model};
  processor.SetOption("use_vad", "false");

  std::vector<float> silence(chunk_samples, 0.0f);
  for (int64_t chunk = 0; chunk < 2; ++chunk) {
    auto inputs = processor.Process(silence.data(), silence.size());
    ASSERT_NE(inputs, nullptr);
    ASSERT_EQ(inputs->size(), 2U);
    EXPECT_NE(inputs->find(std::string(Generators::Config::Defaults::AudioFeaturesName)), inputs->end());
    const auto& origin = inputs->at("chunk_start_sample");
    EXPECT_EQ(origin->GetType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    EXPECT_EQ(origin->GetElementCount(), 1U);
    EXPECT_EQ(*origin->GetData<int64_t>(), chunk * static_cast<int64_t>(chunk_samples));
  }

  processor.Process(silence.data(), chunk_samples / 2);
  auto tail = processor.Flush();
  ASSERT_NE(tail, nullptr);
  ASSERT_EQ(tail->size(), 2U);
  const auto& features = tail->at(std::string(Generators::Config::Defaults::AudioFeaturesName));
  const auto& origin = tail->at("chunk_start_sample");
  EXPECT_EQ(*origin->GetData<int64_t>(), 2 * static_cast<int64_t>(chunk_samples));

  auto params = std::make_shared<Generators::GeneratorParams>(*model);
  auto& nemotron_model = dynamic_cast<Generators::NemotronSpeechModel&>(*model);
  Generators::NemotronSpeechState state{nemotron_model, *params};
  const std::string audio_name{Generators::Config::Defaults::AudioFeaturesName};
  const auto missing_origin = GetExceptionMessage([&] {
    state.SetExtraInputs({{audio_name, features}});
  });
  EXPECT_NE(missing_origin.find("chunk_start_sample"), std::string::npos);
  auto invalid_value = OrtValue::CreateTensor<int64_t>(model->allocator_cpu_, std::array<int64_t, 1>{1});
  *invalid_value->GetTensorMutableData<int64_t>() = -1;
  auto invalid_origin = std::make_shared<Generators::Tensor>(std::move(invalid_value));
  EXPECT_THROW(state.SetExtraInputs({{audio_name, features}, {"chunk_start_sample", invalid_origin}}),
               std::runtime_error);
  EXPECT_NO_THROW(state.SetExtraInputs({{audio_name, features}, {"chunk_start_sample", origin}}));

  auto renamed = *tail;
  renamed.erase("chunk_start_sample");
  renamed.emplace("renamed_chunk_start_sample", origin);
  Generators::Generator mismatched_generator{*model, *params};
  const auto mismatch = GetExceptionMessage([&] { mismatched_generator.SetInputs(renamed); });
  EXPECT_NE(mismatch.find("chunk_start_sample"), std::string::npos);

  Generators::Generator generator{*model, *params};
  generator.SetInputs(*tail);
  ASSERT_FALSE(generator.IsDone());
  generator.GenerateNextToken();
  Generators::TokenPostProcessor postprocessor{generator};
  postprocessor.ProcessNextTokens();
  auto results = postprocessor.GetNextTokenResults();
  postprocessor.Finalize();
  const auto& final_results = postprocessor.GetNextTokenResults();
  results.insert(results.end(), final_results.begin(), final_results.end());
  for (const auto& result : results) {
    for (const auto& word : result.timestamps->words) {
      EXPECT_GE(word.frame_interval.first,
                2 * static_cast<int64_t>(chunk_samples) /
                    (model->config_->model.hop_length * model->config_->model.subsampling_factor));
    }
  }

  Generators::NemotronStreamingProcessor vad_processor{*model};
  vad_processor.SetOption("silence_duration_ms", "1");
  vad_processor.SetOption("prefix_padding_ms", "1");
  EXPECT_EQ(vad_processor.Process(silence.data(), silence.size()), nullptr);
  vad_processor.SetOption("use_vad", "false");
  auto after_silence = vad_processor.Process(silence.data(), silence.size());
  ASSERT_NE(after_silence, nullptr);
  EXPECT_EQ(*after_silence->at("chunk_start_sample")->GetData<int64_t>(),
            static_cast<int64_t>(chunk_samples));
}

TEST(AudioSpeechValidationTests, NemotronTimestampsRejectMissingFrameDurationParameters) {
  Generators::Config config;
  config.model.type = "nemotron_speech";
  config.model.timestamp_level = Generators::Config::TimestampLevel::Word;
  config.model.segment_gap_threshold_seconds = 1.0;

  EXPECT_THROW(Generators::ValidateTimestampConfiguration(config.model), std::runtime_error);
}

TEST(AudioSpeechValidationTests, ConfigConstructorRejectsInvalidTimestampOverlay) {
  EXPECT_THROW(
      Generators::Config(
          fs::path{MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32"},
          R"({"model":{"type":"nemotron_speech","timestamp_level":"word"}})"),
      std::runtime_error);
}

TEST(AudioSpeechValidationTests, NemotronTimestampsRejectOversizedSegmentGap) {
  Generators::Config config;
  config.model.type = "nemotron_speech";
  config.model.timestamp_level = Generators::Config::TimestampLevel::Segment;
  config.model.sample_rate = 16000;
  config.model.hop_length = 160;
  config.model.subsampling_factor = 8;
  config.model.segment_gap_threshold_seconds = 1e20;

  EXPECT_THROW(Generators::ValidateTimestampConfiguration(config.model), std::runtime_error);
}

TEST(AudioSpeechValidationTests, ZeroAndSubFrameGapsSplitEachWord) {
  for (double threshold : {0.0, 0.01}) {
    EXPECT_EQ(Generators::GetSegmentGapThresholdFrames(threshold, 100, 10, 1), 0);
  }
}

TEST(TokenPostProcessorInternalTests, OfflineBatchMatchesIncrementalProcessing) {
  Generators::Config model_config{fs::path{MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32"}, ""};
  auto tokenizer = std::make_shared<Generators::Tokenizer>(model_config);
  const std::string text = "Hello world. Next.";
  const auto tokens = tokenizer->Encode(text.c_str());
  ASSERT_GT(tokens.size(), 1U);
  std::vector<std::pair<int64_t, int64_t>> intervals;
  for (size_t index = 0; index < tokens.size(); ++index)
    intervals.push_back({static_cast<int64_t>(index), static_cast<int64_t>(index + 1)});

  Generators::MetadataCoreConfig config;
  config.timestamps.level = Generators::Config::TimestampLevel::All;
  config.timestamps.segment_separators = {"."};
  config.seconds_per_frame = 0.1;
  const auto process = [&](size_t batch_size) {
    Generators::TokenPostProcessorInternal stream{tokenizer, config};
    Generators::PostProcessorResult transcript;
    transcript.timestamps.emplace();
    const auto collect = [&] {
      for (const auto& result : stream.GetNextTokenResults()) {
        transcript.text += result.text;
        EXPECT_TRUE(result.timestamps.has_value());
        if (!result.timestamps) continue;
        const auto& timestamps = *result.timestamps;
        auto& output = *transcript.timestamps;
        output.words.insert(output.words.end(), timestamps.words.begin(), timestamps.words.end());
        output.segments.insert(output.segments.end(), timestamps.segments.begin(), timestamps.segments.end());
      }
    };
    for (size_t offset = 0; offset < tokens.size(); offset += batch_size) {
      const auto count = std::min(batch_size, tokens.size() - offset);
      stream.Process(Generators::TokenOutputs{
          std::span<const int32_t>{tokens}.subspan(offset, count),
          std::span<const std::pair<int64_t, int64_t>>{intervals}.subspan(offset, count)});
      collect();
    }
    stream.Finalize();
    collect();
    return transcript;
  };

  const auto batch = process(tokens.size());
  const auto incremental = process(1);
  EXPECT_EQ(batch.text, text);
  EXPECT_EQ(batch.text, incremental.text);
  const auto expect_records = [](const auto& actual, const auto& expected) {
    ASSERT_FALSE(actual.empty());
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t index = 0; index < actual.size(); ++index) {
      EXPECT_EQ(actual[index].text, expected[index].text);
      EXPECT_EQ(actual[index].frame_interval, expected[index].frame_interval);
      EXPECT_DOUBLE_EQ(actual[index].time_interval.first, expected[index].time_interval.first);
      EXPECT_DOUBLE_EQ(actual[index].time_interval.second, expected[index].time_interval.second);
    }
  };
  expect_records(batch.timestamps->words, incremental.timestamps->words);
  expect_records(batch.timestamps->segments, incremental.timestamps->segments);
}

TEST(TokenPostProcessorInternalTests, OptionalTimingAndCountValidation) {
  Generators::Config model_config{fs::path{MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32"}, ""};
  auto tokenizer = std::make_shared<Generators::Tokenizer>(model_config);
  const std::string text = "Hello world.";
  const auto tokens = tokenizer->Encode(text.c_str());
  ASSERT_GT(tokens.size(), 1U);

  Generators::TokenOutputs untimed{tokens};
  EXPECT_EQ(untimed.GetAcousticFrameIntervals(), nullptr);
  Generators::TokenPostProcessorInternal plain{tokenizer, {}};
  plain.Process(untimed);
  std::string decoded;
  for (const auto& result : plain.GetNextTokenResults()) {
    EXPECT_FALSE(result.timestamps.has_value());
    decoded += result.text;
  }
  EXPECT_EQ(decoded, text);

  Generators::MetadataCoreConfig config;
  config.timestamps.level = Generators::Config::TimestampLevel::Word;
  config.seconds_per_frame = 0.1;
  Generators::TokenPostProcessorInternal timed{tokenizer, config};
  const std::vector<std::pair<int64_t, int64_t>> incomplete(tokens.size() - 1, {4, 5});
  EXPECT_THROW(timed.Process(Generators::TokenOutputs{tokens, std::span<const std::pair<int64_t, int64_t>>{incomplete}}), std::runtime_error);
  EXPECT_THROW(timed.Process(untimed), std::runtime_error);
  const std::vector<std::pair<int64_t, int64_t>> intervals(tokens.size(), {4, 5});
  timed.Process(Generators::TokenOutputs{tokens, std::span<const std::pair<int64_t, int64_t>>{intervals}});
  decoded.clear();
  for (const auto& result : timed.GetNextTokenResults()) decoded += result.text;
  EXPECT_EQ(decoded, text);
}

class MetadataCoreStateTests : public testing::Test {
 protected:
  struct TestTransducerState : Generators::TransducerState {
    using TransducerState::last_token_intervals_;
    using TransducerState::TransducerState;
    void SetTimestampsEnabled(bool enabled) { timestamps_enabled_ = enabled; }
    void SetExtraInputs(const std::vector<Generators::ExtraInput>& inputs) override {
      received_inputs = inputs;
    }
    std::vector<Generators::ExtraInput> received_inputs;
    Generators::DeviceSpan<float> Run(int, Generators::DeviceSpan<int32_t>&, Generators::DeviceSpan<int32_t>) override {
      throw std::runtime_error("Synthetic metadata test does not run inference");
    }
    void StepToken() override {}
    void SetStep(const std::vector<int32_t>& tokens, int64_t token_frame_position) {
      last_tokens_ = tokens;
      last_token_intervals_.clear();
      for (size_t index = 0; index < tokens.size(); ++index)
        last_token_intervals_.push_back({token_frame_position, token_frame_position + 1});
    }
  };

  void SetUp() override {
    Generators::Config config{fs::path{MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32"}, ""};
    config.model.type = "nemotron_speech";
    config.model.timestamp_level = Generators::Config::TimestampLevel::All;
    config.model.sample_rate = 100;
    config.model.hop_length = 10;
    config.model.subsampling_factor = 1;
    config.model.segment_separators = {"."};
    tokenizer = std::make_shared<Generators::Tokenizer>(config);
    tokens = tokenizer->Encode("Hello world.");
    ASSERT_GT(tokens.size(), 1U);
    model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
    model->config_->model.timestamp_level = Generators::Config::TimestampLevel::All;
    params = std::make_shared<Generators::GeneratorParams>(*model);
    generator = std::make_unique<Generators::Generator>(*model, *params);
    auto transducer = std::make_unique<TestTransducerState>(*params, *model);
    source = transducer.get();
    source->SetTimestampsEnabled(true);
    generator->state_ = std::move(transducer);
    source->SetStep(tokens, 4);
    model->config_->model.type = "nemotron_speech";
    model->config_->model.sample_rate = 100;
    model->config_->model.hop_length = 10;
    model->config_->model.subsampling_factor = 1;
    model->config_->model.segment_separators = {"."};
  }

  std::shared_ptr<Generators::Tokenizer> tokenizer;
  std::vector<int32_t> tokens;
  std::shared_ptr<Generators::Model> model;
  std::shared_ptr<Generators::GeneratorParams> params;
  std::unique_ptr<Generators::Generator> generator;
  TestTransducerState* source{};
  const std::pair<int64_t, int64_t> interval{4, 5};
};

TEST_F(MetadataCoreStateTests, PostProcessorBaseOwnsTokenProcessor) {
  const int initial_count = Generators::TokenPostProcessor::Count();
  {
    auto processor = std::make_unique<Generators::TokenPostProcessor>(*generator);
    processor->ProcessNextTokens();
    ASSERT_EQ(processor->GetNextTokenResults().size(), tokens.size());
    auto& token_processor = *processor;
    std::unique_ptr<Generators::PostProcessor> base = std::move(processor);
    EXPECT_NE(base, nullptr);
    EXPECT_EQ(Generators::TokenPostProcessor::Count(), initial_count + 1);
    base->Finalize();
    ASSERT_EQ(token_processor.GetNextTokenResults().size(), 1U);
    base->Reset();
    EXPECT_TRUE(token_processor.GetNextTokenResults().empty());
    token_processor.ProcessNextTokens();
    EXPECT_EQ(token_processor.GetNextTokenResults().size(), tokens.size());
  }
  EXPECT_EQ(Generators::TokenPostProcessor::Count(), initial_count);
}

TEST_F(MetadataCoreStateTests, ChunkStartSampleSurvivesGeneratorInputHandoff) {
  auto features = std::make_shared<Generators::Tensor>(
      OrtValue::CreateTensor<float>(model->allocator_cpu_, std::array<int64_t, 1>{1}));
  auto origin_value = OrtValue::CreateTensor<int64_t>(model->allocator_cpu_, std::array<int64_t, 1>{1});
  *origin_value->GetTensorMutableData<int64_t>() = 25600;
  auto origin = std::make_shared<Generators::Tensor>(std::move(origin_value));
  Generators::NamedTensors inputs;
  inputs.emplace(std::string(Generators::Config::Defaults::AudioFeaturesName), features);
  inputs.emplace("chunk_start_sample", origin);
  model->config_->model.type = "nemotron_speech";

  generator->SetInputs(inputs);
  ASSERT_EQ(source->received_inputs.size(), 2U);
  const auto origin_input = std::find_if(source->received_inputs.begin(), source->received_inputs.end(),
                                         [](const Generators::ExtraInput& input) {
                                           return input.name == "chunk_start_sample";
                                         });
  ASSERT_NE(origin_input, source->received_inputs.end());
  EXPECT_EQ(origin_input->tensor, origin);
  EXPECT_EQ(*origin_input->tensor->GetData<int64_t>(), 25600);
}

TEST_F(MetadataCoreStateTests, PostProcessorOwnsGroupingAcrossMultipleEmissions) {
  Generators::TokenPostProcessor processor{*generator};
  processor.ProcessNextTokens();
  const auto& results = processor.GetNextTokenResults();
  ASSERT_EQ(results.size(), tokens.size());
  std::string text;
  std::string words;
  for (const auto& result : results) {
    text += result.text;
    for (const auto& word : result.timestamps->words) words += word.text;
  }
  EXPECT_EQ(text, "Hello world.");
  EXPECT_EQ(&processor.GetNextTokenResults(), &results);
  processor.Finalize();
  for (const auto& result : processor.GetNextTokenResults()) {
    for (const auto& word : result.timestamps->words) {
      words += word.text;
      EXPECT_EQ(word.frame_interval, interval);
    }
  }
  EXPECT_EQ(words, text);
  processor.Finalize();
  EXPECT_TRUE(processor.GetNextTokenResults()[0].timestamps->words.empty());
  source->SetStep({}, 99);
  processor.ProcessNextTokens();
  EXPECT_TRUE(processor.GetNextTokenResults().empty());
}

TEST_F(MetadataCoreStateTests, InvalidIntervalsDoNotAdvanceDecoding) {
  Generators::TokenPostProcessor processor{*generator};
  source->last_token_intervals_.clear();
  EXPECT_THROW(processor.ProcessNextTokens(), std::runtime_error);
  source->SetStep(tokens, 4);
  source->last_token_intervals_[0] = {5, 5};
  EXPECT_THROW(processor.ProcessNextTokens(), std::runtime_error);
  source->last_token_intervals_[0] = {-1, 5};
  EXPECT_THROW(processor.ProcessNextTokens(), std::runtime_error);
  source->last_token_intervals_.pop_back();
  EXPECT_THROW(processor.ProcessNextTokens(), std::runtime_error);
  source->SetStep(tokens, 4);
  processor.ProcessNextTokens();
  std::string text;
  for (const auto& result : processor.GetNextTokenResults()) text += result.text;
  EXPECT_EQ(text, "Hello world.");
}

TEST_F(MetadataCoreStateTests, GeneratorOutputSelectionUsesModelConfig) {
  source->SetTimestampsEnabled(false);
  Generators::TokenPostProcessor timed_processor{*generator};
  EXPECT_NO_THROW(timed_processor.ProcessNextTokens());
  ASSERT_EQ(timed_processor.GetNextTokenResults().size(), tokens.size());
  for (const auto& result : timed_processor.GetNextTokenResults())
    EXPECT_TRUE(result.timestamps.has_value());

  model->config_->model.timestamp_level = Generators::Config::TimestampLevel::Off;
  source->SetTimestampsEnabled(true);
  source->last_token_intervals_.clear();
  Generators::TokenPostProcessor plain_processor{*generator};
  EXPECT_NO_THROW(plain_processor.ProcessNextTokens());
  std::string text;
  for (const auto& result : plain_processor.GetNextTokenResults()) {
    EXPECT_FALSE(result.timestamps.has_value());
    text += result.text;
  }
  EXPECT_EQ(text, "Hello world.");
}

TEST_F(MetadataCoreStateTests, RepeatedVocabularyIdsKeepPositionSpecificIntervals) {
  Generators::TokenPostProcessor processor{*generator};
  const auto repeated = tokenizer->Encode(" Hello");
  source->SetStep(repeated, 2);
  for (auto& token_interval : source->last_token_intervals_) token_interval.second = 7;
  processor.ProcessNextTokens();
  source->SetStep(repeated, 20);
  for (auto& token_interval : source->last_token_intervals_) token_interval.second = 23;
  processor.ProcessNextTokens();
  std::vector<Generators::TimestampRecord> words;
  for (const auto& result : processor.GetNextTokenResults())
    words.insert(words.end(), result.timestamps->words.begin(), result.timestamps->words.end());
  processor.Finalize();
  for (const auto& result : processor.GetNextTokenResults())
    words.insert(words.end(), result.timestamps->words.begin(), result.timestamps->words.end());
  ASSERT_EQ(words.size(), 2U);
  EXPECT_EQ(words[0].text, " Hello");
  EXPECT_EQ(words[1].text, words[0].text);
  EXPECT_EQ(words[0].frame_interval.first, 2);
  EXPECT_EQ(words[0].frame_interval.second, 7);
  EXPECT_DOUBLE_EQ(words[0].time_interval.first, 0.2);
  EXPECT_DOUBLE_EQ(words[0].time_interval.second, 0.7);
  EXPECT_EQ(words[1].frame_interval.first, 20);
  EXPECT_EQ(words[1].frame_interval.second, 23);
}

TEST_F(MetadataCoreStateTests, OpaqueFieldGettersReturnBorrowedRecords) {
  auto processor = OgaTokenPostProcessor::Create(*reinterpret_cast<const OgaGenerator*>(generator.get()));
  size_t record_count = 0;
  std::string words;
  std::string segments;
  const auto check_records = [&] {
    size_t result_index = 0;
    for (const auto& result : processor->GetNextTokenResults()) {
      const auto* native_result = OgaTokenPostProcessorGetResult(processor.get(), result_index);
      EXPECT_EQ(result.Text.data(), OgaTokenPostProcessorResultGetText(native_result));
      ASSERT_TRUE(result.TimestampResult.has_value());
      const auto& timestamp_result = *result.TimestampResult;
      const auto* timestamps = OgaTokenPostProcessorResultGetTimestampResult(native_result);
      ASSERT_NE(timestamps, nullptr);
      ++result_index;
      EXPECT_EQ(OgaTokenPostProcessorTimestampResultGetTimestampRecord(timestamps, static_cast<OgaTimestampRecordKind>(99), 0), nullptr);
      for (bool use_segments : {false, true}) {
        const auto& records = use_segments ? timestamp_result.Segments : timestamp_result.Words;
        const auto kind = use_segments ? OgaTimestampRecordKind_Segments : OgaTimestampRecordKind_Words;
        const auto count = records.size();
        EXPECT_EQ(count, use_segments ? OgaTokenPostProcessorTimestampResultGetSegmentCount(timestamps)
                                      : OgaTokenPostProcessorTimestampResultGetWordCount(timestamps));
        for (size_t index = 0; index < count; ++index) {
          const auto& record = records[index];
          const auto* native_record = OgaTokenPostProcessorTimestampResultGetTimestampRecord(timestamps, kind, index);
          ASSERT_NE(native_record, nullptr);
          ++record_count;
          const auto frames = record.FrameInterval;
          EXPECT_EQ(frames.first, 4);
          EXPECT_EQ(frames.second, 5);
          const auto times = record.TimeInterval;
          EXPECT_DOUBLE_EQ(times.first, 0.4);
          EXPECT_DOUBLE_EQ(times.second, 0.5);
          (use_segments ? segments : words).append(record.Text);
          EXPECT_EQ(record.Text.data(), OgaTokenPostProcessorTimestampRecordGetText(native_record));
          EXPECT_EQ(frames.first, OgaTokenPostProcessorTimestampRecordGetStartFrame(native_record));
          EXPECT_EQ(frames.second, OgaTokenPostProcessorTimestampRecordGetStopFrame(native_record));
          EXPECT_DOUBLE_EQ(times.first, OgaTokenPostProcessorTimestampRecordGetStartTime(native_record));
          EXPECT_DOUBLE_EQ(times.second, OgaTokenPostProcessorTimestampRecordGetStopTime(native_record));
        }
        EXPECT_EQ(OgaTokenPostProcessorTimestampResultGetTimestampRecord(timestamps, kind, count), nullptr);
      }
    }
  };
  processor->ProcessNextTokens();
  check_records();
  processor->Finalize();
  check_records();
  EXPECT_GT(record_count, 0U);
  EXPECT_EQ(words, "Hello world.");
  EXPECT_EQ(segments, words);
}

TEST_F(MetadataCoreStateTests, SegmentGapsAndFinalizationPreserveExactText) {
  for (double threshold : {0.0, 0.01, 1.0}) {
    model->config_->model.segment_gap_threshold_seconds = threshold;
    Generators::TokenPostProcessor processor{*generator};
    std::vector<Generators::TimestampRecord> segments;
    const auto collect = [&] {
      for (const auto& result : processor.GetNextTokenResults())
        segments.insert(segments.end(), result.timestamps->segments.begin(), result.timestamps->segments.end());
    };
    source->SetStep(tokenizer->Encode(" first"), 2);
    processor.ProcessNextTokens();
    collect();
    source->SetStep(tokenizer->Encode(" second"), threshold < 1.0 ? 2 : 20);
    processor.ProcessNextTokens();
    collect();
    processor.Finalize();
    collect();
    ASSERT_EQ(segments.size(), 2U);
    EXPECT_EQ(segments[0].text, " first");
    EXPECT_EQ(segments[1].text, " second");
    processor.Finalize();
    EXPECT_TRUE(processor.GetNextTokenResults()[0].timestamps->segments.empty());
  }
}

TEST_F(MetadataCoreStateTests, TimestampLevelsControlOnlyPublishedRecords) {
  using Level = Generators::Config::TimestampLevel;
  for (auto level : {Level::Off, Level::Word, Level::Segment, Level::All}) {
    model->config_->model.timestamp_level = level;
    source->SetTimestampsEnabled(level != Level::Off);
    Generators::TokenPostProcessor processor{*generator};
    processor.ProcessNextTokens();
    size_t words = 0;
    size_t segments = 0;
    const auto collect = [&] {
      for (const auto& result : processor.GetNextTokenResults()) {
        EXPECT_EQ(result.timestamps.has_value(), level != Level::Off);
        words += result.timestamps ? result.timestamps->words.size() : 0;
        segments += result.timestamps ? result.timestamps->segments.size() : 0;
      }
    };
    collect();
    processor.Finalize();
    collect();
    EXPECT_EQ(words, level == Level::Word || level == Level::All ? 2U : 0U);
    EXPECT_EQ(segments, level == Level::Segment || level == Level::All ? 1U : 0U);
  }
}

TEST_F(MetadataCoreStateTests, ResetClearsPendingStateAndResultsAreReadOnly) {
  Generators::TokenPostProcessor processor{*generator};
  processor.ProcessNextTokens();
  const auto owned = processor.GetNextTokenResults();
  source->SetStep({}, 99);
  EXPECT_EQ(processor.GetNextTokenResults().size(), tokens.size());
  EXPECT_EQ(processor.GetNextTokenResults()[0].text, owned[0].text);
  processor.Reset();
  EXPECT_TRUE(processor.GetNextTokenResults().empty());
  processor.Finalize();
  EXPECT_TRUE(processor.GetNextTokenResults()[0].timestamps->words.empty());
  EXPECT_TRUE(processor.GetNextTokenResults()[0].timestamps->segments.empty());
  source->SetStep(tokens, 8);
  processor.ProcessNextTokens();
  std::string text;
  for (const auto& result : processor.GetNextTokenResults()) text += result.text;
  EXPECT_EQ(text, "Hello world.");
  processor.Finalize();
  ASSERT_FALSE(processor.GetNextTokenResults()[0].timestamps->words.empty());
  EXPECT_EQ(processor.GetNextTokenResults()[0].timestamps->words.back().frame_interval.first, 8);
  EXPECT_EQ(owned.size(), tokens.size());
}

TEST_F(MetadataCoreStateTests, RejectsMultipleSequences) {
  params->search.batch_size = 2;
  EXPECT_THROW((Generators::TokenPostProcessor{*generator}), std::runtime_error);
  params->search.batch_size = 1;
  params->search.num_beams = 2;
  EXPECT_THROW((Generators::TokenPostProcessor{*generator}), std::runtime_error);
}

TEST(AudioSpeechValidationTests, ParakeetEncoderChannelDimensionValidation) {
  EXPECT_NO_THROW(Generators::ValidateParakeetEncoderOutputShape({1, 512, 64}, 512));

  const std::string dim_error = GetExceptionMessage([] {
    Generators::ValidateParakeetEncoderOutputShape({1, 511, 64}, 512);
  });
  EXPECT_NE(dim_error.find("hidden_dim"), std::string::npos);
}

TEST(AudioSpeechValidationTests, ParakeetDecoderDimensionValidation) {
  EXPECT_NO_THROW(Generators::ValidateParakeetDecoderOutputShape({1, 1024, 1}, 1024));

  const std::string shape_error = GetExceptionMessage([] {
    Generators::ValidateParakeetDecoderOutputShape({1, 1024, 2}, 1024);
  });
  EXPECT_NE(shape_error.find("must have shape"), std::string::npos);
}

TEST(AudioSpeechValidationTests, Rank1TensorsRejected) {
  EXPECT_THROW(Generators::ValidateWhisperAudioFeaturesShape({3000}, 3000), std::runtime_error);
  EXPECT_THROW(Generators::GetValidatedNemotronMelFrameCount({3000}, 80), std::runtime_error);
  EXPECT_THROW(Generators::ValidateParakeetEncoderOutputShape({512}, 512), std::runtime_error);
  EXPECT_THROW(Generators::ValidateParakeetDecoderOutputShape({1024}, 1024), std::runtime_error);
}

TEST(AudioSpeechValidationTests, DimensionMismatchesCaught) {
  EXPECT_THROW(Generators::ValidateWhisperAudioFeaturesShape({1, 80, 2999}, 3000), std::runtime_error);
  EXPECT_THROW(Generators::GetValidatedNemotronMelFrameCount({2, 304, 128}, 128), std::runtime_error);
  EXPECT_THROW(Generators::ValidateParakeetEncoderOutputShape({1, 256, 64}, 512), std::runtime_error);
  EXPECT_THROW(Generators::ValidateParakeetDecoderOutputShape({2, 1024, 1}, 1024), std::runtime_error);
}

TEST(TokenizerWordGroupingTest, OneTokenCanCompleteMultipleWords) {
  Generators::TokenWordGroupingState grouping;
  grouping.Process({OrtxWordBoundaryStyle_SentencePiece, false, false, false},
                   " first second third");

  const auto& completed = grouping.CompletedWords();
  ASSERT_EQ(completed.size(), 2u);
  EXPECT_EQ(completed[0].text, " first");
  EXPECT_EQ(completed[1].text, " second");
  EXPECT_EQ(completed[0].start_token_index, 0u);
  EXPECT_EQ(completed[0].stop_token_index, 1u);
  EXPECT_EQ(completed[1].start_token_index, 0u);
  EXPECT_EQ(completed[1].stop_token_index, 1u);
  EXPECT_EQ(grouping.FirstPendingTokenIndex(), 0u);

  grouping.Finalize();
  ASSERT_EQ(grouping.CompletedWords().size(), 1u);
  EXPECT_EQ(grouping.CompletedWords()[0].text, " third");
  EXPECT_EQ(grouping.FirstPendingTokenIndex(), 1u);
}

TEST(TokenizerWordGroupingTest, EmptyDecoderOutputRetainsContributingTokenSpan) {
  Generators::TokenWordGroupingState grouping;
  grouping.Process({OrtxWordBoundaryStyle_PrefixBpe, false, false, false}, "");
  EXPECT_EQ(grouping.FirstPendingTokenIndex(), 0u);

  grouping.Process({OrtxWordBoundaryStyle_PrefixBpe, false, false, false}, "é");
  grouping.Finalize();
  ASSERT_EQ(grouping.CompletedWords().size(), 1u);
  EXPECT_EQ(grouping.CompletedWords()[0].text, "é");
  EXPECT_EQ(grouping.CompletedWords()[0].start_token_index, 0u);
  EXPECT_EQ(grouping.CompletedWords()[0].stop_token_index, 2u);
}

TEST(TokenizerWordGroupingTest, SkippedSpecialTokenDoesNotExtendBufferedWordSpan) {
  Generators::TokenWordGroupingState grouping;
  grouping.Process({OrtxWordBoundaryStyle_PrefixBpe, true, false, false}, "");
  EXPECT_TRUE(grouping.CompletedWords().empty());
  EXPECT_EQ(grouping.FirstPendingTokenIndex(), 1u);

  grouping.Process({OrtxWordBoundaryStyle_PrefixBpe, false, false, false}, "");
  EXPECT_TRUE(grouping.CompletedWords().empty());
  EXPECT_EQ(grouping.FirstPendingTokenIndex(), 1u);

  grouping.Process({OrtxWordBoundaryStyle_PrefixBpe, false, false, false}, "\xC3\xA9");
  EXPECT_EQ(grouping.FirstPendingTokenIndex(), 1u);
  grouping.Finalize();

  ASSERT_EQ(grouping.CompletedWords().size(), 1u);
  EXPECT_EQ(grouping.CompletedWords()[0].text, "\xC3\xA9");
  EXPECT_EQ(grouping.CompletedWords()[0].start_token_index, 1u);
  EXPECT_EQ(grouping.CompletedWords()[0].stop_token_index, 3u);
  EXPECT_EQ(grouping.FirstPendingTokenIndex(), 3u);
}

TEST(TokenizerWordGroupingTest, FinalizeDiscardsTrailingEmptyDecoderOutput) {
  Generators::TokenWordGroupingState grouping;
  grouping.Process({OrtxWordBoundaryStyle_PrefixBpe, false, false, false}, "");
  EXPECT_EQ(grouping.FirstPendingTokenIndex(), 0u);

  grouping.Finalize();
  EXPECT_TRUE(grouping.CompletedWords().empty());
  EXPECT_EQ(grouping.FirstPendingTokenIndex(), 1u);
}

TEST(TokenizerWordGroupingTest, DelimiterBeforePunctuationStaysWithPreviousWord) {
  Generators::TokenWordGroupingState grouping;
  grouping.Process({OrtxWordBoundaryStyle_SentencePiece, false, false, false}, " hello");
  grouping.Process({OrtxWordBoundaryStyle_SentencePiece, false, false, false}, " ");
  grouping.Process({OrtxWordBoundaryStyle_SentencePiece, false, false, false}, ",");
  EXPECT_TRUE(grouping.CompletedWords().empty());

  grouping.Process({OrtxWordBoundaryStyle_SentencePiece, false, false, false}, " world");
  ASSERT_EQ(grouping.CompletedWords().size(), 1u);
  EXPECT_EQ(grouping.CompletedWords()[0].text, " hello ,");
  EXPECT_EQ(grouping.CompletedWords()[0].start_token_index, 0u);
  EXPECT_EQ(grouping.CompletedWords()[0].stop_token_index, 3u);
}

TEST(TokenizerWordGroupingTest, NonAsciiContinuationDoesNotStartWord) {
  Generators::TokenWordGroupingState grouping;
  grouping.Process({OrtxWordBoundaryStyle_PrefixBpe, false, false, false}, "caf");
  grouping.Process({OrtxWordBoundaryStyle_PrefixBpe, false, false, false}, "\xC3\xA9");
  EXPECT_TRUE(grouping.CompletedWords().empty());
  grouping.Finalize();
  ASSERT_EQ(grouping.CompletedWords().size(), 1u);
  EXPECT_EQ(grouping.CompletedWords()[0].text, "caf\xC3\xA9");
  EXPECT_EQ(grouping.CompletedWords()[0].start_token_index, 0u);
  EXPECT_EQ(grouping.CompletedWords()[0].stop_token_index, 2u);

  Generators::TokenWordGroupingState buffered;
  buffered.Process({OrtxWordBoundaryStyle_PrefixBpe, false, false, false}, "caf");
  buffered.Process({OrtxWordBoundaryStyle_PrefixBpe, false, false, false}, "");
  buffered.Process({OrtxWordBoundaryStyle_PrefixBpe, false, false, false}, "\xC3\xA9");
  EXPECT_TRUE(buffered.CompletedWords().empty());
  buffered.Finalize();
  ASSERT_EQ(buffered.CompletedWords().size(), 1u);
  EXPECT_EQ(buffered.CompletedWords()[0].text, "caf\xC3\xA9");
  EXPECT_EQ(buffered.CompletedWords()[0].start_token_index, 0u);
  EXPECT_EQ(buffered.CompletedWords()[0].stop_token_index, 3u);
}

TEST(TokenizerWordGroupingTest, UnicodePunctuationStaysWithPreviousWord) {
  for (const std::string punctuation : {"\xE3\x80\x82", "\xE2\x80\xA6"}) {
    Generators::TokenWordGroupingState grouping;
    grouping.Process({OrtxWordBoundaryStyle_PrefixBpe, false, false, false}, "hello");
    grouping.Process({OrtxWordBoundaryStyle_PrefixBpe, false, false, false}, " ");
    grouping.Process({OrtxWordBoundaryStyle_PrefixBpe, false, false, false}, punctuation);
    EXPECT_TRUE(grouping.CompletedWords().empty());
    grouping.Process({OrtxWordBoundaryStyle_PrefixBpe, false, false, false}, " world");
    ASSERT_EQ(grouping.CompletedWords().size(), 1u);
    EXPECT_EQ(grouping.CompletedWords()[0].text, "hello " + punctuation);
    EXPECT_EQ(grouping.CompletedWords()[0].start_token_index, 0u);
    EXPECT_EQ(grouping.CompletedWords()[0].stop_token_index, 3u);
  }
}

TEST(TokenizerWordGroupingTest, UnicodeWhitespacePreservesTextAndSpans) {
  for (const std::string separator : {"\xC2\xA0", "\xE3\x80\x80", "\v", "\f"}) {
    const std::string text = "first" + separator + "second" + separator + "third";
    Generators::TokenWordGroupingState grouping;
    grouping.Process({OrtxWordBoundaryStyle_PrefixBpe, false, false, false}, text);
    ASSERT_EQ(grouping.CompletedWords().size(), 2u);
    EXPECT_EQ(grouping.CompletedWords()[0].text, "first");
    EXPECT_EQ(grouping.CompletedWords()[1].text, separator + "second");
    for (const auto& word : grouping.CompletedWords()) {
      EXPECT_EQ(word.start_token_index, 0u);
      EXPECT_EQ(word.stop_token_index, 1u);
    }
    grouping.Finalize();
    ASSERT_EQ(grouping.CompletedWords().size(), 1u);
    EXPECT_EQ(grouping.CompletedWords()[0].text, separator + "third");
    EXPECT_EQ(grouping.FirstPendingTokenIndex(), 1u);

    Generators::TokenWordGroupingState fragmented;
    fragmented.Process({OrtxWordBoundaryStyle_PrefixBpe, false, false, false}, "first");
    fragmented.Process({OrtxWordBoundaryStyle_PrefixBpe, false, false, false}, separator);
    EXPECT_TRUE(fragmented.CompletedWords().empty());
    fragmented.Process({OrtxWordBoundaryStyle_PrefixBpe, false, false, false}, "second");
    ASSERT_EQ(fragmented.CompletedWords().size(), 1u);
    EXPECT_EQ(fragmented.CompletedWords()[0].text, "first");
    EXPECT_EQ(fragmented.CompletedWords()[0].stop_token_index, 1u);
    EXPECT_EQ(fragmented.FirstPendingTokenIndex(), 1u);
    fragmented.Finalize();
    ASSERT_EQ(fragmented.CompletedWords().size(), 1u);
    EXPECT_EQ(fragmented.CompletedWords()[0].text, separator + "second");
    EXPECT_EQ(fragmented.CompletedWords()[0].start_token_index, 1u);
    EXPECT_EQ(fragmented.CompletedWords()[0].stop_token_index, 3u);
  }
}

TEST(TokenizerWordGroupingTest, ExplicitPrefixAndSuffixBoundaries) {
  for (const auto style : {OrtxWordBoundaryStyle_PrefixBpe,
                           OrtxWordBoundaryStyle_SentencePiece}) {
    Generators::TokenWordGroupingState grouping;
    grouping.Process({style, false, false, false}, "first");
    grouping.Process({style, false, true, false}, "second");
    ASSERT_EQ(grouping.CompletedWords().size(), 1u);
    EXPECT_EQ(grouping.CompletedWords()[0].text, "first");
    EXPECT_EQ(grouping.CompletedWords()[0].stop_token_index, 1u);
    grouping.Finalize();
    ASSERT_EQ(grouping.CompletedWords().size(), 1u);
    EXPECT_EQ(grouping.CompletedWords()[0].text, "second");
    EXPECT_EQ(grouping.CompletedWords()[0].start_token_index, 1u);
  }
  Generators::TokenWordGroupingState suffix;
  suffix.Process({OrtxWordBoundaryStyle_SuffixBpe, false, false, false}, "caf");
  suffix.Process({OrtxWordBoundaryStyle_SuffixBpe, false, false, true}, "\xC3\xA9 ");
  ASSERT_EQ(suffix.CompletedWords().size(), 1u);
  EXPECT_EQ(suffix.CompletedWords()[0].text, "caf\xC3\xA9 ");
  EXPECT_EQ(suffix.CompletedWords()[0].start_token_index, 0u);
  EXPECT_EQ(suffix.CompletedWords()[0].stop_token_index, 2u);
}
