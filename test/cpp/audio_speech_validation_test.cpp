// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <functional>
#include <string>

#include "models/nemotron_speech.h"
#include "models/parakeet.h"
#include "models/preprocessing/timestamp_decode.h"
#include "models/preprocessing/genai_tokenizer.h"
#include "models/whisper.h"

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
  EXPECT_EQ(nemotron_config.segment_separators, (std::vector<std::string>{".", "!"}));
  EXPECT_EQ(nemotron_config.segment_gap_threshold_frames, 13);
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
  EXPECT_EQ(nemotron_config.segment_gap_threshold_frames, 12);

  config.model.segment_gap_threshold_seconds = 1.0;
  nemotron_config.PopulateFromConfig(config);
  EXPECT_EQ(nemotron_config.segment_gap_threshold_frames, 13);
}

TEST(AudioSpeechValidationTests, NemotronGlobalFrameUsesAbsoluteSampleOrigin) {
  EXPECT_EQ(Generators::GetNemotronGlobalFrame(0, 3, 160, 8), 3);
  EXPECT_EQ(Generators::GetNemotronGlobalFrame(25600, 3, 160, 8), 23);
  EXPECT_EQ(Generators::GetNemotronGlobalFrame(16000, 3, 160, 8), 15);
  EXPECT_THROW(Generators::GetNemotronGlobalFrame(-1, 0, 160, 8), std::runtime_error);
}

TEST(AudioSpeechValidationTests, NemotronTimestampsRejectMissingFrameDurationParameters) {
  Generators::Config config;
  config.model.type = "nemotron_speech";
  config.model.timestamp_level = Generators::Config::TimestampLevel::Word;
  config.model.segment_gap_threshold_seconds = 1.0;

  Generators::NemotronConfig nemotron_config;
  EXPECT_THROW(nemotron_config.PopulateFromConfig(config), std::runtime_error);
}

TEST(AudioSpeechValidationTests, TimestampSegmentGapSecondsRoundToNearestFrame) {
  const Generators::TimestampTokenizerConfig config{
      Generators::Config::TimestampLevel::Segment, {}, 0.26, 100, 10, 1};
  Generators::TimestampDecodeState state{config};
  const OrtxTimestampWordMetadata first{"first", 0, 1};
  state.Consume({10, 0, 1}, {&first, 1, 1});
  EXPECT_TRUE(state.result_.segments.empty());
  const OrtxTimestampWordMetadata second{" second", 1, 2};
  state.Consume({11, 3, 4}, {&second, 1, 2});
  EXPECT_TRUE(state.result_.segments.empty());
  const OrtxTimestampWordMetadata third{" third", 2, 3};
  state.Consume({12, 7, 8}, {&third, 1, 3});
  ASSERT_EQ(state.result_.segments.size(), 1U);
  EXPECT_EQ(state.result_.segments[0].text, "first second");
  EXPECT_EQ(state.result_.segments[0].start_frame, 0);
  EXPECT_EQ(state.result_.segments[0].stop_frame, 4);
  state.Finalize({nullptr, 0, 3});
  ASSERT_EQ(state.result_.segments.size(), 2U);
  EXPECT_EQ(state.result_.segments[1].text, " third");
}

TEST(AudioSpeechValidationTests, ZeroAndSubFrameGapsSplitEachWord) {
  for (double threshold : {0.0, 0.01}) {
    for (int64_t second_start_frame : {int64_t{0}, int64_t{1}}) {
      EXPECT_EQ(Generators::GetSegmentGapThresholdFrames(threshold, 100, 10, 1), 0);
      const Generators::TimestampTokenizerConfig config{
          Generators::Config::TimestampLevel::Segment, {}, threshold, 100, 10, 1};
      Generators::TimestampDecodeState state{config};
      const OrtxTimestampWordMetadata first{"first", 0, 1};
      state.Consume({10, 0, 1}, {&first, 1, 1});
      state.ClearResult();
      const OrtxTimestampWordMetadata second{" second", 1, 2};
      state.Consume({11, second_start_frame, second_start_frame + 1}, {&second, 1, 2});
      ASSERT_EQ(state.result_.segments.size(), 1U);
      EXPECT_EQ(state.result_.segments[0].text, "first");
      state.ClearResult();
      state.Finalize({nullptr, 0, 2});
      ASSERT_EQ(state.result_.segments.size(), 1U);
      EXPECT_EQ(state.result_.segments[0].text, " second");
    }
  }
}

TEST(AudioSpeechValidationTests, TimestampAccumulatorAttachesPunctuationAndCompletesSegment) {
  Generators::TimestampTokenizerConfig config{
      Generators::Config::TimestampLevel::All, {"."}, std::nullopt, 100, 10, 1};
  Generators::TimestampDecodeState state{config};

  state.Consume({10, 0, 1}, {nullptr, 0, 0});
  state.ClearResult();
  state.Consume({11, 1, 2}, {nullptr, 0, 0});
  state.ClearResult();
  const OrtxTimestampWordMetadata word{" Hello.", 0, 2};
  state.Consume({12, 3, 4}, {&word, 1, 2});

  ASSERT_EQ(state.result_.words.size(), 1u);
  EXPECT_EQ(state.result_.words[0].text, " Hello.");
  EXPECT_EQ(state.result_.words[0].start_frame, 0);
  EXPECT_EQ(state.result_.words[0].stop_frame, 2);
  EXPECT_DOUBLE_EQ(state.result_.words[0].stop_time, 0.2);
  ASSERT_EQ(state.result_.segments.size(), 1u);
  EXPECT_EQ(state.result_.segments[0].text, " Hello.");
}

TEST(AudioSpeechValidationTests, TimestampAccumulatorCompletesSegmentAtFrameGap) {
  Generators::TimestampTokenizerConfig config{
      Generators::Config::TimestampLevel::Segment, {}, 0.3, 100, 10, 1};
  Generators::TimestampDecodeState state{config};

  state.Consume({1, 0, 1}, {nullptr, 0, 0});
  state.ClearResult();
  const OrtxTimestampWordMetadata first_word{" one", 0, 1};
  state.Consume({2, 1, 2}, {&first_word, 1, 1});
  state.ClearResult();
  const OrtxTimestampWordMetadata second_word{" two", 1, 2};
  state.Consume({3, 8, 9}, {&second_word, 1, 2});
  state.ClearResult();
  const OrtxTimestampWordMetadata trailing_word{" three", 2, 3};
  state.Finalize({&trailing_word, 1, 3});

  EXPECT_TRUE(state.result_.words.empty());
  ASSERT_EQ(state.result_.segments.size(), 2u);
  EXPECT_EQ(state.result_.segments[0].text, " one two");
  EXPECT_EQ(state.result_.segments[1].text, " three");

  state.ClearResult();
  state.Finalize({nullptr, 0, 3});
  EXPECT_TRUE(state.result_.segments.empty());
}

TEST(AudioSpeechValidationTests, TimestampAccumulatorPreservesSharedFrames) {
  Generators::TimestampTokenizerConfig config{
      Generators::Config::TimestampLevel::Word, {}, std::nullopt, 100, 10, 1};
  Generators::TimestampDecodeState state{config};
  state.Consume({1, 4, 5}, {nullptr, 0, 0});
  const OrtxTimestampWordMetadata first_word{" one", 0, 1};
  const OrtxTimestampWordMetadata second_word{" two", 1, 2};
  state.Consume({2, 4, 5}, {&first_word, 1, 1});
  state.Finalize({&second_word, 1, 2});
  ASSERT_EQ(state.result_.words.size(), 2U);
  for (const auto& word : state.result_.words) {
    EXPECT_EQ(word.start_frame, 4);
    EXPECT_EQ(word.stop_frame, 5);
    EXPECT_DOUBLE_EQ(word.start_time, 0.4);
    EXPECT_DOUBLE_EQ(word.stop_time, 0.5);
  }
}

TEST(AudioSpeechValidationTests, TimestampAccumulatorCopiesBorrowedText) {
  Generators::TimestampTokenizerConfig config{
      Generators::Config::TimestampLevel::All, {}, std::nullopt, 100, 10, 1};
  Generators::TimestampDecodeState state{config};
  state.Consume({1, 0, 1}, {nullptr, 0, 0});
  std::string text(" word");
  const OrtxTimestampWordMetadata word{text.c_str(), 0, 1};
  state.Finalize({&word, 1, 1});
  text.assign("changed source");
  ASSERT_EQ(state.result_.words.size(), 1U);
  ASSERT_EQ(state.result_.segments.size(), 1U);
  EXPECT_EQ(state.result_.words[0].text, " word");
  EXPECT_EQ(state.result_.segments[0].text, " word");
}

class MetadataCoreStateTests : public testing::Test {
 protected:
  struct TestTransducerState : Generators::TransducerState {
    using TransducerState::last_token_timings_;
    using TransducerState::TransducerState;
    void SetTimestampsEnabled(bool enabled) { timestamps_enabled_ = enabled; }
    Generators::DeviceSpan<float> Run(int, Generators::DeviceSpan<int32_t>&, Generators::DeviceSpan<int32_t>) override {
      throw std::runtime_error("Synthetic metadata test does not run inference");
    }
    void StepToken() override {}
    void SetStep(const std::vector<int32_t>& tokens, int64_t token_frame_position) {
      last_tokens_ = tokens;
      last_token_timings_.clear();
      for (const auto token : tokens) last_token_timings_.push_back({token, token_frame_position, token_frame_position + 1});
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
    metadata_config = tokenizer->GetMetadataCoreConfig();
    EXPECT_EQ(metadata_config.timestamps.level, Generators::Config::TimestampLevel::All);
    EXPECT_EQ(metadata_config.timestamps.sample_rate, 100);
    EXPECT_EQ(metadata_config.timestamps.hop_length, 10);
    EXPECT_EQ(metadata_config.timestamps.subsampling_factor, 1);
    EXPECT_EQ(metadata_config.timestamps.segment_separators, std::vector<std::string>{"."});
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
  }

  std::shared_ptr<Generators::Tokenizer> tokenizer;
  Generators::MetadataCoreConfig metadata_config;
  std::vector<int32_t> tokens;
  std::shared_ptr<Generators::Model> model;
  std::shared_ptr<Generators::GeneratorParams> params;
  std::unique_ptr<Generators::Generator> generator;
  TestTransducerState* source{};
  const OgaTokenMetadataAcousticFrameInterval interval{4, 5};
};

TEST_F(MetadataCoreStateTests, DecodeProcessesMetadataOncePerStep) {
  auto stream = tokenizer->CreateStream();
  auto state = stream->CreateMetadataCoreState(metadata_config);
  auto plain = tokenizer->CreateStream();
  EXPECT_THROW(state->Metadata(), std::runtime_error);
  EXPECT_THROW(state->ConsumeTimestamps(), std::runtime_error);
  EXPECT_THROW(state->ConsumeFinalTimestamps(), std::runtime_error);

  for (size_t index = 0; index < tokens.size(); ++index) {
    const auto& expected = stream->DecodeWithMetadata({tokens[index], 1, interval});
    EXPECT_EQ(state->Text(), plain->Decode(tokens[index]));
    ASSERT_NE(state->Metadata().timestampMetadata, nullptr);
    EXPECT_EQ(&state->ProcessMetadata(), &expected);
    const auto& result = state->ConsumeTimestamps();
    EXPECT_EQ(result.text, expected.text);
    ASSERT_EQ(result.words.size(), expected.timestampMetadata->word_count);
    for (size_t word_index = 0; word_index < result.words.size(); ++word_index) {
      EXPECT_EQ(result.words[word_index].text, expected.timestampMetadata->words[word_index].text);
      EXPECT_EQ(result.words[word_index].start_frame, expected.timestampMetadata->words[word_index].start_frame);
      EXPECT_EQ(result.words[word_index].stop_frame, expected.timestampMetadata->words[word_index].stop_frame);
    }
    const auto count = result.words.size();
    EXPECT_EQ(&state->ConsumeTimestamps(), &result);
    EXPECT_EQ(result.words.size(), count);
  }

  const auto& expected = stream->FinalizeMetadata();
  const auto& final = state->ConsumeFinalTimestamps();
  ASSERT_EQ(final.words.size(), expected.timestampMetadata->word_count);
  ASSERT_EQ(final.segments.size(), expected.timestampMetadata->segment_count);
  for (size_t index = 0; index < final.words.size(); ++index) {
    EXPECT_EQ(final.words[index].text, expected.timestampMetadata->words[index].text);
    EXPECT_DOUBLE_EQ(final.words[index].start_time, expected.timestampMetadata->words[index].start_time);
    EXPECT_DOUBLE_EQ(final.words[index].stop_time, expected.timestampMetadata->words[index].stop_time);
  }
  EXPECT_EQ(&state->ConsumeFinalTimestamps(), &final);
  stream->FinalizeMetadata();
  EXPECT_TRUE(state->ConsumeFinalTimestamps().words.empty());
  EXPECT_TRUE(state->ConsumeFinalTimestamps().segments.empty());
}

TEST_F(MetadataCoreStateTests, DisabledConsumersDoNotBlockOrEnableProducers) {
  auto stream = tokenizer->CreateStream();
  auto state = stream->CreateMetadataCoreState(Generators::MetadataCoreConfig{});
  auto plain = tokenizer->CreateStream();
  auto enabled = tokenizer->CreateStream();
  auto enabled_state = enabled->CreateMetadataCoreState(metadata_config);
  EXPECT_FALSE(state->TimestampsEnabled());
  for (size_t index = 0; index < tokens.size(); ++index) {
    stream->DecodeWithMetadata({tokens[index], 1, interval});
    EXPECT_EQ(state->Text(), plain->Decode(tokens[index]));
    EXPECT_EQ(state->Metadata().timestampMetadata, nullptr);
    const auto* current_metadata = &state->Metadata();
    const auto current_text = state->Text();
    EXPECT_THROW(state->ConsumeTimestamps(), std::runtime_error);
    EXPECT_EQ(&state->Metadata(), current_metadata);
    EXPECT_EQ(state->Text(), current_text);
    EXPECT_FALSE(state->TimestampsEnabled());
    enabled->DecodeWithMetadata({tokens[index], 1, interval});
    EXPECT_NE(enabled_state->Metadata().timestampMetadata, nullptr);
    enabled_state->ConsumeTimestamps();
  }
  stream->FinalizeMetadata();
  EXPECT_EQ(state->Metadata().timestampMetadata, nullptr);
  EXPECT_THROW(state->ConsumeFinalTimestamps(), std::runtime_error);
  EXPECT_NO_THROW(stream->FinalizeMetadata());
}

TEST_F(MetadataCoreStateTests, StreamBindingResetAndDestructionInvalidateState) {
  auto stream = tokenizer->CreateStream();
  auto state = stream->CreateMetadataCoreState(metadata_config);
  auto other = tokenizer->CreateStream();
  auto other_state = other->CreateMetadataCoreState(metadata_config);
  EXPECT_THROW(stream->CreateMetadataCoreState(metadata_config), std::runtime_error);
  EXPECT_THROW(other_state->Metadata(), std::runtime_error);
  EXPECT_THROW(stream->Decode(tokens[0]), std::runtime_error);
  stream->DecodeWithMetadata({tokens[0], 1, interval});
  stream->Reset();
  EXPECT_THROW(state->Text(), std::runtime_error);
  EXPECT_THROW(state->Metadata(), std::runtime_error);
  EXPECT_THROW(state->ConsumeTimestamps(), std::runtime_error);
  EXPECT_THROW(stream->DecodeWithMetadata({tokens[0], 1, interval}), std::runtime_error);
  auto replacement = stream->CreateMetadataCoreState(Generators::MetadataCoreConfig{});
  EXPECT_NO_THROW(stream->DecodeWithMetadata({tokens[0], 1, interval}));
  stream.reset();
  EXPECT_THROW(replacement->Metadata(), std::runtime_error);
  EXPECT_THROW(replacement->Text(), std::runtime_error);
  EXPECT_NO_THROW(other->DecodeWithMetadata({tokens[0], 1, interval}));
  EXPECT_EQ(other_state->Text(), tokenizer->Decode(std::span<const int32_t>{tokens.data(), 1}));
}

TEST_F(MetadataCoreStateTests, ConfigurationValidationAndSnapshots) {
  auto stream = tokenizer->CreateStream();
  Generators::MetadataCoreConfig config;
  config.timestamps.level = Generators::Config::TimestampLevel::Word;
  auto state = stream->CreateMetadataCoreState(config);
  EXPECT_EQ(state->TimestampsEnabled(), true);
  EXPECT_EQ(tokenizer->GetMetadataCoreConfig().timestamps.sample_rate, 100);
  config.timestamps.level = Generators::Config::TimestampLevel::Off;
  config.timestamps.sample_rate = 1;
  config.timestamps.hop_length = 1;
  config.timestamps.subsampling_factor = 10;
  const char* keys[] = {"track_timestamp_metadata"};
  const char* values[] = {"false"};
  tokenizer->UpdateOptions(keys, values, 1);
  EXPECT_TRUE(state->TimestampsEnabled());
  stream->DecodeWithMetadata({tokens[0], 1, interval});
  ASSERT_NE(state->Metadata().timestampMetadata, nullptr);
  EXPECT_NO_THROW(state->ConsumeTimestamps());
  const auto& final = stream->FinalizeMetadata();
  ASSERT_NE(final.timestampMetadata, nullptr);
  ASSERT_EQ(final.timestampMetadata->word_count, 1U);
  EXPECT_DOUBLE_EQ(final.timestampMetadata->words[0].start_time, 0.4);
  EXPECT_DOUBLE_EQ(final.timestampMetadata->words[0].stop_time, 0.5);
  auto plain = tokenizer->CreateStream();
  plain->Decode(tokens[0]);
  EXPECT_THROW(plain->CreateMetadataCoreState(metadata_config), std::runtime_error);
}

TEST_F(MetadataCoreStateTests, ConfigurationOverlay) {
  Generators::MetadataCoreConfig config;
  Generators::OverlayMetadataCoreConfig(config, R"({"timestamps":{
    "level":"all", "segment_separators":[".","!"], "segment_gap_threshold_seconds":0.31
  }})");
  EXPECT_EQ(config.timestamps.level, Generators::Config::TimestampLevel::All);
  EXPECT_EQ(config.timestamps.segment_separators, (std::vector<std::string>{".", "!"}));
  EXPECT_EQ(config.timestamps.segment_gap_threshold_seconds, 0.31);
  auto stream = tokenizer->CreateStream();
  auto state = stream->CreateMetadataCoreState(config);
  EXPECT_EQ(state->TimestampsEnabled(), true);
  Generators::OverlayMetadataCoreConfig(config, R"({"timestamps":{"level":"off","segment_separators":[],"segment_gap_threshold_seconds":null}})");
  EXPECT_TRUE(state->TimestampsEnabled());
  EXPECT_TRUE(config.timestamps.segment_separators.empty());
  EXPECT_FALSE(config.timestamps.segment_gap_threshold_seconds.has_value());
  EXPECT_THROW(Generators::OverlayMetadataCoreConfig(config, R"({"timestamps":{"segment_gap_threshold_seconds":0.2,"level":"invalid"}})"), std::runtime_error);
  EXPECT_FALSE(config.timestamps.segment_gap_threshold_seconds.has_value());
  EXPECT_THROW(Generators::OverlayMetadataCoreConfig(config, R"({"timestamps":{"segment_separators":[1]}})"), std::runtime_error);
  EXPECT_THROW(Generators::OverlayMetadataCoreConfig(config, R"({"timestamps":{"segment_gap_threshold_seconds":-2}})"), std::runtime_error);
  EXPECT_THROW(Generators::OverlayMetadataCoreConfig(config, R"({"timestamps":{"sample_rate":200}})"), std::runtime_error);
  EXPECT_THROW(Generators::OverlayMetadataCoreConfig(config, R"({"timestamps":{"unknown":1}})"), std::runtime_error);
}

TEST_F(MetadataCoreStateTests, ExplicitGapSecondsRoundUsingModelTiming) {
  auto stream = tokenizer->CreateStream();
  Generators::MetadataCoreConfig config;
  Generators::OverlayMetadataCoreConfig(config, R"({"timestamps":{"level":"segment","segment_gap_threshold_seconds":0.26}})");
  auto state = stream->CreateMetadataCoreState(config);
  EXPECT_TRUE(state->TimestampsEnabled());
  stream->DecodeWithMetadata({tokens[0], 1, interval});
  EXPECT_EQ(state->Metadata().timestampMetadata->word_count, 0U);
  EXPECT_EQ(Generators::GetSegmentGapThresholdFrames(config.timestamps.segment_gap_threshold_seconds,
                                                     100, 10, 1),
            3);
}

TEST_F(MetadataCoreStateTests, MissingModelTimingRejectsExplicitMetadata) {
  Generators::Config config{fs::path{MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32"}, ""};
  config.model.type = "nemotron_speech";
  config.model.timestamp_level = Generators::Config::TimestampLevel::All;
  EXPECT_THROW({ Generators::Tokenizer invalid_tokenizer{config}; }, std::runtime_error);
  config.model.timestamp_level = Generators::Config::TimestampLevel::Off;
  auto missing_timing_tokenizer = std::make_shared<Generators::Tokenizer>(config);
  auto stream = missing_timing_tokenizer->CreateStream();
  Generators::MetadataCoreConfig requested;
  Generators::OverlayMetadataCoreConfig(requested, R"({"timestamps":{"level":"all"}})");
  EXPECT_THROW(stream->CreateMetadataCoreState(requested), std::runtime_error);
}

TEST_F(MetadataCoreStateTests, TokenizerConfigurationIsAnIndependentCopy) {
  auto config = tokenizer->GetMetadataCoreConfig();
  config.timestamps.level = Generators::Config::TimestampLevel::Off;
  config.timestamps.sample_rate = 1;
  config.timestamps.hop_length = 1;
  config.timestamps.subsampling_factor = 10;
  EXPECT_EQ(tokenizer->GetMetadataCoreConfig().timestamps.level, Generators::Config::TimestampLevel::All);
  EXPECT_EQ(tokenizer->GetMetadataCoreConfig().timestamps.sample_rate, 100);
  auto disabled_stream = tokenizer->CreateStream();
  auto disabled = disabled_stream->CreateMetadataCoreState(config);
  EXPECT_FALSE(disabled->TimestampsEnabled());
  auto enabled_stream = tokenizer->CreateStream();
  auto enabled = enabled_stream->CreateMetadataCoreStateUsingTokenizerConfig();
  EXPECT_TRUE(enabled->TimestampsEnabled());
  EXPECT_THROW(enabled_stream->CreateMetadataCoreStateUsingTokenizerConfig(), std::runtime_error);
  for (const auto token : tokens) {
    enabled_stream->DecodeWithMetadata({token, 1, interval});
    enabled->ConsumeTimestamps();
  }
  enabled_stream->FinalizeMetadata();
  const auto& result = enabled->ConsumeFinalTimestamps();
  ASSERT_FALSE(result.words.empty());
  for (const auto& word : result.words) {
    EXPECT_DOUBLE_EQ(word.start_time, word.start_frame * 0.1);
    EXPECT_DOUBLE_EQ(word.stop_time, word.stop_frame * 0.1);
  }

  Generators::Config text_config{fs::path{MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32"}, ""};
  auto text_tokenizer = std::make_shared<Generators::Tokenizer>(text_config);
  EXPECT_EQ(text_tokenizer->GetMetadataCoreConfig().timestamps.level, Generators::Config::TimestampLevel::Off);
  auto text_stream = text_tokenizer->CreateStream();
  auto text_state = text_stream->CreateMetadataCoreStateUsingTokenizerConfig();
  EXPECT_FALSE(text_state->TimestampsEnabled());
  text_stream->DecodeWithMetadata({tokens[0], 1, interval});
  EXPECT_EQ(text_state->Metadata().timestampMetadata, nullptr);
  EXPECT_NO_THROW(text_stream->DecodeWithMetadata({tokens[1], 1, interval}));
}

TEST_F(MetadataCoreStateTests, GenericMetadataRequiresExplicitState) {
  auto stream = tokenizer->CreateStream();
  EXPECT_THROW(stream->DecodeWithMetadata({tokens[0], 1, interval}), std::runtime_error);
  EXPECT_THROW(stream->FinalizeMetadata(), std::runtime_error);
  EXPECT_NO_THROW(stream->CreateMetadataCoreStateUsingTokenizerConfig());
  EXPECT_NO_THROW(stream->DecodeWithMetadata({tokens[0], 1, interval}));
  EXPECT_NO_THROW(stream->FinalizeMetadata());
  stream->Reset();
  EXPECT_THROW(stream->DecodeWithMetadata({tokens[0], 1, interval}), std::runtime_error);
  EXPECT_THROW(stream->FinalizeMetadata(), std::runtime_error);
  EXPECT_NO_THROW(stream->Decode(tokens[0]));
}

TEST_F(MetadataCoreStateTests, GeneratorOmitsTimingWhenDisabled) {
  model->config_->model.timestamp_level = Generators::Config::TimestampLevel::Off;
  source->SetTimestampsEnabled(false);
  source->last_token_timings_[0].token_id = tokens[0] + 1;
  source->last_token_timings_.pop_back();
  const auto records = generator->GetNextTokensWithMetadata();
  ASSERT_EQ(records.size(), tokens.size());
  auto plain = tokenizer->CreateStream();
  auto disabled = tokenizer->CreateStream();
  disabled->CreateMetadataCoreState(Generators::MetadataCoreConfig{});
  auto enabled = tokenizer->CreateStream();
  enabled->CreateMetadataCoreState(metadata_config);
  for (size_t index = 0; index < records.size(); ++index) {
    const auto& token = records[index];
    EXPECT_EQ(token.token_id, tokens[index]);
    EXPECT_EQ(token.has_token_acoustic_frame_interval, 0);
    const auto& result = disabled->DecodeWithMetadata(token);
    EXPECT_EQ(result.text, plain->Decode(tokens[index]));
    EXPECT_EQ(result.timestampMetadata, nullptr);
    EXPECT_THROW(enabled->DecodeWithMetadata(token), std::runtime_error);
  }
  EXPECT_EQ(enabled->FinalizeMetadata().timestampMetadata->word_count, 0U);
  for (const auto level : {Generators::Config::TimestampLevel::Word,
                           Generators::Config::TimestampLevel::Segment,
                           Generators::Config::TimestampLevel::All}) {
    model->config_->model.timestamp_level = level;
    source->SetTimestampsEnabled(true);
    EXPECT_THROW(generator->GetNextTokensWithMetadata(), std::runtime_error);
  }
}

TEST_F(MetadataCoreStateTests, GeneratedMetadataPreservesMultiFrameIntervals) {
  auto stream = tokenizer->CreateStream();
  stream->CreateMetadataCoreState(metadata_config);
  for (auto& timing : source->last_token_timings_) timing.stop_frame = 9;
  const auto emitted = generator->GetNextTokensWithMetadata();
  ASSERT_EQ(emitted.size(), tokens.size());
  std::string transcript;
  std::string words;
  for (size_t index = 0; index < tokens.size(); ++index) {
    const auto token = emitted[index];
    EXPECT_EQ(token.token_id, tokens[index]);
    ASSERT_EQ(token.has_token_acoustic_frame_interval, 1);
    EXPECT_EQ(token.token_acoustic_frame_interval.start, 4);
    EXPECT_EQ(token.token_acoustic_frame_interval.stop, 9);
    const auto& result = stream->DecodeWithMetadata(token);
    transcript += result.text;
    ASSERT_NE(result.timestampMetadata, nullptr);
    for (size_t word_index = 0; word_index < result.timestampMetadata->word_count; ++word_index) {
      words += result.timestampMetadata->words[word_index].text;
      EXPECT_EQ(result.timestampMetadata->words[word_index].start_frame, 4);
      EXPECT_EQ(result.timestampMetadata->words[word_index].stop_frame, 9);
    }
  }
  source->SetStep({}, 99);
  EXPECT_TRUE(generator->GetNextTokensWithMetadata().empty());
  const auto& trailing = stream->FinalizeMetadata();
  for (size_t index = 0; index < trailing.timestampMetadata->word_count; ++index) {
    const auto& word = trailing.timestampMetadata->words[index];
    words += word.text;
    EXPECT_EQ(word.start_frame, 4);
    EXPECT_EQ(word.stop_frame, 9);
    EXPECT_DOUBLE_EQ(word.start_time, 0.4);
    EXPECT_DOUBLE_EQ(word.stop_time, 0.9);
  }
  EXPECT_EQ(transcript, "Hello world.");
  EXPECT_EQ(words, transcript);
  EXPECT_EQ(stream->FinalizeMetadata().timestampMetadata->word_count, 0U);
  EXPECT_EQ(stream->FinalizeMetadata().timestampMetadata->segment_count, 0U);
}

TEST_F(MetadataCoreStateTests, RetainsGenerationTimingAfterGeneratorAdvances) {
  auto stream = tokenizer->CreateStream();
  auto state = stream->CreateMetadataCoreState(metadata_config);
  const auto token = generator->GetNextTokensWithMetadata()[0];
  stream->DecodeWithMetadata(token);
  source->SetStep({}, 99);
  EXPECT_TRUE(generator->GetNextTokensWithMetadata().empty());
  EXPECT_NO_THROW(state->ConsumeTimestamps());
  EXPECT_NO_THROW(state->ConsumeTimestamps());
  const auto* processed = &state->ProcessMetadata();
  EXPECT_EQ(&state->ProcessMetadata(), processed);
  stream->FinalizeMetadata();
  const auto& result = state->ConsumeFinalTimestamps();
  ASSERT_EQ(result.words.size(), 1U);
  EXPECT_EQ(result.words[0].start_frame, 4);
  EXPECT_EQ(result.words[0].stop_frame, 5);
}

TEST_F(MetadataCoreStateTests, EmptyFinalizationAndInputValidation) {
  const OgaTokenMetadataAcousticFrameInterval negative_interval{-1, 0};
  const OgaTokenMetadataAcousticFrameInterval empty_interval{3, 3};
  const OgaTokenMetadataAcousticFrameInterval reversed_interval{4, 2};
  const OgaTokenMetadataAcousticFrameInterval valid_interval{0, 3};
  auto stream = tokenizer->CreateStream();
  auto state = stream->CreateMetadataCoreState(metadata_config);
  stream->FinalizeMetadata();
  EXPECT_EQ(state->Metadata().timestampMetadata, nullptr);
  EXPECT_TRUE(state->ConsumeFinalTimestamps().words.empty());
  EXPECT_THROW(stream->DecodeWithMetadata({tokens[0], 1, negative_interval}), std::runtime_error);
  EXPECT_THROW(stream->DecodeWithMetadata({tokens[0], 1, empty_interval}), std::runtime_error);
  EXPECT_THROW(stream->DecodeWithMetadata({tokens[0], 1, reversed_interval}), std::runtime_error);
  EXPECT_THROW(stream->DecodeWithMetadata({tokens[0], 0, {}}), std::runtime_error);
  source->last_token_timings_[0] = {tokens[0] + 1, 0, 1};
  EXPECT_THROW(generator->GetNextTokensWithMetadata(), std::runtime_error);
  source->last_token_timings_.pop_back();
  EXPECT_THROW(generator->GetNextTokensWithMetadata(), std::runtime_error);
  source->last_token_timings_.clear();
  EXPECT_THROW(generator->GetNextTokensWithMetadata(), std::runtime_error);
  const OgaTokenMetadataInput token{tokens[0], 0, {}};
  EXPECT_THROW(stream->DecodeWithMetadata(token), std::runtime_error);
  EXPECT_NO_THROW(stream->DecodeWithMetadata({tokens[0], 1, valid_interval}));
  const auto& final = stream->FinalizeMetadata();
  ASSERT_EQ(final.timestampMetadata->word_count, 1U);
  EXPECT_EQ(final.timestampMetadata->words[0].start_frame, 0);
  EXPECT_EQ(final.timestampMetadata->words[0].stop_frame, 3);
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
