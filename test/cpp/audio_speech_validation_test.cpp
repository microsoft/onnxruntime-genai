// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <functional>
#include <string>

#include "models/nemotron_speech.h"
#include "models/parakeet.h"
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

TEST(AudioSpeechValidationTests, NemotronTimestampsRejectMissingFrameDurationParameters) {
  Generators::Config config;
  config.model.type = "nemotron_speech";
  config.model.timestamp_level = Generators::Config::TimestampLevel::Word;
  config.model.segment_gap_threshold_seconds = 1.0;

  Generators::NemotronConfig nemotron_config;
  EXPECT_THROW(nemotron_config.PopulateFromConfig(config), std::runtime_error);
}

TEST(AudioSpeechValidationTests, NemotronTimestampsRejectOversizedSegmentGap) {
  Generators::Config config;
  config.model.type = "nemotron_speech";
  config.model.timestamp_level = Generators::Config::TimestampLevel::Segment;
  config.model.sample_rate = 16000;
  config.model.hop_length = 160;
  config.model.subsampling_factor = 8;
  config.model.segment_gap_threshold_seconds = 1e20;

  Generators::NemotronConfig nemotron_config;
  EXPECT_THROW(nemotron_config.PopulateFromConfig(config), std::runtime_error);
}

TEST(AudioSpeechValidationTests, ZeroAndSubFrameGapsSplitEachWord) {
  for (double threshold : {0.0, 0.01}) {
    EXPECT_EQ(Generators::GetSegmentGapThresholdFrames(threshold, 100, 10, 1), 0);
  }
}

class MetadataCoreStateTests : public testing::Test {
 protected:
  struct TestTransducerState : Generators::TransducerState {
    using TransducerState::last_token_intervals_;
    using TransducerState::TransducerState;
    void SetTimestampsEnabled(bool enabled) { timestamps_enabled_ = enabled; }
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
  std::string transcript;
  std::string words;
  std::string segments;

  for (size_t index = 0; index < tokens.size(); ++index) {
    const auto& expected = stream->DecodeWithMetadata({tokens[index], 1, interval});
    EXPECT_EQ(state->Text(), plain->Decode(tokens[index]));
    transcript += expected.text;
    ASSERT_NE(state->Metadata().timestampMetadata, nullptr);
    for (size_t word_index = 0; word_index < expected.timestampMetadata->word_count; ++word_index) {
      const auto& word = expected.timestampMetadata->words[word_index];
      words += word.text;
      EXPECT_EQ(word.start_frame, 4);
      EXPECT_EQ(word.stop_frame, 5);
      EXPECT_DOUBLE_EQ(word.start_time, 0.4);
      EXPECT_DOUBLE_EQ(word.stop_time, 0.5);
    }
    for (size_t segment_index = 0; segment_index < expected.timestampMetadata->segment_count; ++segment_index)
      segments += expected.timestampMetadata->segments[segment_index].text;
  }

  const auto& final = stream->FinalizeMetadata();
  for (size_t index = 0; index < final.timestampMetadata->word_count; ++index) {
    const auto& word = final.timestampMetadata->words[index];
    words += word.text;
    EXPECT_EQ(word.start_frame, 4);
    EXPECT_EQ(word.stop_frame, 5);
  }
  for (size_t index = 0; index < final.timestampMetadata->segment_count; ++index)
    segments += final.timestampMetadata->segments[index].text;
  EXPECT_EQ(transcript, "Hello world.");
  EXPECT_EQ(words, transcript);
  EXPECT_EQ(segments, transcript);
  const auto& repeated = stream->FinalizeMetadata();
  EXPECT_EQ(repeated.timestampMetadata->word_count, 0U);
  EXPECT_EQ(repeated.timestampMetadata->segment_count, 0U);
}

TEST_F(MetadataCoreStateTests, ZeroAndSubFrameGapsSplitWordsEvenOnSharedFrames) {
  for (const double threshold : {0.0, 0.01}) {
    auto config = metadata_config;
    config.timestamps.segment_separators.clear();
    config.timestamps.segment_gap_threshold_seconds = threshold;
    auto stream = tokenizer->CreateStream();
    stream->CreateMetadataCoreState(config);
    std::vector<std::string> words;
    std::vector<std::string> segments;
    const auto collect = [&](const OgaTokenMetadataOutput& result) {
      ASSERT_NE(result.timestampMetadata, nullptr);
      for (size_t i = 0; i < result.timestampMetadata->word_count; ++i)
        words.emplace_back(result.timestampMetadata->words[i].text);
      for (size_t i = 0; i < result.timestampMetadata->segment_count; ++i) {
        const auto& segment = result.timestampMetadata->segments[i];
        segments.emplace_back(segment.text);
        EXPECT_EQ(segment.start_frame, 4);
        EXPECT_EQ(segment.stop_frame, 5);
      }
    };
    for (const auto token : tokens)
      collect(stream->DecodeWithMetadata({token, 1, interval}));
    collect(stream->FinalizeMetadata());
    ASSERT_GE(words.size(), 2U);
    EXPECT_EQ(segments, words);
  }
}

TEST_F(MetadataCoreStateTests, FrameGapCompletesPriorSegmentBeforeNextWord) {
  auto config = metadata_config;
  config.timestamps.segment_separators.clear();
  config.timestamps.segment_gap_threshold_seconds = 0.3;  // Three frames at 100 Hz / 10 samples per frame.
  auto stream = tokenizer->CreateStream();
  stream->CreateMetadataCoreState(config);
  std::vector<std::string> segments;
  std::vector<std::pair<int64_t, int64_t>> bounds;
  size_t completed_words = 0;
  for (size_t index = 0; index < tokens.size(); ++index) {
    const int64_t frame = index == 0 ? 0 : 8;
    const auto& result = stream->DecodeWithMetadata({tokens[index], 1, {frame, frame + 1}});
    completed_words += result.timestampMetadata->word_count;
    for (size_t i = 0; i < result.timestampMetadata->segment_count; ++i) {
      const auto& segment = result.timestampMetadata->segments[i];
      segments.emplace_back(segment.text);
      bounds.emplace_back(segment.start_frame, segment.stop_frame);
    }
  }
  const auto& trailing = stream->FinalizeMetadata();
  completed_words += trailing.timestampMetadata->word_count;
  EXPECT_EQ(completed_words, 2U);
  for (size_t i = 0; i < trailing.timestampMetadata->segment_count; ++i) {
    const auto& segment = trailing.timestampMetadata->segments[i];
    segments.emplace_back(segment.text);
    bounds.emplace_back(segment.start_frame, segment.stop_frame);
  }
  ASSERT_EQ(segments.size(), 2U);
  EXPECT_EQ(bounds[0], (std::pair<int64_t, int64_t>{0, 1}));
  EXPECT_EQ(bounds[1], (std::pair<int64_t, int64_t>{8, 9}));
  EXPECT_EQ(segments[0] + segments[1], "Hello world.");
}

TEST_F(MetadataCoreStateTests, RepeatedTokenIdsKeepPositionSpecificIntervals) {
  source->SetStep({tokens[0], tokens[0]}, 4);
  source->last_token_intervals_[1] = {9, 12};
  const auto emitted = generator->GetNextTokensWithMetadata();
  ASSERT_EQ(emitted.size(), 2U);
  EXPECT_EQ(emitted[0].token_id, emitted[1].token_id);
  EXPECT_EQ(emitted[0].token_acoustic_frame_interval.start, 4);
  EXPECT_EQ(emitted[0].token_acoustic_frame_interval.stop, 5);
  EXPECT_EQ(emitted[1].token_acoustic_frame_interval.start, 9);
  EXPECT_EQ(emitted[1].token_acoustic_frame_interval.stop, 12);
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
    EXPECT_EQ(&state->Metadata(), current_metadata);
    EXPECT_EQ(state->Text(), current_text);
    EXPECT_FALSE(state->TimestampsEnabled());
    enabled->DecodeWithMetadata({tokens[index], 1, interval});
    EXPECT_NE(enabled_state->Metadata().timestampMetadata, nullptr);
  }
  stream->FinalizeMetadata();
  EXPECT_EQ(state->Metadata().timestampMetadata, nullptr);
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
  }
  const auto& result = enabled_stream->FinalizeMetadata();
  ASSERT_GT(result.timestampMetadata->word_count, 0U);
  for (size_t index = 0; index < result.timestampMetadata->word_count; ++index) {
    const auto& word = result.timestampMetadata->words[index];
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
  source->last_token_intervals_.pop_back();
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
  for (auto& timing : source->last_token_intervals_) timing.stop = 9;
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
  const auto& result = stream->FinalizeMetadata();
  ASSERT_EQ(result.timestampMetadata->word_count, 1U);
  EXPECT_EQ(result.timestampMetadata->words[0].start_frame, 4);
  EXPECT_EQ(result.timestampMetadata->words[0].stop_frame, 5);
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
  EXPECT_THROW(stream->DecodeWithMetadata({tokens[0], 1, negative_interval}), std::runtime_error);
  EXPECT_THROW(stream->DecodeWithMetadata({tokens[0], 1, empty_interval}), std::runtime_error);
  EXPECT_THROW(stream->DecodeWithMetadata({tokens[0], 1, reversed_interval}), std::runtime_error);
  EXPECT_THROW(stream->DecodeWithMetadata({tokens[0], 0, {}}), std::runtime_error);
  source->last_token_intervals_.pop_back();
  EXPECT_THROW(generator->GetNextTokensWithMetadata(), std::runtime_error);
  source->last_token_intervals_.clear();
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
