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
  config.model.timestamp_level = Generators::Config::TimestampLevel::All;
  config.model.sample_rate = 16000;
  config.model.hop_length = 160;
  config.model.subsampling_factor = 8;
  config.model.segment_separators = {".", "!"};
  config.model.segment_gap_threshold_frames = 12;

  Generators::NemotronConfig nemotron_config;
  EXPECT_NO_THROW(nemotron_config.PopulateFromConfig(config));
  EXPECT_EQ(nemotron_config.timestamp_level, Generators::Config::TimestampLevel::All);
  EXPECT_EQ(nemotron_config.segment_separators, (std::vector<std::string>{".", "!"}));
  EXPECT_EQ(nemotron_config.segment_gap_threshold_frames, 12);
}

TEST(AudioSpeechValidationTests, NemotronGlobalFrameUsesAbsoluteSampleOrigin) {
  EXPECT_EQ(Generators::GetNemotronGlobalFrame(0, 3, 160, 8), 3);
  EXPECT_EQ(Generators::GetNemotronGlobalFrame(25600, 3, 160, 8), 23);
  EXPECT_EQ(Generators::GetNemotronGlobalFrame(16000, 3, 160, 8), 15);
  EXPECT_THROW(Generators::GetNemotronGlobalFrame(-1, 0, 160, 8), std::runtime_error);
}

TEST(AudioSpeechValidationTests, NemotronTimestampsRequireFrameDurationParameters) {
  Generators::Config config;
  config.model.timestamp_level = Generators::Config::TimestampLevel::Word;

  Generators::NemotronConfig nemotron_config;
  EXPECT_THROW(nemotron_config.PopulateFromConfig(config), std::runtime_error);
}

TEST(AudioSpeechValidationTests, TimestampAccumulatorAttachesPunctuationAndCompletesSegment) {
  Generators::TimestampTokenizerConfig config{
      Generators::Config::TimestampLevel::All, {"."}, std::nullopt, 100, 10, 1};
  Generators::TimestampDecodeState state{config};

  state.Consume({10, 0, 1}, {nullptr, 0, 0});
  state.ClearResult();
  state.Consume({11, 1, 2}, {nullptr, 0, 0});
  state.ClearResult();
  const OrtxDetokenizedWord completed_word{" Hello.", 0, 2};
  state.Consume({12, 3, 4}, {&completed_word, 1, 2});

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
      Generators::Config::TimestampLevel::Segment, {}, 3, 100, 10, 1};
  Generators::TimestampDecodeState state{config};

  state.Consume({1, 0, 1}, {nullptr, 0, 0});
  state.ClearResult();
  const OrtxDetokenizedWord first_word{" one", 0, 1};
  state.Consume({2, 1, 2}, {&first_word, 1, 1});
  state.ClearResult();
  const OrtxDetokenizedWord second_word{" two", 1, 2};
  state.Consume({3, 8, 9}, {&second_word, 1, 2});
  state.ClearResult();
  const OrtxDetokenizedWord trailing_word{" three", 2, 3};
  state.Finalize({&trailing_word, 1, 3});

  EXPECT_TRUE(state.result_.words.empty());
  ASSERT_EQ(state.result_.segments.size(), 2u);
  EXPECT_EQ(state.result_.segments[0].text, " one two");
  EXPECT_EQ(state.result_.segments[1].text, " three");

  state.ClearResult();
  state.Finalize({nullptr, 0, 3});
  EXPECT_TRUE(state.result_.segments.empty());
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
