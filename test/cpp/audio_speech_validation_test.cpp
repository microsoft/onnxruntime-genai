// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <functional>
#include <string>

#include "models/nemotron_speech.h"
#include "models/parakeet.h"
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

TEST(AudioSpeechValidationTests, NemotronWordTimestampsGroupDecodedTokenPieces) {
  Generators::NemotronWordTimestampBuilder builder;
  const std::map<int32_t, std::string> decoded{{1, " hello"}, {2, "world"}, {3, " again"}};
  const auto decode = [&decoded](std::span<const int32_t> tokens) {
    std::string text;
    for (const auto token : tokens) text += decoded.at(token);
    return text;
  };
  builder.AddToken(1, "▁hello", " hello", 1600, 2880, decode);
  builder.AddToken(2, "world", "world", 2880, 4160, decode);
  builder.AddToken(3, "▁again", " again", 5440, 6720, decode);

  auto words = builder.GetCompletedWords();
  ASSERT_EQ(words.size(), 1U);
  EXPECT_EQ(words[0].word, "helloworld");
  EXPECT_EQ(words[0].start_sample, 1600);
  EXPECT_EQ(words[0].end_sample, 4160);

  builder.Flush(decode);
  words = builder.GetCompletedWords();
  ASSERT_EQ(words.size(), 2U);
  EXPECT_EQ(words[1].word, "again");
  EXPECT_EQ(words[1].start_sample, 5440);
  EXPECT_EQ(words[1].end_sample, 6720);
}

TEST(AudioSpeechValidationTests, NemotronWordTimestampsAllowTokensOnSameFrame) {
  Generators::NemotronWordTimestampBuilder builder;
  const std::map<int32_t, std::string> decoded{{1, " multi"}, {2, "piece"}, {3, " next"}};
  const auto decode = [&decoded](std::span<const int32_t> tokens) {
    std::string text;
    for (const auto token : tokens) text += decoded.at(token);
    return text;
  };
  builder.AddToken(1, "▁multi", " multi", 8000, 9280, decode);
  builder.AddToken(2, "piece", "piece", 8000, 9280, decode);
  builder.AddToken(3, "▁next", " next", 10560, 11840, decode);

  auto words = builder.GetCompletedWords();
  ASSERT_EQ(words.size(), 1U);
  EXPECT_EQ(words[0].word, "multipiece");
  EXPECT_EQ(words[0].start_sample, 8000);
  EXPECT_EQ(words[0].end_sample, 9280);
}

TEST(AudioSpeechValidationTests, NemotronWordTimestampsAttachPunctuation) {
  Generators::NemotronWordTimestampBuilder builder;
  const std::map<int32_t, std::string> decoded{{1, " hello"}, {2, "."}, {3, " next"}};
  const auto decode = [&decoded](std::span<const int32_t> tokens) {
    std::string text;
    for (const auto token : tokens) text += decoded.at(token);
    return text;
  };
  builder.AddToken(1, "▁hello", " hello", 1600, 2880, decode);
  builder.AddToken(2, ".", ".", 2880, 4160, decode);
  builder.AddToken(3, "▁next", " next", 5440, 6720, decode);

  const auto words = builder.GetCompletedWords();
  ASSERT_EQ(words.size(), 1U);
  EXPECT_EQ(words[0].word, "hello.");
  EXPECT_EQ(words[0].start_sample, 1600);
  EXPECT_EQ(words[0].end_sample, 4160);
}
