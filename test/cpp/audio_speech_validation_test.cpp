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
  EXPECT_NO_THROW(Generators::ValidateNemotronMelInputShape({1, 80, 1234}, 80));

  const std::string rank_error = GetExceptionMessage([] {
    Generators::ValidateNemotronMelInputShape({80, 1234}, 80);
  });
  EXPECT_NE(rank_error.find("rank 3"), std::string::npos);
}

TEST(AudioSpeechValidationTests, NemotronMelDimensionValidation) {
  const std::string mels_error = GetExceptionMessage([] {
    Generators::ValidateNemotronMelInputShape({1, 79, 1234}, 80);
  });
  EXPECT_NE(mels_error.find("expected num_mels"), std::string::npos);
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
  EXPECT_THROW(Generators::ValidateNemotronMelInputShape({3000}, 80), std::runtime_error);
  EXPECT_THROW(Generators::ValidateParakeetEncoderOutputShape({512}, 512), std::runtime_error);
  EXPECT_THROW(Generators::ValidateParakeetDecoderOutputShape({1024}, 1024), std::runtime_error);
}

TEST(AudioSpeechValidationTests, DimensionMismatchesCaught) {
  EXPECT_THROW(Generators::ValidateWhisperAudioFeaturesShape({1, 80, 2999}, 3000), std::runtime_error);
  EXPECT_THROW(Generators::ValidateNemotronMelInputShape({2, 80, 3000}, 80), std::runtime_error);
  EXPECT_THROW(Generators::ValidateParakeetEncoderOutputShape({1, 256, 64}, 512), std::runtime_error);
  EXPECT_THROW(Generators::ValidateParakeetDecoderOutputShape({2, 1024, 1}, 1024), std::runtime_error);
}
