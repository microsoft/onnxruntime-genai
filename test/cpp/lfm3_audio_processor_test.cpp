// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "models/model_type.h"
#include "models/preprocessing/lfm2_audio_processor.h"
#include "models/preprocessing/lfm3_audio_processor.h"

// These pin the frame count and the log floor of the checkpoint's Lfm3AudioFeatureExtractor. test_lfm3_audio_models.py
// checks the mel values against the formula, and the opt-in test_lfm3_audio_checkpoint.py against the extractor.

namespace Generators::test {
namespace {

const Lfm3AudioMelConfig kConfig{128, 400, 160, 16000};

}  // namespace

TEST(Lfm3AudioMelFramesTest, MatchesTheCheckpointExtractor) {
  const std::vector<std::pair<int64_t, int64_t>> samples_and_frames{
      {0, 1}, {1, 1}, {400, 1}, {401, 2}, {560, 2}, {561, 3}, {720, 3}, {721, 4}, {16000, 99}, {16160, 100}, {480000, 2999}};
  for (const auto& [samples, frames] : samples_and_frames) {
    EXPECT_EQ(Lfm3AudioNumMelFrames(samples, kConfig), frames) << samples << " samples";
  }
}

TEST(Lfm3AudioMelTest, EmptyClipIsOneFrameAtTheFloor) {
  int64_t frames = 0;
  const float unused = 0.0f;
  const auto mel = ComputeLfm3AudioMel(&unused, 0, kConfig, frames);
  EXPECT_EQ(frames, 1);
  EXPECT_EQ(mel, std::vector<float>(128, -2.0f));
}

TEST(Lfm3AudioMelTest, SilenceStaysAtTheFloor) {
  int64_t frames = 0;
  const std::vector<float> silence(16000, 0.0f);
  const auto mel = ComputeLfm3AudioMel(silence.data(), 16000, kConfig, frames);
  EXPECT_EQ(frames, 99);
  EXPECT_EQ(mel, std::vector<float>(static_cast<size_t>(128 * 99), -2.0f));
}

// ResampleLfm2Audio is shared with LFM2-Audio; a rate that cannot be resampled must not reach its division.
TEST(Lfm3AudioResampleTest, RefusesRatesAtOrBelowZero) {
  const std::vector<float> pcm(16, 0.0f);
  EXPECT_THROW(ResampleLfm2Audio(pcm.data(), 16, 0, 16000, "Lfm3AudioProcessor"), std::runtime_error);
  EXPECT_THROW(ResampleLfm2Audio(pcm.data(), 16, -44100, 16000, "Lfm3AudioProcessor"), std::runtime_error);
  EXPECT_THROW(ResampleLfm2Audio(pcm.data(), 16, 44100, 0, "Lfm3AudioProcessor"), std::runtime_error);
}

TEST(Lfm3AudioModelTypeTest, RunsThroughTheSpeechPipelineNotThePlainDecoder) {
  EXPECT_TRUE(ModelType::IsALM("lfm3_audio"));
  EXPECT_FALSE(ModelType::IsLLM("lfm3_audio"));
  EXPECT_FALSE(ModelType::IsLFM2("lfm3_audio"));
}

}  // namespace Generators::test
