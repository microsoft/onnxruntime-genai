// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "generator/generators.h"
#include "models/multi_modal.h"
#include "models/preprocessing/lfm2_audio_processor.h"

// LFM2-Audio turns a clip into log-mel frames, one per 10 ms hop, and the encoder subsamples those
// by 8, so the decoder sees one token per 80 ms. These tests pin the three things that have to agree
// for the pipeline to line up: how many frames a clip is worth, how many of them the reference
// preprocessor counts as valid, and how the prompt is split around the audio markers.

namespace Generators::test {
namespace {

constexpr Lfm2AudioMelConfig kMelConfig{};  // settings of the published checkpoints
constexpr int64_t kSubsamplingFactor = 8;

template <typename Fn>
std::string CaptureThrowMessage(Fn&& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    return e.what();
  }
  return {};
}

}  // namespace

TEST(Lfm2AudioFramesTest, OneSecondIsOneHundredFramesPlusTheCenterPadOne) {
  // torch.stft(center=True) pads by fft_size / 2 on both sides, which buys exactly one extra frame.
  EXPECT_EQ(Lfm2AudioNumMelFrames(16000, kMelConfig), 101);
  // NeMo's get_seq_len counts num_samples / hop_length instead, one fewer.
  EXPECT_EQ(Lfm2AudioNumValidMelFrames(16000, kMelConfig), 100);
}

TEST(Lfm2AudioFramesTest, PartialHopsAreTruncated) {
  EXPECT_EQ(Lfm2AudioNumMelFrames(16159, kMelConfig), 101);
  EXPECT_EQ(Lfm2AudioNumMelFrames(16160, kMelConfig), 102);
  EXPECT_EQ(Lfm2AudioNumValidMelFrames(16159, kMelConfig), 100);
  EXPECT_EQ(Lfm2AudioNumValidMelFrames(16160, kMelConfig), 101);
}

TEST(Lfm2AudioFramesTest, ClipShorterThanTheWindowStillHasFrames) {
  // The center padding alone is fft_size samples wide, so even one sample produces frames; they are
  // just not all valid.
  EXPECT_EQ(Lfm2AudioNumMelFrames(1, kMelConfig), 1);
  EXPECT_EQ(Lfm2AudioNumValidMelFrames(1, kMelConfig), 0);
  EXPECT_EQ(Lfm2AudioNumMelFrames(0, kMelConfig), 1);
}

TEST(Lfm2AudioFramesTest, TokensRoundUpToWholeEncoderFrames) {
  EXPECT_EQ(Lfm2AudioNumTokens(8, kSubsamplingFactor), 1);
  EXPECT_EQ(Lfm2AudioNumTokens(9, kSubsamplingFactor), 2);
  EXPECT_EQ(Lfm2AudioNumTokens(101, kSubsamplingFactor), 13);
  EXPECT_EQ(Lfm2AudioNumTokens(0, kSubsamplingFactor), 0);
}

TEST(Lfm2AudioFramesTest, TokensRejectANonPositiveSubsamplingFactor) {
  EXPECT_NE(CaptureThrowMessage([] { Lfm2AudioNumTokens(64, 0); }).find("subsampling_factor must be positive"),
            std::string::npos);
}

TEST(Lfm2AudioMelTest, NormalizesEveryValidFrameAndZeroesTheRest) {
  // Half a second of a tone: every mel bin should end up zero-mean over the valid frames, and the
  // frames past the valid length are the pad_value NeMo writes.
  constexpr int64_t kNumSamples = 8000;
  std::vector<float> pcm(kNumSamples);
  for (int64_t i = 0; i < kNumSamples; ++i) {
    pcm[static_cast<size_t>(i)] = 0.25f * std::sin(2.0f * 3.14159265358979323846f * 440.0f * static_cast<float>(i) / 16000.0f);
  }

  int64_t num_frames = 0;
  const auto mel = ComputeLfm2AudioMel(pcm.data(), kNumSamples, kMelConfig, num_frames);

  const int64_t valid_frames = Lfm2AudioNumValidMelFrames(kNumSamples, kMelConfig);
  ASSERT_EQ(num_frames, Lfm2AudioNumMelFrames(kNumSamples, kMelConfig));
  ASSERT_EQ(num_frames, valid_frames + 1);
  ASSERT_EQ(mel.size(), static_cast<size_t>(num_frames) * static_cast<size_t>(kMelConfig.num_mels));

  for (int64_t m = 0; m < kMelConfig.num_mels; ++m) {
    double sum = 0.0;
    double square_sum = 0.0;
    for (int64_t t = 0; t < valid_frames; ++t) {
      const double value = mel[static_cast<size_t>(t) * static_cast<size_t>(kMelConfig.num_mels) + static_cast<size_t>(m)];
      sum += value;
      square_sum += value * value;
    }
    EXPECT_NEAR(sum / valid_frames, 0.0, 1e-4) << "mel bin " << m << " is not zero-mean";
    // Unbiased standard deviation of 1, up to the epsilon added to the divisor.
    EXPECT_NEAR(std::sqrt(square_sum / (valid_frames - 1)), 1.0, 1e-3) << "mel bin " << m << " is not unit-variance";
  }
  for (int64_t m = 0; m < kMelConfig.num_mels; ++m) {
    EXPECT_EQ(mel[static_cast<size_t>(valid_frames) * static_cast<size_t>(kMelConfig.num_mels) + static_cast<size_t>(m)], 0.0f);
  }
}

TEST(Lfm2AudioMelTest, RejectsAClipTooShortToNormalize) {
  const std::vector<float> pcm(160, 0.1f);  // one valid frame; the standard deviation would divide by zero
  int64_t num_frames = 0;
  const auto message = CaptureThrowMessage([&] { ComputeLfm2AudioMel(pcm.data(), 160, kMelConfig, num_frames); });
  EXPECT_NE(message.find("too short"), std::string::npos) << message;
  EXPECT_NE(message.find("320 samples are needed"), std::string::npos) << message;
}

TEST(Lfm2AudioMelTest, RejectsAWindowLongerThanTheFftSize) {
  Lfm2AudioMelConfig config = kMelConfig;
  config.win_length = config.fft_size + 1;
  int64_t num_frames = 0;
  const std::vector<float> pcm(16000, 0.0f);
  EXPECT_NE(CaptureThrowMessage([&] { ComputeLfm2AudioMel(pcm.data(), 16000, config, num_frames); })
                .find("invalid mel configuration"),
            std::string::npos);
}

TEST(Lfm2AudioPromptTest, SplitsAroundEveryMarker) {
  const auto segments = SplitLfm2AudioPrompt("before <|audio|> middle <|audio|> after", 2);
  ASSERT_EQ(segments.size(), 3u);
  EXPECT_EQ(segments[0], "before ");
  EXPECT_EQ(segments[1], " middle ");
  EXPECT_EQ(segments[2], " after");
}

TEST(Lfm2AudioPromptTest, KeepsEmptySegmentsAroundAdjacentMarkers) {
  // Adjacent markers and a prompt that is nothing but a marker still need their empty text segments,
  // so the clips stay in order and nothing is dropped.
  const auto segments = SplitLfm2AudioPrompt("<|audio|><|audio|>", 2);
  ASSERT_EQ(segments.size(), 3u);
  EXPECT_EQ(segments[0], "");
  EXPECT_EQ(segments[1], "");
  EXPECT_EQ(segments[2], "");
}

TEST(Lfm2AudioPromptTest, TextOnlyPromptWithNoClipsIsOneSegment) {
  const auto segments = SplitLfm2AudioPrompt("no audio here", 0);
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], "no audio here");
}

TEST(Lfm2AudioPromptTest, RejectsAMarkerCountThatDoesNotMatchTheClips) {
  EXPECT_NE(CaptureThrowMessage([] { SplitLfm2AudioPrompt("only text", 1); })
                .find("contains 0 <|audio|> markers but 1 audio clips"),
            std::string::npos);
  EXPECT_NE(CaptureThrowMessage([] { SplitLfm2AudioPrompt("<|audio|> and <|audio|>", 1); })
                .find("contains 2 <|audio|> markers but 1 audio clips"),
            std::string::npos);
  EXPECT_NE(CaptureThrowMessage([] { SplitLfm2AudioPrompt("<|audio|>", 0); })
                .find("contains 1 <|audio|> markers but 0 audio clips"),
            std::string::npos);
}

TEST(Lfm2AudioResampleTest, SameRateIsACopy) {
  const std::vector<float> samples{0.1f, -0.2f, 0.3f};
  EXPECT_EQ(ResampleLfm2Audio(samples.data(), 3, 16000, 16000), samples);
}

TEST(Lfm2AudioResampleTest, LengthIsTheCeilingOfTheRateRatio) {
  // torchaudio trims to ceil(target * num_samples / source), up as well as down.
  const std::vector<float> samples(1001, 0.0f);
  EXPECT_EQ(ResampleLfm2Audio(samples.data(), 1001, 44100, 16000).size(), 364u);
  EXPECT_EQ(ResampleLfm2Audio(samples.data(), 1001, 48000, 16000).size(), 334u);
  EXPECT_EQ(ResampleLfm2Audio(samples.data(), 1001, 8000, 16000).size(), 2002u);
}

TEST(Lfm2AudioResampleTest, KeepsAConstantSignalConstantAwayFromTheEdges) {
  // The kernels of every output phase sum to one, so only the zero padding at the ends shows.
  const std::vector<float> samples(4410, 0.5f);
  for (const int64_t source_rate : {8000, 22050, 44100}) {
    const auto resampled = ResampleLfm2Audio(samples.data(), 4410, source_rate, 16000);
    for (size_t i = resampled.size() / 4; i < 3 * resampled.size() / 4; ++i) {
      ASSERT_NEAR(resampled[i], 0.5f, 2e-3f) << source_rate << " Hz, sample " << i;
    }
  }
}

TEST(Lfm2AudioResampleTest, RejectsARateThatIsNotPositive) {
  const std::vector<float> samples(16, 0.0f);
  EXPECT_NE(CaptureThrowMessage([&] { ResampleLfm2Audio(samples.data(), 16, 0, 16000); }).find("cannot resample"),
            std::string::npos);
}

// The embedding and speech sessions are handed buffers allocated on the decoder's devices, so neither
// may be left on CPU by session_options of its own while such a buffer is device memory.
namespace {

Config CpuSubModelsConfig() {
  Config config;  // as a CPU export writes it: both graphs have session_options with no provider
  config.model.speech.session_options = Config::SessionOptions{};
  config.model.embedding.session_options = Config::SessionOptions{};
  return config;
}

}  // namespace

TEST(Lfm2AudioSessionDevicesTest, ACpuEmbeddingCannotTakeTheInputsOfADecoderOnCuda) {
  const auto message = CaptureThrowMessage([] {
    CheckLfm2AudioSessionDevices(CpuSubModelsConfig(), DeviceType::CUDA, DeviceType::CUDA, /*with_audio=*/false);
  });
  EXPECT_NE(message.find("model.embedding.session_options run the embedding model on CPU, but the decoder takes "
                         "its inputs in CUDA memory"),
            std::string::npos)
      << message;
}

TEST(Lfm2AudioSessionDevicesTest, CpuSubModelsTakeTextOnlyPromptsWhenTheInputsAreOnTheHost) {
  // WebGPU without graph capture keeps the decoder's inputs on the host, so text-only prompts run.
  EXPECT_NO_THROW(
      CheckLfm2AudioSessionDevices(CpuSubModelsConfig(), DeviceType::WEBGPU, DeviceType::CPU, /*with_audio=*/false));

  // Audio features are allocated on the decoder's device, which a CPU encoder cannot write.
  const auto message = CaptureThrowMessage([] {
    CheckLfm2AudioSessionDevices(CpuSubModelsConfig(), DeviceType::WEBGPU, DeviceType::CPU, /*with_audio=*/true);
  });
  EXPECT_NE(message.find("model.speech.session_options run the speech model on CPU, but the audio features are "
                         "passed in WebGPU memory"),
            std::string::npos)
      << message;

  // Nor can a CPU embedding read them.
  auto config = CpuSubModelsConfig();
  config.model.speech.session_options.reset();
  EXPECT_NE(CaptureThrowMessage([&] {
              CheckLfm2AudioSessionDevices(config, DeviceType::WEBGPU, DeviceType::CPU, /*with_audio=*/true);
            }).find("model.embedding.session_options"),
            std::string::npos);
}

TEST(Lfm2AudioSessionDevicesTest, SubModelsThatCanUseTheBuffersAreAccepted) {
  // Without session_options of their own both graphs follow the decoder.
  EXPECT_NO_THROW(CheckLfm2AudioSessionDevices(Config{}, DeviceType::CUDA, DeviceType::CUDA, /*with_audio=*/true));

  auto config = CpuSubModelsConfig();
  config.model.speech.session_options->providers = {"cuda"};
  config.model.embedding.session_options->providers = {"cuda"};
  EXPECT_NO_THROW(CheckLfm2AudioSessionDevices(config, DeviceType::CUDA, DeviceType::CUDA, /*with_audio=*/true));

  // An explicit CPU provider is still CPU.
  config.model.speech.session_options->providers = {"CPU"};
  EXPECT_THROW(CheckLfm2AudioSessionDevices(config, DeviceType::CUDA, DeviceType::CUDA, /*with_audio=*/true),
               std::runtime_error);

  // Everything on CPU, and a decoder whose buffers a CPU session can use: OpenVINO allocates from the CPU.
  for (const auto device : {DeviceType::CPU, DeviceType::OpenVINO}) {
    EXPECT_NO_THROW(CheckLfm2AudioSessionDevices(CpuSubModelsConfig(), device, device, /*with_audio=*/true));
  }
}

}  // namespace Generators::test
