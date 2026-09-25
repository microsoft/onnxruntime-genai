// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "models/parallel_utils.h"
#include "models/preprocessing/nemotron_streaming_processor.h"
#include "models/preprocessing/qwen2_5_vl_image_processor.h"
#include "models/preprocessing/videochat_flash_processor.h"

namespace Generators {
namespace {

using Clock = std::chrono::steady_clock;

volatile unsigned char performance_sink;

void Consume(const void* data) {
  performance_sink = static_cast<const unsigned char*>(data)[0];
}

double MedianMicroseconds(std::vector<double>& samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

std::pair<double, double> MeasurePair(const std::function<void()>& sequential,
                                      const std::function<void()>& parallel) {
  for (int warmup = 0; warmup < 3; ++warmup) {
    sequential();
    parallel();
  }

  std::vector<double> sequential_samples;
  std::vector<double> parallel_samples;
  constexpr int kSamples = 15;
  sequential_samples.reserve(kSamples);
  parallel_samples.reserve(kSamples);

  const auto measure = [](const std::function<void()>& function) {
    const auto start = Clock::now();
    function();
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
  };

  for (int sample = 0; sample < kSamples; ++sample) {
    if (sample % 2 == 0) {
      sequential_samples.push_back(measure(sequential));
      parallel_samples.push_back(measure(parallel));
    } else {
      parallel_samples.push_back(measure(parallel));
      sequential_samples.push_back(measure(sequential));
    }
  }

  return {MedianMicroseconds(sequential_samples), MedianMicroseconds(parallel_samples)};
}

void ExpectNoRegression(const char* operation, const std::function<void()>& sequential,
                        const std::function<void()>& parallel) {
#ifdef NDEBUG
  if (std::thread::hardware_concurrency() < 4) {
    GTEST_SKIP() << operation << " requires at least four hardware threads";
  }

  const auto [sequential_us, parallel_us] = MeasurePair(sequential, parallel);
  constexpr double kNoiseTolerance = 1.25;
  std::cout << "[ PERFORMANCE ] " << operation << ": sequential=" << sequential_us
            << " us, parallel=" << parallel_us
            << " us, speedup=" << sequential_us / parallel_us << '\n';
  EXPECT_LE(parallel_us, sequential_us * kNoiseTolerance)
      << operation << " regressed: sequential=" << sequential_us
      << " us, parallel=" << parallel_us << " us";
#else
  GTEST_SKIP() << operation << " performance is measured only in Release builds";
#endif
}

TEST(ParallelPerformanceTests, QwenPatchExtractionDoesNotRegress) {
  constexpr int64_t height = 896;
  constexpr int64_t width = 896;
  constexpr int64_t channels = 3;
  constexpr int64_t patch = 14;
  constexpr int64_t temporal = 2;
  const size_t element_count = static_cast<size_t>(height * width * channels);
  std::vector<float> source(element_count, 1.0f);
  std::vector<float> sequential_output(element_count * temporal);
  std::vector<float> parallel_output(sequential_output.size());
  ThreadPool pool{3};

  ExpectNoRegression(
      "Qwen patch extraction",
      [&] {
        ExtractQwenImagePatches(nullptr, source.data(), sequential_output.data(), height,
                                width, channels, patch, temporal);
        Consume(sequential_output.data());
      },
      [&] {
        ExtractQwenImagePatches(&pool, source.data(), parallel_output.data(), height,
                                width, channels, patch, temporal);
        Consume(parallel_output.data());
      });
}

TEST(ParallelPerformanceTests, VideoTransposeDoesNotRegress) {
  constexpr int64_t images = 8;
  constexpr int64_t channels = 3;
  constexpr int64_t height = 448;
  constexpr int64_t width = 448;
  const size_t element_count = static_cast<size_t>(images * channels * height * width);
  std::vector<float> source(element_count, 1.0f);
  std::vector<float> sequential_output(element_count);
  std::vector<float> parallel_output(element_count);
  ThreadPool pool{3};

  ExpectNoRegression(
      "VideoChatFlash transpose",
      [&] {
        TransposeVideoChatFlashHwcToChw(nullptr, source.data(), sequential_output.data(),
                                        images, channels, height, width);
        Consume(sequential_output.data());
      },
      [&] {
        TransposeVideoChatFlashHwcToChw(&pool, source.data(), parallel_output.data(),
                                        images, channels, height, width);
        Consume(parallel_output.data());
      });
}

TEST(ParallelPerformanceTests, NemotronMelPopulationDoesNotRegress) {
  constexpr int frames = 6000;
  constexpr int mels = 128;
  constexpr int cache_frames = 16;
  std::vector<float> cache(static_cast<size_t>(cache_frames * mels), 1.0f);
  std::vector<float> mel(static_cast<size_t>(frames * mels), 2.0f);
  auto& allocator = Ort::Allocator::GetWithDefaultOptions();
  auto sequential_output = OrtValue::CreateTensor<float>(
      allocator, std::array<int64_t, 3>{1, cache_frames + frames, mels});
  auto parallel_output = OrtValue::CreateTensor<float>(
      allocator, std::array<int64_t, 3>{1, cache_frames + frames, mels});
  ThreadPool pool{3};

  ExpectNoRegression(
      "Nemotron mel population",
      [&] {
        PopulateMelTensor(nullptr, *sequential_output, cache, 13, mel, frames, mels);
        Consume(sequential_output->GetTensorRawData());
      },
      [&] {
        PopulateMelTensor(&pool, *parallel_output, cache, 13, mel, frames, mels);
        Consume(parallel_output->GetTensorRawData());
      });

  auto sequential_fp16 = OrtValue::CreateTensor<Ort::Float16_t>(
      allocator, std::array<int64_t, 3>{1, cache_frames + frames, mels});
  auto parallel_fp16 = OrtValue::CreateTensor<Ort::Float16_t>(
      allocator, std::array<int64_t, 3>{1, cache_frames + frames, mels});
  ExpectNoRegression(
      "Nemotron FP16 mel population",
      [&] {
        PopulateMelTensor(nullptr, *sequential_fp16, cache, 13, mel, frames, mels);
        Consume(sequential_fp16->GetTensorRawData());
      },
      [&] {
        PopulateMelTensor(&pool, *parallel_fp16, cache, 13, mel, frames, mels);
        Consume(parallel_fp16->GetTensorRawData());
      });
}

TEST(ParallelPerformanceTests, GenericTransformAndCopyDoNotRegress) {
  constexpr size_t elements = 1U << 22;
  std::vector<int64_t> source(elements);
  std::iota(source.begin(), source.end(), 0);
  std::vector<float> sequential_values(elements);
  std::vector<float> parallel_values(elements);
  std::vector<int64_t> sequential_converted(elements);
  std::vector<int64_t> parallel_converted(elements);
  ThreadPool pool{3};

  ExpectNoRegression(
      "ParallelTransform",
      [&] {
        std::transform(source.begin(), source.end(), sequential_values.begin(),
                       [](int64_t value) { return static_cast<float>(value); });
        Consume(sequential_values.data());
      },
      [&] {
        ParallelTransform(&pool, std::span<const int64_t>{source},
                          std::span<float>{parallel_values}, 2.0,
                          [](int64_t value) { return static_cast<float>(value); });
        Consume(parallel_values.data());
      });

  ExpectNoRegression(
      "Reverse ParallelTransform",
      [&] {
        std::transform(sequential_values.begin(), sequential_values.end(),
                       sequential_converted.begin(),
                       [](float value) { return static_cast<int64_t>(value + 0.5f); });
        Consume(sequential_converted.data());
      },
      [&] {
        ParallelTransform(&pool, std::span<const float>{parallel_values},
                          std::span<int64_t>{parallel_converted}, 2.0,
                          [](float value) { return static_cast<int64_t>(value + 0.5f); });
        Consume(parallel_converted.data());
      });

  ExpectNoRegression(
      "ParallelCopy",
      [&] {
        std::copy(sequential_values.begin(), sequential_values.end(),
                  parallel_values.begin());
        Consume(parallel_values.data());
      },
      [&] {
        ParallelCopy(&pool, std::span<const float>{sequential_values},
                     std::span<float>{parallel_values});
        Consume(parallel_values.data());
      });
}

}  // namespace
}  // namespace Generators
