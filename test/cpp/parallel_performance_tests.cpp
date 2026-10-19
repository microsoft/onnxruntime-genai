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
#include "models/preprocessing/lfm2_vl_image_processor.h"
#include "models/preprocessing/nemotron_streaming_processor.h"
#include "models/preprocessing/qwen2_5_vl_image_processor.h"
#include "models/preprocessing/videochat_flash_processor.h"
#include "preprocessing_reference_kernels.h"

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
                        const std::function<void()>& parallel,
                        const std::function<void()>& caller_only = {}) {
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
  if (caller_only) {
    const auto [old_us, caller_us] = MeasurePair(sequential, caller_only);
    std::cout << "[ CALLER CONTROL ] " << operation << ": pre-PR=" << old_us
              << " us, current-null-pool=" << caller_us << " us\n";
  }
#else
  GTEST_SKIP() << operation << " performance is measured only in Release builds";
#endif
}

TEST(ParallelPerformanceTests, QwenPatchExtractionDoesNotRegress) {
  constexpr int64_t channels = 3, patch = 14, temporal = 2;
  ThreadPool pool{3};
  for (int64_t side : {56, 448, 896}) {
    SCOPED_TRACE(side);
    const size_t element_count = static_cast<size_t>(side * side * channels);
    std::vector<float> source(element_count);
    std::iota(source.begin(), source.end(), 0.0f);
    std::vector<float> sequential_output(element_count * temporal);
    std::vector<float> parallel_output(sequential_output.size());
    std::vector<float> caller_output(sequential_output.size());
    ExpectNoRegression(
        "Qwen patch extraction",
        [&] {
          ExtractQwenPatchesSequential(source.data(), sequential_output.data(),
                                       side, side, channels, patch);
          Consume(sequential_output.data());
        },
        [&] {
          ExtractQwenImagePatches(&pool, source.data(), parallel_output.data(), side,
                                  side, channels, patch, temporal);
          Consume(parallel_output.data());
        },
        [&] {
          ExtractQwenImagePatches(nullptr, source.data(), caller_output.data(), side,
                                  side, channels, patch, temporal);
          Consume(caller_output.data());
        });
    EXPECT_EQ(parallel_output, sequential_output);
    EXPECT_EQ(caller_output, sequential_output);
  }
}

TEST(ParallelPerformanceTests, VideoTransposeDoesNotRegress) {
  constexpr int64_t channels = 3;
  ThreadPool pool{3};
  for (const auto& [images, side] : {std::pair{1, 56}, std::pair{1, 112}, std::pair{8, 448}}) {
    SCOPED_TRACE(side);
    const size_t element_count = static_cast<size_t>(images * channels * side * side);
    std::vector<float> source(element_count);
    std::iota(source.begin(), source.end(), 0.0f);
    std::vector<float> sequential_output(element_count);
    std::vector<float> parallel_output(element_count);
    std::vector<float> caller_output(element_count);
    ExpectNoRegression(
        "VideoChatFlash transpose",
        [&] {
          TransposeVideoSequential(source.data(), sequential_output.data(),
                                   images, channels, side, side);
          Consume(sequential_output.data());
        },
        [&] {
          TransposeVideoChatFlashHwcToChw(&pool, source.data(), parallel_output.data(),
                                          images, channels, side, side);
          Consume(parallel_output.data());
        },
        [&] {
          TransposeVideoChatFlashHwcToChw(nullptr, source.data(), caller_output.data(),
                                          images, channels, side, side);
          Consume(caller_output.data());
        });
    EXPECT_EQ(parallel_output, sequential_output);
    EXPECT_EQ(caller_output, sequential_output);
  }
}

TEST(ParallelPerformanceTests, Lfm2PatchExtractionDoesNotRegress) {
  constexpr int64_t channels = 3, patch = 16;
  ThreadPool pool{3};
  for (int64_t side : {64, 128, 512}) {
    SCOPED_TRACE(side);
    const auto geometry = ComputeLfm2VlImageGeometry(side, side, patch, 2);
    const size_t element_count = static_cast<size_t>(channels * side * side);
    std::vector<float> source(element_count);
    std::iota(source.begin(), source.end(), 0.0f);
    std::vector<float> sequential_output(element_count);
    std::vector<float> parallel_output(element_count);
    std::vector<float> caller_output(element_count);
    ExpectNoRegression(
        "LFM2 patch extraction",
        [&] {
          WriteLfm2PatchesSequential(source.data(), sequential_output.data(),
                                     channels, side, side, patch);
          Consume(sequential_output.data());
        },
        [&] {
          WriteLfm2VlImagePatches(&pool, source.data(), channels, side, side,
                                  geometry, patch, parallel_output.data());
          Consume(parallel_output.data());
        },
        [&] {
          WriteLfm2VlImagePatches(nullptr, source.data(), channels, side, side,
                                  geometry, patch, caller_output.data());
          Consume(caller_output.data());
        });
    EXPECT_EQ(parallel_output, sequential_output);
    EXPECT_EQ(caller_output, sequential_output);
  }
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
