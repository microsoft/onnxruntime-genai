// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#include "models/preprocessing/nemotron_streaming_processor.h"
#include "models/preprocessing/qwen2_5_vl_image_processor.h"
#include "models/preprocessing/videochat_flash_processor.h"
#include "models/threadpool.h"

namespace Generators {
namespace {

TEST(ThreadPoolTests, NullPoolRunsSynchronously) {
  std::vector<int> values(7);
  ThreadPool::TryParallelFor(nullptr, values.size(), 1.0,
                             [&](std::ptrdiff_t first, std::ptrdiff_t last) {
                               EXPECT_EQ(first, 0);
                               EXPECT_EQ(last, values.size());
                               std::fill(values.begin() + first, values.begin() + last, 1);
                             });
  EXPECT_EQ(values, std::vector<int>(7, 1));
}

TEST(ThreadPoolTests, RejectsNegativeAndAcceptsEmptyRanges) {
  bool called = false;
  EXPECT_THROW(ThreadPool::TryParallelFor(nullptr, -1, 1.0, [](auto, auto) {}),
               std::invalid_argument);
  EXPECT_NO_THROW(ThreadPool::TryParallelFor(nullptr, 0, 1.0,
                                             [&](auto, auto) { called = true; }));
  EXPECT_FALSE(called);
}

TEST(ThreadPoolTests, CoversEveryItemExactlyOnce) {
  for (size_t workers : {0U, 1U, 2U, 3U}) {
    ThreadPool pool{workers};
    std::vector<std::atomic<int>> visits(1000);
    for (auto& visit : visits)
      visit.store(0);
    ThreadPool::TryParallelFor(&pool, visits.size(), 100.0,
                               [&](std::ptrdiff_t first, std::ptrdiff_t last) {
                                 EXPECT_GE(first, 0);
                                 EXPECT_LE(first, last);
                                 EXPECT_LE(last, visits.size());
                                 for (auto i = first; i < last; ++i)
                                   ++visits[static_cast<size_t>(i)];
                               });
    for (const auto& visit : visits)
      EXPECT_EQ(visit.load(), 1);
  }
}

TEST(ThreadPoolTests, SmallAndSingleItemRangesStaySequential) {
  ThreadPool pool{3};
  int calls = 0;
  ThreadPool::TryParallelFor(&pool, 1, 100000.0, [&](auto first, auto last) {
    ++calls;
    EXPECT_EQ(first, 0);
    EXPECT_EQ(last, 1);
  });
  ThreadPool::TryParallelFor(&pool, 10, 1.0, [&](auto first, auto last) {
    ++calls;
    EXPECT_EQ(first, 0);
    EXPECT_EQ(last, 10);
  });
  EXPECT_EQ(calls, 2);
}

TEST(ThreadPoolTests, NestedCallsRunSynchronously) {
  ThreadPool pool{3};
  std::atomic<int> outer_items{};
  std::atomic<int> inner_items{};
  ThreadPool::TryParallelFor(&pool, 100, 1000.0, [&](auto first, auto last) {
    ThreadPool::TryParallelFor(&pool, 5, 100000.0, [&](auto inner_first, auto inner_last) {
      EXPECT_EQ(inner_first, 0);
      EXPECT_EQ(inner_last, 5);
      inner_items += static_cast<int>(inner_last - inner_first);
    });
    outer_items += static_cast<int>(last - first);
  });
  EXPECT_EQ(outer_items.load(), 100);
  EXPECT_GT(inner_items.load(), 0);
}

TEST(ThreadPoolTests, PropagatesExceptionsAndRemainsReusable) {
  ThreadPool pool{3};
  EXPECT_THROW(
      ThreadPool::TryParallelFor(&pool, 1000, 100.0, [](auto first, auto last) {
        for (auto i = first; i < last; ++i) {
          if (i == 0)
            throw std::runtime_error("worker failure");
        }
      }),
      std::runtime_error);

  for (int invocation = 0; invocation < 50; ++invocation) {
    std::atomic<int> count{};
    ThreadPool::TryParallelFor(&pool, 100, 1000.0,
                               [&](auto first, auto last) { count += static_cast<int>(last - first); });
    EXPECT_EQ(count.load(), 100);
  }
}

TEST(ThreadPoolTests, StopsAssigningChunksAfterException) {
  ThreadPool pool{3};
  std::atomic<int> processed{};
  EXPECT_THROW(
      ThreadPool::TryParallelFor(&pool, 1000, 100.0, [&](auto first, auto last) {
        if (first == 0)
          throw std::runtime_error("stop");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        processed += static_cast<int>(last - first);
      }),
      std::runtime_error);
  EXPECT_LT(processed.load(), 1000);
}

TEST(ThreadPoolTests, ComputeCompatibility) {
  ThreadPool pool{4};
  std::vector<std::atomic<int>> visits(4);
  for (auto& visit : visits)
    visit.store(0);
  pool.Compute([&](size_t i) { ++visits[i]; });
  for (const auto& visit : visits)
    EXPECT_EQ(visit.load(), 1);
}

std::vector<float> ScalarQwenPatches(const std::vector<float>& source, int64_t height,
                                     int64_t width, int64_t channels, int64_t patch_size,
                                     int64_t temporal) {
  const int64_t hp = height / patch_size;
  const int64_t wp = width / patch_size;
  const int64_t patch_dim = temporal * channels * patch_size * patch_size;
  std::vector<float> output(static_cast<size_t>(hp * wp * patch_dim));
  int64_t write = 0;
  for (int64_t ph = 0; ph < hp; ++ph)
    for (int64_t pw = 0; pw < wp; ++pw)
      for (int64_t t = 0; t < temporal; ++t)
        for (int64_t c = 0; c < channels; ++c)
          for (int64_t h = 0; h < patch_size; ++h)
            for (int64_t w = 0; w < patch_size; ++w)
              output[write++] = source[((ph * patch_size + h) * width +
                                        pw * patch_size + w) *
                                           channels +
                                       c];
  return output;
}

TEST(ParallelPreprocessingTests, QwenPatchExtractionMatchesScalarReference) {
  for (int64_t patch_size : {14, 16}) {
    for (int64_t channels : {1, 3, 4}) {
      const int64_t height = patch_size * 3;
      const int64_t width = patch_size * 2;
      std::vector<float> source(static_cast<size_t>(height * width * channels));
      std::iota(source.begin(), source.end(), 0.0f);
      const auto expected =
          ScalarQwenPatches(source, height, width, channels, patch_size, 2);
      for (size_t workers : {0U, 1U, 2U, 3U}) {
        ThreadPool pool{workers};
        std::vector<float> actual(expected.size());
        ExtractQwenImagePatches(&pool, source.data(), actual.data(), height, width,
                                channels, patch_size, 2);
        EXPECT_EQ(actual, expected);
      }
      std::vector<float> null_actual(expected.size());
      ExtractQwenImagePatches(nullptr, source.data(), null_actual.data(), height, width,
                              channels, patch_size, 2);
      EXPECT_EQ(null_actual, expected);
    }
  }
}

TEST(ParallelPreprocessingTests, QwenSinglePatchAndEmptyPatchInputs) {
  std::vector<float> source(14 * 14 * 3);
  std::iota(source.begin(), source.end(), 0.0f);
  const auto expected = ScalarQwenPatches(source, 14, 14, 3, 14, 2);
  std::vector<float> actual(expected.size());
  ThreadPool pool{3};
  ExtractQwenImagePatches(&pool, source.data(), actual.data(), 14, 14, 3, 14, 2);
  EXPECT_EQ(actual, expected);
  EXPECT_NO_THROW(
      ExtractQwenImagePatches(&pool, source.data(), actual.data(), 13, 14, 3, 14, 2));
}

TEST(ParallelPreprocessingTests, QwenLargeParallelExtractionMatchesScalarReference) {
  constexpr int64_t patch = 14;
  constexpr int64_t height = 448;
  constexpr int64_t width = 448;
  constexpr int64_t channels = 3;
  std::vector<float> source(height * width * channels);
  std::iota(source.begin(), source.end(), 0.0f);
  const auto expected = ScalarQwenPatches(source, height, width, channels, patch, 2);
  std::vector<float> actual(expected.size());
  ThreadPool pool{3};
  ExtractQwenImagePatches(&pool, source.data(), actual.data(), height, width,
                          channels, patch, 2);
  EXPECT_EQ(actual, expected);
}

std::vector<float> ScalarHwcToChw(const std::vector<float>& source, int64_t images,
                                  int64_t channels, int64_t height, int64_t width) {
  std::vector<float> output(source.size());
  for (int64_t n = 0; n < images; ++n)
    for (int64_t c = 0; c < channels; ++c)
      for (int64_t h = 0; h < height; ++h)
        for (int64_t w = 0; w < width; ++w)
          output[((n * channels + c) * height + h) * width + w] =
              source[((n * height + h) * width + w) * channels + c];
  return output;
}

TEST(ParallelPreprocessingTests, VideoTransposeMatchesScalarReference) {
  for (int64_t images : {1, 5}) {
    for (int64_t side : {4, 64, 448}) {
      constexpr int64_t channels = 3;
      std::vector<float> source(static_cast<size_t>(images * side * side * channels));
      std::iota(source.begin(), source.end(), 0.0f);
      const auto expected = ScalarHwcToChw(source, images, channels, side, side);
      for (size_t workers : {0U, 1U, 2U, 3U}) {
        ThreadPool pool{workers};
        std::vector<float> actual(source.size());
        TransposeVideoChatFlashHwcToChw(&pool, source.data(), actual.data(), images,
                                        channels, side, side);
        EXPECT_EQ(actual, expected);
      }
      std::vector<float> actual(source.size());
      TransposeVideoChatFlashHwcToChw(nullptr, source.data(), actual.data(), images,
                                      channels, side, side);
      EXPECT_EQ(actual, expected);
    }
  }
}

template <typename T>
std::vector<T> ScalarMel(const std::vector<float>& cache, int cache_pos,
                         const std::vector<float>& mel, int frames, int mels,
                         const std::function<T(float)>& convert) {
  const int cache_frames = static_cast<int>(cache.size()) / mels;
  std::vector<T> output(cache.size() + mel.size());
  for (int frame = 0; frame < cache_frames; ++frame)
    for (int bin = 0; bin < mels; ++bin)
      output[frame * mels + bin] =
          convert(cache[((cache_pos + frame) % cache_frames) * mels + bin]);
  for (int frame = 0; frame < frames; ++frame)
    for (int bin = 0; bin < mels; ++bin)
      output[(cache_frames + frame) * mels + bin] = convert(mel[bin * frames + frame]);
  return output;
}

TEST(ParallelPreprocessingTests, NemotronFloatAndFloat16MatchScalarReference) {
  constexpr int frames = 600;
  constexpr int mels = 128;
  std::vector<float> cache(7 * mels);
  std::vector<float> mel(frames * mels);
  std::iota(cache.begin(), cache.end(), -100.0f);
  std::iota(mel.begin(), mel.end(), 1.0f);
  auto& allocator = Ort::Allocator::GetWithDefaultOptions();

  for (size_t workers : {0U, 1U, 2U, 3U}) {
    ThreadPool pool{workers};
    auto fp32 = OrtValue::CreateTensor<float>(
        allocator, std::array<int64_t, 3>{1, 607, mels});
    PopulateMelTensor(&pool, *fp32, cache, 5, mel, frames, mels);
    const auto expected_fp32 =
        ScalarMel<float>(cache, 5, mel, frames, mels, [](float value) { return value; });
    EXPECT_TRUE(std::equal(expected_fp32.begin(), expected_fp32.end(),
                           fp32->GetTensorData<float>()));

    auto fp16 = OrtValue::CreateTensor<Ort::Float16_t>(
        allocator, std::array<int64_t, 3>{1, 607, mels});
    PopulateMelTensor(&pool, *fp16, cache, 5, mel, frames, mels);
    const auto expected_fp16 = ScalarMel<Ort::Float16_t>(
        cache, 5, mel, frames, mels,
        [](float value) { return Ort::Float16_t{FastFloat32ToFloat16(value)}; });
    EXPECT_TRUE(std::equal(expected_fp16.begin(), expected_fp16.end(),
                           fp16->GetTensorData<Ort::Float16_t>()));
  }
}

TEST(ParallelPreprocessingTests, NemotronCacheUpdateWrapsAcrossChunks) {
  constexpr int mels = 3;
  std::vector<float> cache(4 * mels, 0.0f);
  int cache_pos = 0;
  std::vector<float> first = {1, 2, 3, 4, 10, 20, 30, 40, 100, 200, 300, 400};
  UpdateMelCache(cache, cache_pos, first, 4, mels);
  EXPECT_EQ(cache_pos, 0);
  std::vector<float> second = {5, 6, 50, 60, 500, 600};
  UpdateMelCache(cache, cache_pos, second, 2, mels);
  EXPECT_EQ(cache_pos, 2);

  auto& allocator = Ort::Allocator::GetWithDefaultOptions();
  auto output = OrtValue::CreateTensor<float>(
      allocator, std::array<int64_t, 3>{1, 6, mels});
  PopulateMelTensor(nullptr, *output, cache, cache_pos, second, 2, mels);
  const auto expected =
      ScalarMel<float>(cache, cache_pos, second, 2, mels, [](float value) { return value; });
  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), output->GetTensorData<float>()));
}

}  // namespace
}  // namespace Generators

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
