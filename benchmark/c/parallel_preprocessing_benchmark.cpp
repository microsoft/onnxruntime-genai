// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "models/preprocessing/nemotron_streaming_processor.h"
#include "models/preprocessing/qwen2_5_vl_image_processor.h"
#include "models/preprocessing/videochat_flash_processor.h"
#include "models/threadpool.h"

namespace {

using Clock = std::chrono::steady_clock;
volatile float benchmark_sink;

void Consume(const void* output) {
  benchmark_sink = static_cast<const unsigned char*>(output)[0];
}

double Measure(const std::function<void()>& function) {
  for (int i = 0; i < 5; ++i)
    function();
  std::vector<double> samples;
  for (int i = 0; i < 30; ++i) {
    const auto start = Clock::now();
    function();
    const auto end = Clock::now();
    samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

void ScalarQwen(const float* source, float* destination, int64_t height, int64_t width,
                int64_t channels, int64_t patch_size, int64_t temporal) {
  int64_t write = 0;
  for (int64_t ph = 0; ph < height / patch_size; ++ph)
    for (int64_t pw = 0; pw < width / patch_size; ++pw)
      for (int64_t t = 0; t < temporal; ++t)
        for (int64_t c = 0; c < channels; ++c)
          for (int64_t h = 0; h < patch_size; ++h)
            for (int64_t w = 0; w < patch_size; ++w)
              destination[write++] =
                  source[((ph * patch_size + h) * width + pw * patch_size + w) *
                             channels +
                         c];
}

void ScalarTranspose(const float* source, float* destination, int64_t images,
                     int64_t channels, int64_t height, int64_t width) {
  for (int64_t n = 0; n < images; ++n)
    for (int64_t c = 0; c < channels; ++c)
      for (int64_t h = 0; h < height; ++h)
        for (int64_t w = 0; w < width; ++w)
          destination[((n * channels + c) * height + h) * width + w] =
              source[((n * height + h) * width + w) * channels + c];
}

void PlaneParallelTranspose(Generators::ThreadPool* thread_pool, const float* source,
                            float* destination, int64_t images, int64_t channels,
                            int64_t height, int64_t width) {
  const int64_t plane_size = height * width;
  Generators::ThreadPool::TryParallelFor(
      thread_pool, images * channels, static_cast<double>(plane_size),
      [&](std::ptrdiff_t first, std::ptrdiff_t last) {
        for (auto plane = first; plane < last; ++plane) {
          const int64_t image = plane / channels;
          const int64_t channel = plane % channels;
          for (int64_t i = 0; i < plane_size; ++i)
            destination[plane * plane_size + i] =
                source[(image * plane_size + i) * channels + channel];
        }
      });
}

void PrintResult(const std::string& operation, const std::string& dimensions,
                 size_t workers, double scalar_us, double parallel_us) {
  std::cout << std::left << std::setw(14) << operation << std::setw(22) << dimensions
            << std::right << std::setw(8) << workers << std::setw(14) << scalar_us
            << std::setw(14) << parallel_us << std::setw(12)
            << scalar_us / parallel_us << '\n';
}

void BenchmarkQwen(size_t workers, int64_t patch, int64_t height, int64_t width) {
  constexpr int64_t channels = 3;
  constexpr int64_t temporal = 2;
  const int64_t output_size =
      (height / patch) * (width / patch) * channels * temporal * patch * patch;
  std::vector<float> source(static_cast<size_t>(height * width * channels), 1.0f);
  std::vector<float> output(static_cast<size_t>(output_size));
  Generators::ThreadPool pool{workers};
  const double scalar = Measure([&] {
    ScalarQwen(source.data(), output.data(), height, width, channels, patch, temporal);
    Consume(output.data());
  });
  const double parallel = Measure([&] {
    Generators::ExtractQwenImagePatches(&pool, source.data(), output.data(), height,
                                        width, channels, patch, temporal);
    Consume(output.data());
  });
  PrintResult("qwen", std::to_string(height) + "x" + std::to_string(width) + " p" + std::to_string(patch),
              workers, scalar, parallel);
}

void BenchmarkVideo(size_t workers, int64_t images, int64_t height, int64_t width) {
  constexpr int64_t channels = 3;
  std::vector<float> source(static_cast<size_t>(images * height * width * channels), 1.0f);
  std::vector<float> output(source.size());
  Generators::ThreadPool pool{workers};
  const double scalar = Measure(
      [&] {
        ScalarTranspose(source.data(), output.data(), images, channels, height, width);
        Consume(output.data());
      });
  const double plane = Measure([&] {
    PlaneParallelTranspose(&pool, source.data(), output.data(), images, channels,
                           height, width);
    Consume(output.data());
  });
  const double row = Measure([&] {
    Generators::TransposeVideoChatFlashHwcToChw(
        &pool, source.data(), output.data(), images, channels, height, width);
    Consume(output.data());
  });
  const auto dimensions = std::to_string(images) + "x" + std::to_string(height) +
                          "x" + std::to_string(width);
  PrintResult("video-plane", dimensions, workers, scalar, plane);
  PrintResult("video-row", dimensions, workers, scalar, row);
}

void BenchmarkMel(size_t workers, int frames, int mels,
                  ONNXTensorElementDataType type) {
  constexpr int cache_frames = 16;
  std::vector<float> cache(static_cast<size_t>(cache_frames) * static_cast<size_t>(mels), 1.0f);
  std::vector<float> mel(static_cast<size_t>(frames) * static_cast<size_t>(mels), 2.0f);
  auto& allocator = Ort::Allocator::GetWithDefaultOptions();
  auto output = OrtValue::CreateTensor(
      allocator, std::array<int64_t, 3>{1, cache_frames + frames, mels}, type);
  Generators::ThreadPool pool{workers};
  const double sequential = Measure([&] {
    Generators::PopulateMelTensor(nullptr, *output, cache, 13, mel, frames, mels);
    Consume(output->GetTensorRawData());
  });
  const double parallel = Measure([&] {
    Generators::PopulateMelTensor(&pool, *output, cache, 13, mel, frames, mels);
    Consume(output->GetTensorRawData());
  });
  PrintResult(type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? "mel-fp32" : "mel-fp16",
              std::to_string(frames) + "x" + std::to_string(mels), workers,
              sequential, parallel);
}

}  // namespace

int main() {
  std::cout << "hardware_threads=" << std::thread::hardware_concurrency() << '\n';
#ifdef NDEBUG
  std::cout << "build=Release\n";
#else
  std::cout << "build=Debug\n";
#endif
#ifdef _WIN32
  std::cout << "os=Windows\n";
#elif defined(__APPLE__)
  std::cout << "os=macOS\n";
#else
  std::cout << "os=Linux\n";
#endif
#ifdef _MSC_VER
  std::cout << "compiler=MSVC " << _MSC_VER << '\n';
#elif defined(__clang__)
  std::cout << "compiler=Clang " << __clang_version__ << '\n';
#else
  std::cout << "compiler=GCC " << __VERSION__ << '\n';
#endif
  std::cout << std::fixed << std::setprecision(2)
            << "operation     dimensions             workers     scalar_us   parallel_us     speedup\n";
  for (size_t workers : {0U, 1U, 2U, 3U}) {
    BenchmarkQwen(workers, 14, 56, 56);
    BenchmarkQwen(workers, 14, 448, 448);
    BenchmarkQwen(workers, 14, 896, 896);
    BenchmarkQwen(workers, 16, 448, 448);
    BenchmarkVideo(workers, 1, 64, 64);
    BenchmarkVideo(workers, 1, 448, 448);
    BenchmarkVideo(workers, 8, 448, 448);
    BenchmarkMel(workers, 4, 128, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    BenchmarkMel(workers, 64, 128, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    BenchmarkMel(workers, 4, 128, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    BenchmarkMel(workers, 64, 128, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  }
}
