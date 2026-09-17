// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "models/parallel_utils.h"

namespace {
using Clock = std::chrono::steady_clock;
volatile unsigned char benchmark_sink;

double Measure(const std::function<void()>& function) {
  for (int i = 0; i < 5; ++i)
    function();
  const auto start = Clock::now();
  for (int i = 0; i < 30; ++i)
    function();
  return std::chrono::duration<double, std::micro>(Clock::now() - start).count() / 30;
}

template <typename T>
void Consume(const std::vector<T>& values) {
  benchmark_sink = reinterpret_cast<const unsigned char*>(values.data())[0];
}

void Print(const char* operation, size_t elements, size_t workers,
           double sequential, double parallel) {
  std::cout << std::left << std::setw(18) << operation << std::right
            << std::setw(12) << elements << std::setw(9) << workers
            << std::setw(14) << sequential << std::setw(14) << parallel
            << std::setw(12) << sequential / parallel << '\n';
}

void Run(size_t elements, size_t workers) {
  Generators::ThreadPool pool{workers};
  std::vector<int64_t> integers(elements);
  std::iota(integers.begin(), integers.end(), 0);
  std::vector<float> floats(elements);
  std::vector<int64_t> converted(elements);

  auto sequential = Measure([&] {
    std::transform(integers.begin(), integers.end(), floats.begin(),
                   [](int64_t value) { return static_cast<float>(value); });
    Consume(floats);
  });
  auto parallel = Measure([&] {
    Generators::ParallelTransform(
        &pool, std::span<const int64_t>{integers}, std::span<float>{floats}, 2.0,
        [](int64_t value) { return static_cast<float>(value); });
    Consume(floats);
  });
  Print("int64-to-float", elements, workers, sequential, parallel);

  sequential = Measure([&] {
    std::transform(floats.begin(), floats.end(), converted.begin(),
                   [](float value) { return static_cast<int64_t>(value + 0.5f); });
    Consume(converted);
  });
  parallel = Measure([&] {
    Generators::ParallelTransform(
        &pool, std::span<const float>{floats}, std::span<int64_t>{converted}, 2.0,
        [](float value) { return static_cast<int64_t>(value + 0.5f); });
    Consume(converted);
  });
  Print("float-to-int64", elements, workers, sequential, parallel);

  std::vector<float> copy(elements);
  sequential = Measure([&] {
    std::copy(floats.begin(), floats.end(), copy.begin());
    Consume(copy);
  });
  parallel = Measure([&] {
    Generators::ParallelCopy(&pool, std::span<const float>{floats},
                             std::span<float>{copy});
    Consume(copy);
  });
  Print("copy-fp32", elements, workers, sequential, parallel);
}
}  // namespace

int main() {
  std::cout << "operation             elements  workers     scalar_us   parallel_us     speedup\n";
  std::cout << std::fixed << std::setprecision(2);
  for (size_t workers : {0U, 1U, 2U, 3U})
    for (size_t elements : {256U, 8192U, 65536U, 1048576U})
      Run(elements, workers);
}
