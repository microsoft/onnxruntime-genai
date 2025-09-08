// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if USE_CUDA
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "../src/cuda/cuda_topk.h"

// A struct to hold the parameters for a test configuration
struct TopKTestParams {
  int batch_size;
  int vocab_size;
  int k;
};

// Function to compare final raw scores and indices
bool CompareResults(const std::vector<float>& reference_scores, const std::vector<int>& reference_indices,
                    const std::vector<float>& actual_scores, const std::vector<int>& actual_indices,
                    const std::string& algo_name, int batch_size, int k) {
  bool match = true;
  const float epsilon = 1e-4f;

  for (int b = 0; b < batch_size && match; ++b) {
    for (int i = 0; i < k; ++i) {
      size_t idx = static_cast<size_t>(b) * k + i;
      if (reference_indices[idx] != actual_indices[idx] ||
          std::abs(reference_scores[idx] - actual_scores[idx]) > epsilon) {
        std::cerr << "Parity Test Failed for " << algo_name << ": Mismatch in batch " << b << " at position " << i
                  << ". Expected: (" << reference_indices[idx] << ", " << std::fixed << std::setprecision(6)
                  << reference_scores[idx] << "), Got: (" << actual_indices[idx] << ", " << actual_scores[idx] << ")"
                  << std::endl;
        match = false;
        break;
      }
    }
  }

  return match;
}

// Function to run parity tests for all algorithms against a reference implementation (Full Sort)
void RunParityTests(const TopKTestParams& params) {
  std::cout << "\n--- Running Parity Tests with batch_size=" << params.batch_size
            << ", vocab_size=" << params.vocab_size << ", k=" << params.k << " ---\n";

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  size_t total_vocab_size = static_cast<size_t>(params.batch_size) * params.vocab_size;
  size_t topk_size = static_cast<size_t>(params.batch_size) * params.k;

  auto scores_in_d = Generators::CudaMallocArray<float>(total_vocab_size);

  // Use a fixed seed for reproducibility
  std::mt19937 gen(3407);
  std::uniform_real_distribution<float> dis(0.0f, 100.0f);
  std::vector<float> scores_in_h(total_vocab_size);
  for (auto& val : scores_in_h) {
    val = dis(gen);
  }
  CUDA_CHECK(cudaMemcpy(scores_in_d.get(), scores_in_h.data(), scores_in_h.size() * sizeof(float), cudaMemcpyHostToDevice));

  // --- Get Reference Result using Full Sort ---
  auto topk_data = std::make_unique<Generators::cuda::TopkDataCompact>(params.batch_size, params.vocab_size, stream);
  Generators::cuda::RunTopKViaFullSort(topk_data.get(), stream, scores_in_d.get(),
                                       params.vocab_size, params.batch_size, params.k);
  topk_data->CompactOutput(params.batch_size, params.vocab_size, stream, params.k);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<float> ref_scores_h(topk_size);
  std::vector<int> ref_indices_h(topk_size);
  CUDA_CHECK(cudaMemcpy(ref_scores_h.data(), topk_data->topk_scores_compact.get(), ref_scores_h.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(ref_indices_h.data(), topk_data->topk_indices_compact.get(), ref_indices_h.size() * sizeof(int),
                        cudaMemcpyDeviceToHost));

  // --- Test Other Algorithms ---
  auto test_algo = [&](const std::string& name, auto func) {
    func();

    std::vector<float> actual_scores_h(topk_size);
    std::vector<int> actual_indices_h(topk_size);

    topk_data->CompactOutput(params.batch_size, params.vocab_size, stream, params.k);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(actual_scores_h.data(), topk_data->topk_scores_compact.get(), actual_scores_h.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(actual_indices_h.data(), topk_data->topk_indices_compact.get(), actual_indices_h.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));

    ASSERT_TRUE(CompareResults(ref_scores_h, ref_indices_h, actual_scores_h, actual_indices_h, name, params.batch_size,
                               params.k));
    std::cout << "  [PASS] " << name << " (Raw Scores & Indices)" << std::endl;
  };

  test_algo("SELECTION_SORT", [&]() {
    Generators::cuda::RunTopKViaSelectionSort(topk_data.get(), stream, scores_in_d.get(),
                                              params.vocab_size, params.batch_size, params.k);
  });

  if (params.k <= Generators::cuda::kHybridSortMaxK) {
    for (int partition_size : {1024, 2048, 4096, 8192}) {
      if (partition_size > 1024 && partition_size > params.vocab_size * 2) {
        continue;
      }

      topk_data->hybrid_sort_partition_size = partition_size;
      std::string algo_name = "HYBRID (" + std::to_string(partition_size) + ")";
      test_algo(algo_name, [&]() {
        Generators::cuda::RunTopKViaHybridSort(topk_data.get(), stream, scores_in_d.get(),
                                               params.vocab_size, params.batch_size, params.k);
      });
    }
  }

  test_algo("RADIX_SORT", [&]() {
    Generators::cuda::RunTopKViaRadixSort(topk_data.get(), stream, scores_in_d.get(),
                                          params.vocab_size, params.batch_size, params.k);
  });

  CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(TopKTests, ParityTests) {
  std::vector<TopKTestParams> test_cases = {
      {1, 10000, 50},
      {2, 10000, Generators::cuda::kHybridSortMaxK},
      {3, 32000, 100},
      {1, 32000, 16},
      {1, 512000, 50},
      {4, 1024, 18},
      {1, 256, 16},
      {2, 128, 5}};

  for (const auto& params : test_cases) {
    RunParityTests(params);
  }
}
#endif
