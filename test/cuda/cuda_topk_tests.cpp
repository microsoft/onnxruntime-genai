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

#include "../../src/cuda/cuda_topk.h"

// A struct to hold the parameters for a test configuration
struct TopKTestParams {
  int batch_size;
  int vocab_size;
  int k;
};

// A single, unified function to compare results for both stable and unstable sorts.
bool CompareResults(const std::vector<float>& reference_scores, const std::vector<int>& reference_indices,
                    const std::vector<float>& actual_scores, const std::vector<int>& actual_indices,
                    const std::vector<float>& scores_in, int batch_size, int vocab_size, int k,
                    const std::string& algo_name) {
  bool match = true;
  const float epsilon = 1e-4f;

  for (int b = 0; b < batch_size && match; ++b) {
    for (int i = 0; i < k; ++i) {
      size_t idx = static_cast<size_t>(b) * k + i;

#ifdef STABLE_TOPK
      // --- STABLE SORT CHECK ---
      // For stable sort, both the score and the index must match the reference exactly.
      if (reference_indices[idx] != actual_indices[idx] ||
          std::abs(reference_scores[idx] - actual_scores[idx]) > epsilon) {
        std::cerr << "Parity Test Failed for " << algo_name << ": Mismatch in batch " << b << " at position " << i
                  << ". Expected: (" << reference_indices[idx] << ", " << std::fixed << std::setprecision(6)
                  << reference_scores[idx] << "), Got: (" << actual_indices[idx] << ", " << actual_scores[idx] << ")"
                  << std::endl;
        match = false;
        break;
      }
#else
      // --- UNSTABLE SORT CHECK ---
      // 1. The score must match the reference score.
      if (std::abs(reference_scores[idx] - actual_scores[idx]) > epsilon) {
        std::cerr << "Parity Test Failed for " << algo_name << ": Mismatch in batch " << b << " at position " << i
                  << ". Expected score: " << std::fixed << std::setprecision(6) << reference_scores[idx]
                  << ", Got score: " << actual_scores[idx] << std::endl;
        match = false;
        break;
      }

      // 2. The returned index must be valid (i.e., its score in the original input must match the returned score).
      //    This correctly handles tie-breaking, as different valid indices can be returned.
      size_t original_input_idx = static_cast<size_t>(b) * vocab_size + actual_indices[idx];
      if (std::abs(scores_in[original_input_idx] - actual_scores[idx]) > epsilon) {
        std::cerr << "Parity Test Failed for " << algo_name << ": Mismatch in batch " << b << " at position " << i
                  << ". Returned index " << actual_indices[idx] << " has original score "
                  << scores_in[original_input_idx] << " but top-k score was " << actual_scores[idx]
                  << std::endl;
        match = false;
        break;
      }
#endif
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
  // The constructor will self-allocate memory as the buffer argument is defaulted to nullptr.
  auto topk_data = std::make_unique<Generators::cuda::TopkDataCompact>(params.batch_size, params.vocab_size, stream);
  Generators::cuda::full_sort::RunTopK(topk_data.get(), stream, scores_in_d.get(),
                                       params.vocab_size, params.batch_size, params.k);
  topk_data->CompactOutput(params.batch_size, params.k, stream);
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

    topk_data->CompactOutput(params.batch_size, params.k, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(actual_scores_h.data(), topk_data->topk_scores_compact.get(), actual_scores_h.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(actual_indices_h.data(), topk_data->topk_indices_compact.get(), actual_indices_h.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));

    ASSERT_TRUE(CompareResults(ref_scores_h, ref_indices_h, actual_scores_h, actual_indices_h, scores_in_h,
                               params.batch_size, params.vocab_size, params.k, name));

    std::cout << "  [PASS] " << name << " (Raw Scores & Indices)" << std::endl;
  };

  test_algo(Generators::cuda::select_sort::kAlgorithmName, [&]() {
    Generators::cuda::select_sort::RunTopK(topk_data.get(), stream, scores_in_d.get(),
                                           params.vocab_size, params.batch_size, params.k);
  });

  test_algo(Generators::cuda::per_batch_radix_sort::kAlgorithmName, [&]() {
    Generators::cuda::per_batch_radix_sort::RunTopK(topk_data.get(), stream, scores_in_d.get(),
                                                    params.vocab_size, params.batch_size, params.k);
  });

  if (Generators::cuda::hybrid_sort::IsSupported(params.batch_size, params.vocab_size, params.k)) {
    test_algo(Generators::cuda::hybrid_sort::kAlgorithmName, [&]() {
      Generators::cuda::hybrid_sort::RunTopK(topk_data.get(), stream, scores_in_d.get(),
                                             params.vocab_size, params.batch_size, params.k);
    });
  }

  if (Generators::cuda::iterative_sort::IsSupported(params.batch_size, params.vocab_size, params.k)) {
    test_algo(Generators::cuda::iterative_sort::kAlgorithmName, [&]() {
      Generators::cuda::iterative_sort::RunTopK(topk_data.get(), stream, scores_in_d.get(),
                                                params.vocab_size, params.batch_size, params.k);
    });
  }

  if (Generators::cuda::cascaded_sort::IsSupported(params.batch_size, params.vocab_size, params.k)) {
    test_algo(Generators::cuda::cascaded_sort::kAlgorithmName, [&]() {
      Generators::cuda::cascaded_sort::RunTopK(topk_data.get(), stream, scores_in_d.get(),
                                               params.vocab_size, params.batch_size, params.k);
    });
  }

  if (Generators::cuda::flash_convergent::IsSupported(params.batch_size, params.vocab_size, params.k)) {
    test_algo(Generators::cuda::flash_convergent::kAlgorithmName, [&]() {
      Generators::cuda::flash_convergent::RunTopK(topk_data.get(), stream, scores_in_d.get(),
                                                  params.vocab_size, params.batch_size, params.k);
    });
  }

  if (Generators::cuda::distributed_select_sort::IsSupported(params.batch_size, params.vocab_size, params.k)) {
    test_algo(Generators::cuda::distributed_select_sort::kAlgorithmName, [&]() {
      Generators::cuda::distributed_select_sort::RunTopK(topk_data.get(), stream, scores_in_d.get(),
                                                         params.vocab_size, params.batch_size, params.k);
    });
  }

  // Test RunTopK (Backend can be any of the above algorithms).
  test_algo("DEFAULT", [&]() {
    Generators::cuda::RunTopK(topk_data.get(), stream, scores_in_d.get(),
                              params.vocab_size, params.batch_size, params.k);
  });

  CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(TopKTests, ParityTests) {
  std::vector<TopKTestParams> test_cases;

  std::vector<int> batch_sizes = {1, 4, 32};
  std::vector<int> vocab_sizes = {200, 2000, 20000, 200000};
  std::vector<int> ks = {1, 2, 4, 5, 8, 16, 32, 50, 64, 100, 128, 256};

  for (int batch_size : batch_sizes) {
    for (int vocab_size : vocab_sizes) {
      for (int k : ks) {
        if (k > vocab_size) continue;
        test_cases.push_back({batch_size, vocab_size, k});
      }
    }
  }

  for (const auto& params : test_cases) {
    RunParityTests(params);
  }
}
#endif
