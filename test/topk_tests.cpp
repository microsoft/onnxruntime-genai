// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is for macro-benchmarking the GetTopKSubset function.

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <random>
#include <chrono>
#include <memory>
#include <cmath>

#include "cuda_runtime.h"
#include "../src/span.h"
#include "models/onnxruntime_api.h"
#include "../src/cuda/cuda_sampling.cuh"
#include "smartptrs.h" // For CudaMallocArray
#include <gtest/gtest.h>

// Forward declarations of the internal functions we want to benchmark,
// as they are not in the .cuh header.
namespace Generators {
namespace cuda {

void RunTopKViaDirectKernel(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature);
void RunTopKViaMapReduce(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions);
void RunTopKViaMapReduceShared(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions);
void RunTopKViaSingleKernelMapReduce(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions);
void RunTopKViaFullSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature);

} // namespace cuda
} // namespace Generators

// A struct to hold the parameters for a benchmark configuration
struct BenchmarkParams {
    int batch_size;
    int vocab_size;
    int k;
};

// A struct to hold the results of a single benchmark run
struct BenchmarkResult {
    BenchmarkParams params;
    std::string algo_name;
    int num_partitions;
    float latency_ms;
};

// Helper to check for CUDA errors
void CudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

// Function to compare results between a test algorithm and a reference
bool CompareResults(int batch_size, int k,
                    const std::vector<float>& reference_scores, const std::vector<int>& reference_indices,
                    const std::vector<float>& actual_scores, const std::vector<int>& actual_indices,
                    const std::string& algo_name) {
    bool match = true;
    const float epsilon = 1e-5f;

    for (size_t i = 0; i < reference_scores.size(); ++i) {
        // Compare indices
        if (reference_indices[i] != actual_indices[i]) {
            std::cerr << "Parity Test Failed for " << algo_name << ": Index mismatch at position " << i
                      << ". Expected: " << reference_indices[i] << ", Got: " << actual_indices[i] << std::endl;
            match = false;
            break;
        }
        // Compare scores
        if (std::abs(reference_scores[i] - actual_scores[i]) > epsilon) {
            std::cerr << "Parity Test Failed for " << algo_name << ": Score mismatch at position " << i
                      << ". Expected: " << reference_scores[i] << ", Got: " << actual_scores[i] << std::endl;
            match = false;
            break;
        }
    }
    if (!match) {
        // Optional: Dump full arrays on mismatch for debugging
        // std::cout << "Reference Indices: "; for(int v : reference_indices) std::cout << v << " "; std::cout << std::endl;
        // std::cout << "Actual Indices:    "; for(int v : actual_indices) std::cout << v << " "; std::cout << std::endl;
    }
    return match;
}

// Function to run parity tests for all algorithms against a reference implementation
void RunParityTests() {
    std::cout << "\n--- Running Parity Tests ---\n";
    BenchmarkParams params = {1, 200000, 50}; // A representative test case
    const float temperature = 1.0f;

    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));

    // --- Setup ---
    auto sampling_data = std::make_unique<Generators::cuda::SamplingData>(1234, params.batch_size, params.vocab_size, stream);
    auto scores_in_d = Generators::CudaMallocArray<float>(params.batch_size * params.vocab_size);
    
    // Use a fixed seed for reproducibility
    std::mt19937 gen(3407);
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);
    std::vector<float> scores_in_h(params.batch_size * params.vocab_size);
    for (auto& val : scores_in_h) {
        val = dis(gen);
    }
    CudaCheck(cudaMemcpy(scores_in_d.get(), scores_in_h.data(), scores_in_h.size() * sizeof(float), cudaMemcpyHostToDevice));

    // --- Get Reference Result using Full Sort ---
    auto ref_scores_d = Generators::CudaMallocArray<float>(params.batch_size * params.k);
    auto ref_indices_d = Generators::CudaMallocArray<int>(params.batch_size * params.k);
    Generators::cuda::RunTopKViaFullSort(sampling_data.get(), stream, scores_in_d.get(), ref_scores_d.get(), ref_indices_d.get(), params.vocab_size, params.batch_size, params.k, temperature);
    CudaCheck(cudaStreamSynchronize(stream));

    std::vector<float> ref_scores_h(params.batch_size * params.k);
    std::vector<int> ref_indices_h(params.batch_size * params.k);
    CudaCheck(cudaMemcpy(ref_scores_h.data(), ref_scores_d.get(), ref_scores_h.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CudaCheck(cudaMemcpy(ref_indices_h.data(), ref_indices_d.get(), ref_indices_h.size() * sizeof(int), cudaMemcpyDeviceToHost));

    // --- Test Other Algorithms ---
    auto test_algo = [&](const std::string& name, auto func) {
        auto actual_scores_d = Generators::CudaMallocArray<float>(params.batch_size * params.k);
        auto actual_indices_d = Generators::CudaMallocArray<int>(params.batch_size * params.k);
        func(actual_scores_d.get(), actual_indices_d.get());
        CudaCheck(cudaStreamSynchronize(stream));

        std::vector<float> actual_scores_h(params.batch_size * params.k);
        std::vector<int> actual_indices_h(params.batch_size * params.k);
        CudaCheck(cudaMemcpy(actual_scores_h.data(), actual_scores_d.get(), actual_scores_h.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CudaCheck(cudaMemcpy(actual_indices_h.data(), actual_indices_d.get(), actual_indices_h.size() * sizeof(int), cudaMemcpyDeviceToHost));

        if (CompareResults(params.batch_size, params.k, ref_scores_h, ref_indices_h, actual_scores_h, actual_indices_h, name)) {
            std::cout << "  [PASS] " << name << std::endl;
        } else {
            std::cout << "  [FAIL] " << name << std::endl;
        }
    };

    if (params.k <= 64) {
        test_algo("DIRECT_KERNEL", [&](float* s_d, int* i_d){
            Generators::cuda::RunTopKViaDirectKernel(sampling_data.get(), stream, scores_in_d.get(), s_d, i_d, params.vocab_size, params.batch_size, params.k, temperature);
        });
        test_algo("MAP_REDUCE (p=256)", [&](float* s_d, int* i_d){
            Generators::cuda::RunTopKViaMapReduce(sampling_data.get(), stream, scores_in_d.get(), s_d, i_d, params.vocab_size, params.batch_size, params.k, temperature, 256);
        });
        test_algo("MAP_REDUCE_SHARED (p=64)", [&](float* s_d, int* i_d){
            Generators::cuda::RunTopKViaMapReduceShared(sampling_data.get(), stream, scores_in_d.get(), s_d, i_d, params.vocab_size, params.batch_size, params.k, temperature, 64);
        });
        if (params.batch_size == 1) {
            test_algo("SINGLE_KERNEL_MAP_REDUCE (p=256)", [&](float* s_d, int* i_d){
                Generators::cuda::RunTopKViaSingleKernelMapReduce(sampling_data.get(), stream, scores_in_d.get(), s_d, i_d, params.vocab_size, params.batch_size, params.k, temperature, 256);
            });
        }
    }

    CudaCheck(cudaStreamDestroy(stream));
}

// Main benchmark function
void RunBenchmarks() {
    // --- Define Benchmark Configurations ---
    std::vector<int> batch_sizes = {1, 4, 8};
    std::vector<int> vocab_sizes = {200000, 20000, 40000, 100000, 300000};
    std::vector<int> ks = {50, 1, 4, 8, 32, 64};

    // By default, only test the first combination. Change it to True to test all combinations.
    constexpr bool all_combinations = false;
    if (!all_combinations) {
        batch_sizes.resize(1);
        vocab_sizes.resize(1);
        ks.resize(1);
    }

    constexpr int warmup_runs = 5;
    constexpr int timing_runs = 100;
    constexpr float temperature = 0.9f;

    std::vector<BenchmarkParams> configs;
    for (int batch_size : batch_sizes) {
        for (int vocab_size : vocab_sizes) {
            for (int k : ks) {
                configs.push_back({batch_size, vocab_size, k});
            }
        }
    }

    std::vector<BenchmarkResult> all_results;

    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));

    for (const auto& params : configs) {
        std::cout << "\nRunning benchmark for: batch_size=" << params.batch_size
                  << ", vocab_size=" << params.vocab_size
                  << ", k=" << params.k << "..." << std::endl;

        // --- Setup ---
        unsigned long long random_seed = 1234;
        auto sampling_data = std::make_unique<Generators::cuda::SamplingData>(random_seed, params.batch_size, params.vocab_size, stream);

        auto scores_in = Generators::CudaMallocArray<float>(params.batch_size * params.vocab_size);
        auto scores_out = Generators::CudaMallocArray<float>(params.batch_size * params.k);
        auto indices_out = Generators::CudaMallocArray<int>(params.batch_size * params.k);

        cudaEvent_t start, stop;
        CudaCheck(cudaEventCreate(&start));
        CudaCheck(cudaEventCreate(&stop));

        const int total_size = params.batch_size * params.vocab_size;

        auto measure_latency = [&](const std::string& name, int num_partitions, auto func) {
            // Warmup
            for (int i = 0; i < warmup_runs; ++i) {
                // Regenerate data for each warmup run as well to ensure caches are not misleading
                Generators::cuda::RandomTopkInput(stream, scores_in.get(), sampling_data->curand_states.get(), total_size, params.batch_size);
                func();
            }
            CudaCheck(cudaStreamSynchronize(stream));

            // Timing
            CudaCheck(cudaEventRecord(start, stream));
            for (int i = 0; i < timing_runs; ++i) {
                // Regenerate random data before each timed run to bust caches
                Generators::cuda::RandomTopkInput(stream, scores_in.get(), sampling_data->curand_states.get(), total_size, params.batch_size);
                func();
            }
            CudaCheck(cudaEventRecord(stop, stream));
            CudaCheck(cudaEventSynchronize(stop));

            float ms = 0.0f;
            CudaCheck(cudaEventElapsedTime(&ms, start, stop));
            all_results.push_back({params, name, num_partitions, ms / timing_runs});
        };

        // --- Run Benchmarks for each algorithm ---

        if (params.k <= 64) {
            measure_latency("DIRECT_KERNEL", 0, [&]() {
                Generators::cuda::RunTopKViaDirectKernel(sampling_data.get(), stream, scores_in.get(), scores_out.get(), indices_out.get(), params.vocab_size, params.batch_size, params.k, temperature);
            });

            for (int num_partitions : {64, 128, 256}) {
                measure_latency("MAP_REDUCE", num_partitions, [&]() {
                    Generators::cuda::RunTopKViaMapReduce(sampling_data.get(), stream, scores_in.get(), scores_out.get(), indices_out.get(), params.vocab_size, params.batch_size, params.k, temperature, num_partitions);
                });
            }

            for (int num_partitions : {32, 64}) {
                 size_t shared_mem_size = num_partitions * 64 * (sizeof(float) + sizeof(int));
                 if (shared_mem_size < 48 * 1024) { // Check against common shared memory limit
                    measure_latency("MAP_REDUCE_SHARED", num_partitions, [&]() {
                        Generators::cuda::RunTopKViaMapReduceShared(sampling_data.get(), stream, scores_in.get(), scores_out.get(), indices_out.get(), params.vocab_size, params.batch_size, params.k, temperature, num_partitions);
                    });
                 }
            }
            
            if (params.batch_size == 1) {
                for (int num_partitions : {64, 128, 256}) {
                    measure_latency("SINGLE_KERNEL_MAP_REDUCE", num_partitions, [&]() {
                        Generators::cuda::RunTopKViaSingleKernelMapReduce(sampling_data.get(), stream, scores_in.get(), scores_out.get(), indices_out.get(), params.vocab_size, params.batch_size, params.k, temperature, num_partitions);
                    });
                }
            }
        }

        measure_latency("FULL_SORT", 0, [&]() {
            Generators::cuda::RunTopKViaFullSort(sampling_data.get(), stream, scores_in.get(), scores_out.get(), indices_out.get(), params.vocab_size, params.batch_size, params.k, temperature);
        });

        CudaCheck(cudaEventDestroy(start));
        CudaCheck(cudaEventDestroy(stop));
    }

    CudaCheck(cudaStreamDestroy(stream));

    // --- Print Results ---
    std::cout << "\n--- Benchmark Results ---\n";
    std::cout << std::left << std::setw(12) << "Batch Size"
              << std::setw(12) << "Vocab Size"
              << std::setw(5) << "K"
              << std::setw(28) << "Algorithm"
              << std::setw(14) << "Partitions"
              << "Latency (ms)\n";
    std::cout << std::string(85, '-') << "\n";

    for (const auto& result : all_results) {
        std::cout << std::left << std::setw(12) << result.params.batch_size
                  << std::setw(12) << result.params.vocab_size
                  << std::setw(5) << result.params.k
                  << std::setw(28) << result.algo_name
                  << std::setw(14) << (result.num_partitions > 0 ? std::to_string(result.num_partitions) : "N/A")
                  << std::fixed << std::setprecision(4) << result.latency_ms << "\n";
    }
}

TEST(TopKTests, ParityTests) {
    RunParityTests();
}

TEST(TopKTests, BenchmarkTests) {
    RunBenchmarks();
}
