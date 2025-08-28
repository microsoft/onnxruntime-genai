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
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>
#include <limits>

#include "cuda_runtime.h"
#include "../src/span.h"
#include "models/onnxruntime_api.h"
#include "../src/cuda/cuda_sampling.cuh"
#include "smartptrs.h" // For CudaMallocArray
#include <gtest/gtest.h>

// Robust CUDA error checking macro
#define CUDA_CHECK(call)                                         \
do {                                                             \
    cudaError_t err = call;                                      \
    if (err != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",             \
                cudaGetErrorString(err), __FILE__, __LINE__);    \
        exit(EXIT_FAILURE);                                      \
    }                                                            \
} while (0)


// Forward declarations of the internal functions we want to benchmark,
// as they are not in the .cuh header.
namespace Generators {
namespace cuda {

void RunTopKViaDirectKernel(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature);
void RunTopKViaMapReduce(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions);
void RunTopKViaMapReduceShared(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions);
void RunTopKViaSingleKernelMapReduce(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions);
void RunTopKViaFullSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature);
void RunTopKViaMapReduceVec(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions, int block_size);
void RunTopKViaMapReduceHybridSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions, int block_size);
void RunTopKViaMapReduceBitonicSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions);
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
    int block_size;
    float latency_ms;
};

// Global mutex to serialize benchmark tests and prevent parallel execution on the GPU
static std::mutex benchmark_mutex;

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
                      << ". Expected: " << std::fixed << std::setprecision(6) << reference_scores[i] 
                      << ", Got: " << actual_scores[i] << std::endl;
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
void RunParityTests(const BenchmarkParams &params, float temperature) {
    std::cout << "\n--- Running Parity Tests with batch_size=" 
              << params.batch_size << ", vocab_size=" << params.vocab_size <<", k=" << params.k << ", temperature=" << temperature
              << "---\n";

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

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
    CUDA_CHECK(cudaMemcpy(scores_in_d.get(), scores_in_h.data(), scores_in_h.size() * sizeof(float), cudaMemcpyHostToDevice));

    // --- Get Reference Result using Full Sort ---
    auto ref_scores_d = Generators::CudaMallocArray<float>(params.batch_size * params.k);
    auto ref_indices_d = Generators::CudaMallocArray<int>(params.batch_size * params.k);
    Generators::cuda::RunTopKViaFullSort(sampling_data.get(), stream, scores_in_d.get(), ref_scores_d.get(), ref_indices_d.get(), params.vocab_size, params.batch_size, params.k, temperature);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> ref_scores_h(params.batch_size * params.k);
    std::vector<int> ref_indices_h(params.batch_size * params.k);
    CUDA_CHECK(cudaMemcpy(ref_scores_h.data(), ref_scores_d.get(), ref_scores_h.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref_indices_h.data(), ref_indices_d.get(), ref_indices_h.size() * sizeof(int), cudaMemcpyDeviceToHost));

    // --- Test Other Algorithms ---
    auto test_algo = [&](const std::string& name, auto func) {
        auto actual_scores_d = Generators::CudaMallocArray<float>(params.batch_size * params.k);
        auto actual_indices_d = Generators::CudaMallocArray<int>(params.batch_size * params.k);
        func(actual_scores_d.get(), actual_indices_d.get());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<float> actual_scores_h(params.batch_size * params.k);
        std::vector<int> actual_indices_h(params.batch_size * params.k);
        CUDA_CHECK(cudaMemcpy(actual_scores_h.data(), actual_scores_d.get(), actual_scores_h.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(actual_indices_h.data(), actual_indices_d.get(), actual_indices_h.size() * sizeof(int), cudaMemcpyDeviceToHost));

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
        test_algo("MAP_REDUCE_VEC (p=256, b=256)", [&](float* s_d, int* i_d){
            Generators::cuda::RunTopKViaMapReduceVec(sampling_data.get(), stream, scores_in_d.get(), s_d, i_d, params.vocab_size, params.batch_size, params.k, temperature, 256, 256);
        });
        test_algo("MAP_REDUCE_HYBRID_SORT (p=64, b=256)", [&](float* s_d, int* i_d){
            Generators::cuda::RunTopKViaMapReduceHybridSort(sampling_data.get(), stream, scores_in_d.get(), s_d, i_d, params.vocab_size, params.batch_size, params.k, temperature, 64, 256);
        });
        test_algo("MAP_REDUCE_BITONIC_SORT (p=128)", [&](float* s_d, int* i_d){
            Generators::cuda::RunTopKViaMapReduceBitonicSort(sampling_data.get(), stream, scores_in_d.get(), s_d, i_d, params.vocab_size, params.batch_size, params.k, temperature, 128);
        });
        if (params.batch_size == 1) {
            test_algo("SINGLE_KERNEL_MAP_REDUCE (p=256)", [&](float* s_d, int* i_d){
                Generators::cuda::RunTopKViaSingleKernelMapReduce(sampling_data.get(), stream, scores_in_d.get(), s_d, i_d, params.vocab_size, params.batch_size, params.k, temperature, 256);
            });
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
}

// New function to print the summary in CSV format
void PrintSummary(const std::vector<BenchmarkResult>& all_results) {
    // This struct holds the aggregated data for a unique key (batch_size, vocab_size, k)
    struct SummaryData {
        std::string best_algo_full_name;
        float min_latency = std::numeric_limits<float>::max();
        float latency_full_sort = -1.0f;
        float latency_direct_kernel = -1.0f;
    };

    // Use a map to group results by their parameters
    std::map<std::tuple<int, int, int>, SummaryData> summary_map;

    // Iterate over all detailed results to populate the summary map
    for (const auto& result : all_results) {
        auto key = std::make_tuple(result.params.batch_size, result.params.vocab_size, result.params.k);
        
        // Create a more descriptive name for map-reduce algorithms
        std::string full_algo_name = result.algo_name;
        if (result.num_partitions > 0) {
            full_algo_name += " (p=" + std::to_string(result.num_partitions) + ")";
        }

        // Check if this is the best performing algorithm for this key so far
        if (result.latency_ms < summary_map[key].min_latency) {
            summary_map[key].min_latency = result.latency_ms;
            summary_map[key].best_algo_full_name = full_algo_name;
        }

        // Store the latencies for the specific baseline algorithms
        if (result.algo_name == "FULL_SORT") {
            summary_map[key].latency_full_sort = result.latency_ms;
        } else if (result.algo_name == "DIRECT_KERNEL") {
            summary_map[key].latency_direct_kernel = result.latency_ms;
        }
    }

    // Print the CSV header
    std::cout << "\n--- Benchmark Summary (CSV) ---\n";
    std::cout << "batch_size,vocab_size,k,best_algo,min_latency,latency_of_baseline_1(full_sort),latency_of_baseline_2(direct_kernel),ratio_1,ratio_2\n";

    // Print each row of the summary table
    for (const auto& pair : summary_map) {
        const auto& key = pair.first;
        const auto& data = pair.second;

        // Calculate performance ratios against the baselines
        float ratio_1 = (data.min_latency > 0 && data.latency_full_sort > 0) ? data.latency_full_sort / data.min_latency : 0.0f;
        float ratio_2 = (data.min_latency > 0 && data.latency_direct_kernel > 0) ? data.latency_direct_kernel / data.min_latency : 0.0f;

        std::cout << std::get<0>(key) << ","
                  << std::get<1>(key) << ","
                  << std::get<2>(key) << ","
                  << "\"" << data.best_algo_full_name << "\"," // Quote algorithm name
                  << std::fixed << std::setprecision(4) << data.min_latency << ","
                  << (data.latency_full_sort > 0 ? std::to_string(data.latency_full_sort) : "N/A") << ","
                  << (data.latency_direct_kernel > 0 ? std::to_string(data.latency_direct_kernel) : "N/A") << ","
                  << std::fixed << std::setprecision(2) << ratio_1 << ","
                  << ratio_2 << "\n";
    }
}


// Main benchmark function
void RunBenchmarks() {
    // --- Define Benchmark Configurations ---
    std::vector<int> batch_sizes = {1, 4, 16};
    std::vector<int> vocab_sizes = {201088, 32000, 128256}; // GPT-OSS 201088, LLAMA2 32000, LLAMA3 128256, DeepSeek 102400, QWen3 151646
    std::vector<int> ks = {50, 1, 2, 4, 8, 16, 32, 64};

    // By default, only test the first combination. Change it to True to test all combinations.
    constexpr bool all_combinations = true;
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
    CUDA_CHECK(cudaStreamCreate(&stream));

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
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        const int total_size = params.batch_size * params.vocab_size;

        auto measure_latency = [&](const std::string& name, int num_partitions, int block_size, auto func) {
            // Warmup
            for (int i = 0; i < warmup_runs; ++i) {
                // Regenerate data for each warmup run as well to ensure caches are not misleading
                Generators::cuda::RandomTopkInput(stream, scores_in.get(), sampling_data->curand_states.get(), total_size, params.batch_size);
                func();
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Timing
            float total_ms = 0.0f;
            for (int i = 0; i < timing_runs; ++i) {
                // Regenerate random data before each timed run to bust caches
                Generators::cuda::RandomTopkInput(stream, scores_in.get(), sampling_data->curand_states.get(), total_size, params.batch_size);

                CUDA_CHECK(cudaEventRecord(start, stream));
                func();
                CUDA_CHECK(cudaEventRecord(stop, stream));
                
                CUDA_CHECK(cudaEventSynchronize(stop));
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                total_ms += ms;
            }
            all_results.push_back({params, name, num_partitions, block_size, total_ms / timing_runs});
        };

        // --- Run Benchmarks for each algorithm ---

        if (params.k <= 64) {
            measure_latency("DIRECT_KERNEL", 0, 256, [&]() {
                Generators::cuda::RunTopKViaDirectKernel(sampling_data.get(), stream, scores_in.get(), scores_out.get(), indices_out.get(), params.vocab_size, params.batch_size, params.k, temperature);
            });

            for (int num_partitions : {64, 128, 256}) {
                measure_latency("MAP_REDUCE", num_partitions, 256, [&]() {
                    Generators::cuda::RunTopKViaMapReduce(sampling_data.get(), stream, scores_in.get(), scores_out.get(), indices_out.get(), params.vocab_size, params.batch_size, params.k, temperature, num_partitions);
                });
            }

            for (int num_partitions : {32, 64}) {
                 size_t shared_mem_size = num_partitions * 64 * (sizeof(float) + sizeof(int));
                 if (shared_mem_size < 48 * 1024) { // Check against common shared memory limit
                    measure_latency("MAP_REDUCE_SHARED", num_partitions, 256, [&]() {
                        Generators::cuda::RunTopKViaMapReduceShared(sampling_data.get(), stream, scores_in.get(), scores_out.get(), indices_out.get(), params.vocab_size, params.batch_size, params.k, temperature, num_partitions);
                    });
                 }
            }

            for (int num_partitions : {64, 128, 256}) {
                measure_latency("MAP_REDUCE_VEC", num_partitions, 256, [&]() {
                    Generators::cuda::RunTopKViaMapReduceVec(sampling_data.get(), stream, scores_in.get(), scores_out.get(), indices_out.get(), params.vocab_size, params.batch_size, params.k, temperature, num_partitions, 256);
                });
            }

            for (int num_partitions : {64, 128}) {
                measure_latency("MAP_REDUCE_HYBRID_SORT", num_partitions, 256, [&]() {
                    Generators::cuda::RunTopKViaMapReduceHybridSort(sampling_data.get(), stream, scores_in.get(), scores_out.get(), indices_out.get(), params.vocab_size, params.batch_size, params.k, temperature, num_partitions, 256);
                });
            }
            
            for (int num_partitions : {64, 128}) {
                measure_latency("MAP_REDUCE_BITONIC_SORT", num_partitions, 256, [&]() {
                    Generators::cuda::RunTopKViaMapReduceBitonicSort(sampling_data.get(), stream, scores_in.get(), scores_out.get(), indices_out.get(), params.vocab_size, params.batch_size, params.k, temperature, num_partitions);
                });
            }

            if (params.batch_size == 1) {
                for (int num_partitions : {64, 128, 256}) {
                    measure_latency("SINGLE_KERNEL_MAP_REDUCE", num_partitions, 256, [&]() {
                        Generators::cuda::RunTopKViaSingleKernelMapReduce(sampling_data.get(), stream, scores_in.get(), scores_out.get(), indices_out.get(), params.vocab_size, params.batch_size, params.k, temperature, num_partitions);
                    });
                }
            }
        }

        measure_latency("FULL_SORT", 0, 256, [&]() {
            Generators::cuda::RunTopKViaFullSort(sampling_data.get(), stream, scores_in.get(), scores_out.get(), indices_out.get(), params.vocab_size, params.batch_size, params.k, temperature);
        });

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    CUDA_CHECK(cudaStreamDestroy(stream));

    // --- Print Detailed Results ---
    std::cout << "\n--- Benchmark Results ---\n";
    std::cout << std::left << std::setw(12) << "Batch Size"
              << std::setw(12) << "Vocab Size"
              << std::setw(5) << "K"
              << std::setw(28) << "Algorithm"
              << std::setw(14) << "Partitions"
              << std::setw(12) << "Block Size"
              << "Latency (ms)\n";
    std::cout << std::string(97, '-') << "\n";

    for (const auto& result : all_results) {
        std::cout << std::left << std::setw(12) << result.params.batch_size
                  << std::setw(12) << result.params.vocab_size
                  << std::setw(5) << result.params.k
                  << std::setw(28) << result.algo_name
                  << std::setw(14) << (result.num_partitions > 0 ? std::to_string(result.num_partitions) : "N/A")
                  << std::setw(12) << result.block_size
                  << std::fixed << std::setprecision(4) << result.latency_ms << "\n";
    }

    // --- Print CSV Summary ---
    PrintSummary(all_results);
}

TEST(TopKTests, ParityTests) {
    std::lock_guard<std::mutex> lock(benchmark_mutex);

    std::vector<int> batch_sizes = {1, 2};
    std::vector<int> vocab_sizes = {10000, 204800};
    std::vector<int> ks = {1, 50, 64};
    std::vector<float> temperatures = {1.0f, 0.5f};

    for (int batch_size : batch_sizes) {
        for (int vocab_size : vocab_sizes) {
            for (int k: ks) {
                for (float temperature : temperatures) {
                    BenchmarkParams params = {batch_size, vocab_size, k};
                    RunParityTests(params, temperature);
                }
            }
        }
    }
}

TEST(TopKTests, BenchmarkTests) {
    std::lock_guard<std::mutex> lock(benchmark_mutex);
    RunBenchmarks();
}
