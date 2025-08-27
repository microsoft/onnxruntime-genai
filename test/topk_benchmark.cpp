// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is for macro-benchmarking the GetTopKSubset function.
// To build this, add it as a new executable in your CMakeLists.txt,
// and enable it with a build flag, e.g., -DENABLE_SAMPLING_BENCHMARK=ON.
// Example CMake entry:
// if(ENABLE_SAMPLING_BENCHMARK)
//   add_executable(sampling_benchmark sampling_benchmark.cpp)
//   target_link_libraries(sampling_benchmark PRIVATE ${PROJECT_NAME})
// endif()

#ifdef ENABLE_SAMPLING_BENCHMARK

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <random>
#include <chrono>
#include <memory>

#include "cuda_runtime.h"
#include "cuda_sampling.cuh"
#include "smartptrs.h" // For CudaMallocArray

// Forward declarations of the internal functions we want to benchmark,
// as they are not in the .cuh header.
namespace Generators {
namespace cuda {

void RunTopKViaDirectKernel(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature);
void RunTopKViaMapReduce(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions);
void RunTopKViaMapReduceShared(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions);
void RunTopKViaSingleKernelMapReduce(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions);
void RunTopKViaFullSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature);

__global__ void FillRandom(float* array, curandState* states, int n, int batch_size);

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

// Main benchmark function
void RunBenchmarks() {
    // --- Define Benchmark Configurations ---
    std::vector<BenchmarkParams> configs = {
        {1, 50257, 10},
        {1, 50257, 50},
        {1, 200000, 50},
        {4, 50257, 50},
        {1, 32000, 64},
        {8, 32000, 64}
    };

    std::vector<BenchmarkResult> all_results;

    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));

    for (const auto& params : configs) {
        std::cout << "Running benchmark for: batch_size=" << params.batch_size
                  << ", vocab_size=" << params.vocab_size
                  << ", k=" << params.k << "..." << std::endl;

        // --- Setup ---
        unsigned long long random_seed = 1234;
        auto sampling_data = std::make_unique<Generators::cuda::SamplingData>(random_seed, params.batch_size, params.vocab_size, stream);

        auto scores_in = Generators::CudaMallocArray<float>(params.batch_size * params.vocab_size);
        auto scores_out = Generators::CudaMallocArray<float>(params.batch_size * params.k);
        auto indices_out = Generators::CudaMallocArray<int>(params.batch_size * params.k);

        // Fill input with random data on the GPU
        int total_size = params.batch_size * params.vocab_size;
        Generators::cuda::FillRandom<<<(total_size + 255) / 256, 256, 0, stream>>>(scores_in.get(), sampling_data->curand_states.get(), total_size, params.batch_size);

        cudaEvent_t start, stop;
        CudaCheck(cudaEventCreate(&start));
        CudaCheck(cudaEventCreate(&stop));

        const int warmup_runs = 5;
        const int timing_runs = 50;
        const float temperature = 1.0f;

        auto measure_latency = [&](const std::string& name, int num_partitions, auto func) {
            // Warmup
            for (int i = 0; i < warmup_runs; ++i) {
                func();
            }
            CudaCheck(cudaStreamSynchronize(stream));

            // Timing
            CudaCheck(cudaEventRecord(start, stream));
            for (int i = 0; i < timing_runs; ++i) {
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

int main() {
    RunBenchmarks();
    return 0;
}

#endif // ENABLE_SAMPLING_BENCHMARK
