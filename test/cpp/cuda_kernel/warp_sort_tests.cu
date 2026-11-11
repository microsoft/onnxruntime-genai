// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cuda_topk_warp_sort_helper.cuh"

// Correct output alone does not rule out shared-memory races. Run both standalone targets with:
// CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool racecheck --error-exitcode 99 \
//   <build>/cuda_warp_sort_tests --gtest_filter='*Storage/Size64Valid64Threads128Unique'
// Repeat with cuda_warp_sort_stable_tests to cover STABLE_TOPK.

namespace Generators::cuda::test {
namespace {

template <int BufferSize>
struct SortBuffers {
  alignas(128) float scores[BufferSize];
  alignas(128) int indices[BufferSize];
};

template <int BufferSize>
using SortTempStorage = typename cub::WarpMergeSort<
    uint64_t, BufferSize / topk_common::kThreadsInWarp, topk_common::kThreadsInWarp, cub::NullType>::TempStorage;

// Match production's keys-only scratch capacity/alignment even in the float/int default path.
// Current production dispatch uses WarpMergeSort<64>; larger sizes cover the helper's full range.
template <int BufferSize>
union AliasedStorage {
  SortBuffers<BufferSize> buffers;
  SortTempStorage<BufferSize> scratch;
};

// Negative control: scratch cannot overlap score/index writeback.
template <int BufferSize>
struct SeparateStorage {
  SortBuffers<BufferSize> buffers;
  SortTempStorage<BufferSize> scratch;
};

template <int BufferSize, bool Aliased>
__global__ void SortKernel(const float* input_scores, const int* input_indices,
                           float* output_scores, int* output_indices, int valid_count) {
  using Storage = std::conditional_t<Aliased, AliasedStorage<BufferSize>, SeparateStorage<BufferSize>>;
  static_assert(alignof(Storage) == 128);
  static_assert(offsetof(SortBuffers<BufferSize>, scores) == 0);
  static_assert(offsetof(SortBuffers<BufferSize>, indices) == sizeof(float) * BufferSize);
  __shared__ Storage storage;
  for (int i = threadIdx.x; i < BufferSize; i += blockDim.x) {
    storage.buffers.scores[i] = input_scores[i];
    storage.buffers.indices[i] = input_indices[i];
  }
  __syncthreads();

  // Call the real helper from the entire block, including the warps that return immediately.
  topk_common::WarpMergeSort<BufferSize>(
      storage.buffers.scores, storage.buffers.indices, &storage.scratch, valid_count);

  __syncthreads();
  for (int i = threadIdx.x; i < BufferSize; i += blockDim.x) {
    output_scores[i] = storage.buffers.scores[i];
    output_indices[i] = storage.buffers.indices[i];
  }
}

struct SortCase {
  int size;
  int valid_count;
  int threads;
  bool ties;
};

std::vector<SortCase> MakeCases() {
  std::vector<SortCase> cases;
  for (int size : {32, 64, 128, 256}) {
    for (int valid_count : {0, 1, size / 2 + 1, size - 1, size}) {
      for (int threads : {32, 128}) {
        for (bool ties : {false, true}) {
          cases.push_back({size, valid_count, threads, ties});
        }
      }
    }
  }
  return cases;
}

class WarpMergeSortTest : public ::testing::TestWithParam<SortCase> {
 protected:
  void SetUp() override {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }
    stream_.Create();
  }

  template <int BufferSize, bool Aliased>
  void Run() {
    const auto& param = GetParam();
    std::vector<float> input_scores(BufferSize);
    std::vector<int> input_indices(BufferSize);
    std::vector<float> scores_by_index(BufferSize);
    std::vector<std::pair<float, int>> expected;
    for (int i = 0; i < BufferSize; ++i) {
      // Odd multipliers permute power-of-two sizes. Both positive and negative scores are present.
      const int rank = (73 * i + 19) % BufferSize;
      input_scores[i] = static_cast<float>(param.ties ? rank / 4 - BufferSize / 8 : rank - BufferSize / 2) + 0.25f;
      input_indices[i] = (37 * i + 11) % BufferSize;
      scores_by_index[input_indices[i]] = input_scores[i];
      if (i < param.valid_count) {
        expected.emplace_back(input_scores[i], input_indices[i]);
      }
    }
    std::sort(expected.begin(), expected.end(), [](const auto& a, const auto& b) {
      return a.first > b.first || (a.first == b.first && a.second < b.second);
    });

    auto device_input_scores = CudaMallocArray<float>(BufferSize);
    auto device_input_indices = CudaMallocArray<int>(BufferSize);
    auto device_output_scores = CudaMallocArray<float>(BufferSize);
    auto device_output_indices = CudaMallocArray<int>(BufferSize);
    CUDA_CHECK(cudaMemcpyAsync(device_input_scores.get(), input_scores.data(), sizeof(float) * BufferSize,
                               cudaMemcpyHostToDevice, stream_.get()));
    CUDA_CHECK(cudaMemcpyAsync(device_input_indices.get(), input_indices.data(), sizeof(int) * BufferSize,
                               cudaMemcpyHostToDevice, stream_.get()));
    SortKernel<BufferSize, Aliased><<<1, param.threads, 0, stream_.get()>>>(
        device_input_scores.get(), device_input_indices.get(), device_output_scores.get(), device_output_indices.get(),
        param.valid_count);
    CUDA_CHECK_LAUNCH();

    std::vector<float> output_scores(BufferSize);
    std::vector<int> output_indices(BufferSize);
    CUDA_CHECK(cudaMemcpyAsync(output_scores.data(), device_output_scores.get(), sizeof(float) * BufferSize,
                               cudaMemcpyDeviceToHost, stream_.get()));
    CUDA_CHECK(cudaMemcpyAsync(output_indices.data(), device_output_indices.get(), sizeof(int) * BufferSize,
                               cudaMemcpyDeviceToHost, stream_.get()));
    CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    std::vector<bool> seen(BufferSize, false);
    for (int i = 0; i < BufferSize; ++i) {
      SCOPED_TRACE(i);
      if (i >= param.valid_count) {
        EXPECT_EQ(output_scores[i], -FLT_MAX);
        EXPECT_EQ(output_indices[i], INT_MAX);
        continue;
      }
      EXPECT_EQ(output_scores[i], expected[i].first);
      ASSERT_GE(output_indices[i], 0);
      ASSERT_LT(output_indices[i], BufferSize);
      EXPECT_EQ(output_scores[i], scores_by_index[output_indices[i]]);
      EXPECT_FALSE(seen[output_indices[i]]);
      seen[output_indices[i]] = true;
      // Returned indices must belong to the valid prefix, not just have a matching score.
      EXPECT_NE(std::find(input_indices.begin(), input_indices.begin() + param.valid_count, output_indices[i]),
                input_indices.begin() + param.valid_count);
#ifdef STABLE_TOPK
      EXPECT_EQ(output_indices[i], expected[i].second);
#endif
    }
  }

  template <bool Aliased>
  void Dispatch() {
    switch (GetParam().size) {
      case 32:
        Run<32, Aliased>();
        break;
      case 64:
        Run<64, Aliased>();
        break;
      case 128:
        Run<128, Aliased>();
        break;
      case 256:
        Run<256, Aliased>();
        break;
      default:
        FAIL() << "Unsupported buffer size";
    }
  }

  cuda_stream_holder stream_;
};

TEST_P(WarpMergeSortTest, AliasedStorage) {
  Dispatch<true>();
}

TEST_P(WarpMergeSortTest, SeparateStorage) {
  Dispatch<false>();
}

INSTANTIATE_TEST_SUITE_P(
    Cases, WarpMergeSortTest, ::testing::ValuesIn(MakeCases()),
    [](const ::testing::TestParamInfo<SortCase>& info) {
      const auto& p = info.param;
      return "Size" + std::to_string(p.size) + "Valid" + std::to_string(p.valid_count) +
             "Threads" + std::to_string(p.threads) + (p.ties ? "Ties" : "Unique");
    });

}  // namespace
}  // namespace Generators::cuda::test
