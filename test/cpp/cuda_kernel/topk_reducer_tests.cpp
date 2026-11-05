// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include <gtest/gtest.h>

#include "ep/cuda/cuda_topk.h"

namespace {

class DistributedTopKReducerTest : public ::testing::TestWithParam<int> {};

TEST_P(DistributedTopKReducerTest, DoesNotReadPreviousPartialResults) {
  constexpr int vocab_size = 248320;
  constexpr int max_k = Generators::cuda::topk_impl_details::kTopKDistributedSelectSortMaxTopK;
  const int k = GetParam();
  Generators::cuda_stream_holder stream;
  stream.Create();
  auto device_scores = Generators::CudaMallocArray<float>(vocab_size);
  std::vector<float> scores(vocab_size, -100.0f);
  std::vector<int> expected_indices{32645, 1049};
  scores[32645] = 23.59375f;
  scores[1049] = 22.703125f;
  scores[32471] = 5.796875f;
  for (int i = 0; i < max_k - 2; ++i) {
    expected_indices.push_back(100 + i);
    scores[100 + i] = 22.0f - i * 0.25f;
  }
  CUDA_CHECK(cudaMemcpyAsync(device_scores.get(), scores.data(), scores.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream));

  Generators::cuda::TopkData data(1, vocab_size, stream);
  const size_t partial_count = static_cast<size_t>(data.top_k_distributed_select_sort_shards) * k;
  // Make stale partials beat every current score so an early read cannot pass unnoticed.
  std::vector<float> previous_scores(partial_count, 1000.0f);
  std::vector<int> previous_indices(partial_count, 42);
  std::vector<int> indices(k);
  std::vector<float> selected_scores(k);
  for (int repeat = 0; repeat < 4096; ++repeat) {
    CUDA_CHECK(cudaMemcpyAsync(data.top_k_distributed_select_sort_values, previous_scores.data(),
                               previous_scores.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(data.top_k_distributed_select_sort_keys, previous_indices.data(),
                               previous_indices.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    Generators::cuda::distributed_select_sort::RunTopK(
        &data, stream, device_scores.get(), vocab_size, 1, k);
    CUDA_CHECK(cudaMemcpyAsync(indices.data(), data.topk_indices, indices.size() * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(selected_scores.data(), data.topk_scores, selected_scores.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < k; ++i) {
      ASSERT_EQ(indices[i], expected_indices[i]) << "repeat=" << repeat << ", rank=" << i;
      ASSERT_EQ(selected_scores[i], scores[expected_indices[i]]) << "repeat=" << repeat << ", rank=" << i;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(TopK, DistributedTopKReducerTest,
                         ::testing::Values(1, 4, 20, Generators::cuda::topk_impl_details::kTopKDistributedSelectSortMaxTopK));

}  // namespace
