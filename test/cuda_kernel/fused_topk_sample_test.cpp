// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if USE_CUDA
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "../../src/cuda/cuda_common.h"
#include "../../src/cuda/cuda_sampling.h"

namespace Generators::cuda::test {

TEST(FusedTopKSampleTest, ExternalUniformEmitsSparseDistribution) {
  constexpr int batch_size = 2;
  constexpr int vocab_size = 5;
  constexpr int k = 3;
  constexpr float p = 0.8f;
  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto sampling_data = std::make_unique<SamplingData>(0, batch_size, vocab_size, stream);
  auto d_scores = CudaMallocArray<float>(batch_size * vocab_size);
  auto d_uniforms = CudaMallocArray<float>(batch_size);
  auto d_tokens = CudaMallocArray<int64_t>(batch_size);
  auto d_indices = CudaMallocArray<int64_t>(batch_size * k);
  auto d_probs = CudaMallocArray<float>(batch_size * k);
  const std::vector<float> scores{5.0f, 4.0f, 3.0f, 0.0f, -1.0f,
                                  0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> uniforms{0.0f, 0.99f};
  CUDA_CHECK(cudaMemcpyAsync(d_scores.get(), scores.data(), scores.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_uniforms.get(), uniforms.data(), uniforms.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream));

  RunTopK(sampling_data.get(), stream, d_scores.get(), vocab_size, batch_size, k);
  LaunchFusedSampleKernelWithOutput(
      sampling_data.get(), stream, sampling_data->topk_scores, sampling_data->topk_indices,
      d_uniforms.get(), d_tokens.get(), d_indices.get(), d_probs.get(), k, batch_size, p,
      1.0f, sampling_data->topk_stride);

  std::vector<int64_t> tokens(batch_size);
  std::vector<int64_t> indices(batch_size * k);
  std::vector<float> probs(batch_size * k);
  CUDA_CHECK(cudaMemcpyAsync(tokens.data(), d_tokens.get(), tokens.size() * sizeof(int64_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(indices.data(), d_indices.get(), indices.size() * sizeof(int64_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(probs.data(), d_probs.get(), probs.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));

  EXPECT_EQ(indices, (std::vector<int64_t>{0, 1, 2, 4, 3, 2}));
  EXPECT_EQ(tokens, (std::vector<int64_t>{0, 3}));
  const float denominator = std::exp(1.0f) + 1.0f;
  for (int row = 0; row < batch_size; ++row) {
    EXPECT_NEAR(probs[row * k], std::exp(1.0f) / denominator, 1e-6f);
    EXPECT_NEAR(probs[row * k + 1], 1.0f / denominator, 1e-6f);
    EXPECT_EQ(probs[row * k + 2], 0.0f);
  }
}

}  // namespace Generators::cuda::test
#endif