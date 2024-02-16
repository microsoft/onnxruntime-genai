// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "smartptrs.h"

namespace Generators {
namespace cuda {

struct SamplingData {
  SamplingData(int batch_size, int vocab_size, cudaStream_t stream);
  std::unique_ptr<int, Generators::CudaDeleter> indices_sorted = nullptr;
  std::unique_ptr<float, Generators::CudaDeleter> scores_sorted = nullptr;
  std::unique_ptr<float, Generators::CudaDeleter> scores_softmaxed = nullptr;
  std::unique_ptr<float, Generators::CudaDeleter> prefix_sums = nullptr;
  std::unique_ptr<float, Generators::CudaDeleter> thresholds = nullptr;
  std::unique_ptr<int, Generators::CudaDeleter> indices_in = nullptr;
  std::unique_ptr<int, Generators::CudaDeleter> offsets = nullptr;
  std::unique_ptr<float, Generators::CudaDeleter> temp_buffer = nullptr;
  size_t temp_storage_bytes = 0;
};

void LaunchPopulateIndices(int* indices, int size, int batch_size, cudaStream_t stream);
void GetTopKSubset(cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature=1.0f);
void GetSample(SamplingData* data, cudaStream_t stream, int32_t* d_next_token, float* d_scores, int vocab_size, int batch_size, int k, float p, float temperature);

}  // namespace cuda
}  // namespace Generators