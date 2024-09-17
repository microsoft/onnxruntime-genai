// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <limits>

namespace Generators {
namespace cuda {

template <typename T>
__global__ void UpdatePositionIds(T* positions, int batch_beam_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch_beam_size)
    positions[i]++;
}

template <typename T>
void Launch_UpdatePositionIds(T* positions, int batch_beam_size, cudaStream_t stream) {
  UpdatePositionIds<T><<<(batch_beam_size + 255) / 256, 256, 0, stream>>>(positions, batch_beam_size);
}

template void Launch_UpdatePositionIds(int32_t* positions, int batch_beam_size, cudaStream_t stream);
template void Launch_UpdatePositionIds(int64_t* positions, int batch_beam_size, cudaStream_t stream);

template <typename T>
__global__ void UpdatePositionIds(T* positions, int total_length, int new_kv_length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < new_kv_length) {
    positions[i] = total_length + i;
  }
}

template <typename T>
void Launch_UpdatePositionIds(T* positions, int total_length, int new_kv_length, cudaStream_t stream) {
  int threads = std::min(256, new_kv_length);
  int blocks = (new_kv_length + threads - 1) / threads;
  UpdatePositionIds<T><<<blocks, threads, 0, stream>>>(positions, total_length, new_kv_length);
}

template void Launch_UpdatePositionIds(int32_t* positions, int total_length, int new_kv_length, cudaStream_t stream);
template void Launch_UpdatePositionIds(int64_t* positions, int total_length, int new_kv_length, cudaStream_t stream);

template <typename T>
__global__ void CopyAndUpdateAttentionMask(T* mask_data, const T* old_mask_data, int batch_beam_size,
                                           int current_length, int max_length) {
  int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  int i = global_index / current_length;
  int j = global_index % current_length;
  if (i < batch_beam_size) {
    if (j < current_length - 1) {
      mask_data[i * max_length + j] = old_mask_data[i * (current_length - 1) + j];
    } else {
      mask_data[i * max_length + j] = 1;
    }
  }
}

template <typename T>
__global__ void UpdateAttentionMask(T* mask_data, int batch_beam_size, int current_length, int max_length) {
  int i = blockIdx.x;
  if (i < batch_beam_size) {
    mask_data[i * max_length + current_length] = 1;
  }
}

template <typename T>
void Launch_UpdateAttentionMask(T* mask_data, const T* old_mask_data, int batch_beam_size, int current_length,
                                int max_length, bool update_only, cudaStream_t stream) {
  if (update_only) {
    UpdateAttentionMask<T>
        <<<batch_beam_size, 1, 0, stream>>>(mask_data, batch_beam_size, current_length, max_length);
  } else {
    CopyAndUpdateAttentionMask<T><<<(batch_beam_size * max_length + 255) / 256, 256, 0, stream>>>(
        mask_data, old_mask_data, batch_beam_size, current_length, max_length);
  }
}

template void Launch_UpdateAttentionMask(int32_t* mask_data, const int32_t* old_mask_data, int batch_beam_size,
                                         int current_length, int max_length, bool update_only, cudaStream_t stream);
template void Launch_UpdateAttentionMask(int64_t* mask_data, const int64_t* old_mask_data, int batch_beam_size,
                                         int current_length, int max_length, bool update_only, cudaStream_t stream);

template <typename T>
__global__ void UpdateAttentionMaskStatic(T* mask_data, int new_kv_length, int total_length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int past_length = total_length - new_kv_length;
  if (i < new_kv_length) {
    mask_data[past_length + i] = 1;
  }
}

template <typename T>
__global__ void UpdateAttentionMask(T* mask_data, int total_length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total_length) {
    mask_data[i] = 1;
  }
}

template <typename T>
void Launch_UpdateAttentionMask(T* mask_data, int new_kv_length , int total_length, bool update_static, cudaStream_t stream) {
  // LEFT OFF ABOUT THE UPDATE THING AND HOW SOMETIMES WE'LL JUST WANT TO UPDATE IN PLACE AND HAVE ACTUAL 0'S AND OTHER TIMES IT'S JUST 1'S ALL THE WAY THROUGH ON A NEW TENSOR SO WE DON'T NEEDT HE OLD ONE
  
  if (update_static) {
    int threads = std::min(256, new_kv_length);
    int blocks = (new_kv_length + threads - 1) / threads;
    UpdateAttentionMaskStatic<T><<<blocks, threads, 0, stream>>>(mask_data, new_kv_length, total_length);
  } else {
    int threads = std::min(256, total_length);
    int blocks = (total_length + threads - 1) / threads;
    UpdateAttentionMask<T><<<blocks, threads, 0, stream>>>(mask_data, total_length);
  }
}

template void Launch_UpdateAttentionMask(int32_t* mask_data, int new_kv_length , int total_length, bool update_static, cudaStream_t stream);
template void Launch_UpdateAttentionMask(int64_t* mask_data, int new_kv_length , int total_length, bool update_static, cudaStream_t stream);

__global__ void HandleEOSArray(float* batch_logits, int batch_beam_size, int vocab_size, const int32_t* eos_token_ids, int eos_token_ids_count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= batch_beam_size)
    return;

  float* logits = batch_logits + index * vocab_size;
  float max = std::numeric_limits<float>::lowest();
  for (int i = 0; i < eos_token_ids_count; i++) {
    max = std::max(max, logits[eos_token_ids[i]]);
    logits[eos_token_ids[i]] = std::numeric_limits<float>::lowest();  // Set all EOS token options to never happen (the first will get the max of all)
  }

  logits[eos_token_ids[0]] = max;  // Set the score of the primary EOS token to the highest of any of the EOS tokens
}

void LaunchHandleEOSArray(float* batch_logits, int batch_beam_size, int vocab_size, const int32_t* eos_token_ids, int eos_token_ids_count, cudaStream_t stream) {
  HandleEOSArray<<<(batch_beam_size + 255) / 256, 256, 0, stream>>>(batch_logits, batch_beam_size, vocab_size, eos_token_ids, eos_token_ids_count);
}

__global__ void ConvertFp16ToFp32(const half* src, float* dst, int count) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < count)
    dst[idx] = __half2float(src[idx]);
}

void LaunchFp16ToFp32(const uint16_t* fp16, float* fp32, int count, cudaStream_t stream) {
  int block_size = 256;
  int num_blocks = (count + block_size - 1) / block_size;
  ConvertFp16ToFp32<<<num_blocks, block_size, 0, stream>>>(reinterpret_cast<const half*>(fp16), fp32, count);
}

__global__ void ConvertInt32ToInt64(const int32_t* src, int64_t* dst, int count) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < count) {
    dst[idx] = src[idx];
  }
}

void LaunchInt32ToInt64(const int32_t* src, int64_t* dst, int count, cudaStream_t stream) {
  int block_size = 256;
  int num_blocks = (count + block_size - 1) / block_size;
  ConvertInt32ToInt64<<<num_blocks, block_size, 0, stream>>>(src, dst, count);
}

}  // namespace cuda
}  // namespace Generators
