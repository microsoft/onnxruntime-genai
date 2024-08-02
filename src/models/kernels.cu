// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <limits>
// #include "types.cuh"

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

// template <typename T>
// __global__ void CacheExpansionKernel(const T* input,
//                                      T* output,
//                                      int beam_width,
//                                      int max_seq_length,
//                                      int head_size) {
//   const int num_heads = gridDim.y;
//   const int sequence_length = gridDim.z;

//   const int bbid = blockIdx.x;
//   const int batch_id = bbid / beam_width;
//   const int head_id = blockIdx.y;
//   const int s = blockIdx.z;
//   const int tidx = threadIdx.x;

//   const int input_offset = ((batch_id * num_heads + head_id) * sequence_length + s) * head_size + tidx;
//   const int output_offset = ((bbid * num_heads + head_id) * max_seq_length + s) * head_size + tidx;

//   if (tidx < head_size) {
//     output[output_offset] = input[input_offset];
//   }
// }

// template <typename T>
// void CacheExpansionKernelLauncher(const T* key_cache,
//                                   T* key_cache_expanded,
//                                   int batch_size,
//                                   int beam_width,
//                                   int num_heads,
//                                   int sequence_length,
//                                   int max_seq_length,
//                                   int head_size,
//                                   cudaStream_t stream) {
//   const dim3 grid(batch_size * beam_width, num_heads, sequence_length);

//   int equiv_head_size = (head_size & 1) == 0 ? (head_size >> 1) : head_size;
//   equiv_head_size = (equiv_head_size & 1) == 0 ? (equiv_head_size >> 1) : equiv_head_size;

//   // Here we know head_size is smaller than max_thread_num_per_block
//   int tpb = std::max(GPU_WARP_SIZE_HOST, equiv_head_size);

//   // round up tpb to power of 2
//   --tpb;
//   tpb |= (tpb >> 1);
//   tpb |= (tpb >> 2);
//   tpb |= (tpb >> 4);
//   tpb |= (tpb >> 8);
//   tpb |= (tpb >> 16);
//   tpb++;

//   if ((head_size % 4) == 0) {
//     using vec_type = typename TypeMapper<T, 4>::Type;
//     const dim3 block(tpb);
//     CacheExpansionKernel<<<grid, block, 0, stream>>>(reinterpret_cast<const vec_type*>(key_cache),
//                                                      reinterpret_cast<vec_type*>(key_cache_expanded),
//                                                      beam_width,
//                                                      max_seq_length,
//                                                      equiv_head_size);
//   } else if ((head_size & 1) == 0) {
//     using vec_type = typename TypeMapper<T, 2>::Type;
//     const dim3 block(tpb);
//     CacheExpansionKernel<<<grid, block, 0, stream>>>(reinterpret_cast<const vec_type*>(key_cache),
//                                                      reinterpret_cast<vec_type*>(key_cache_expanded),
//                                                      beam_width,
//                                                      max_seq_length,
//                                                      equiv_head_size);
//   } else {
//     const dim3 block(tpb);
//     CacheExpansionKernel<<<grid, block, 0, stream>>>(key_cache,
//                                                      key_cache_expanded,
//                                                      beam_width,
//                                                      max_seq_length,
//                                                      head_size);
//   }
// }

// template void CacheExpansionKernelLauncher(const float* key_cache,
//                                            float* key_cache_expanded,
//                                            int batch_size,
//                                            int beam_width,
//                                            int num_heads,
//                                            int sequence_length,
//                                            int max_seq_length,
//                                            int head_size,
//                                            cudaStream_t stream);

// template void CacheExpansionKernelLauncher(const half* key_cache,
//                                            half* key_cache_expanded,
//                                            int batch_size,
//                                            int beam_width,
//                                            int num_heads,
//                                            int sequence_length,
//                                            int max_seq_length,
//                                            int head_size,
//                                            cudaStream_t stream);

// template void CacheExpansionKernelLauncher(const int32_t* key_cache,
//                                            int32_t* key_cache_expanded,
//                                            int batch_size,
//                                            int beam_width,
//                                            int num_heads,
//                                            int sequence_length,
//                                            int max_seq_length,
//                                            int head_size,
//                                            cudaStream_t stream);

// Support head_size up to 128
constexpr unsigned int kTileSize = 32;
constexpr unsigned int kSeqTileSize = 16;

__global__ void ReorderPastStatesKernel(float4* out_buffer,
                                        const float4* in_buffer,
                                        int batch_size,
                                        int num_heads,
                                        int max_length,
                                        int chunked_head_size) {
  __shared__ float4 tile[kSeqTileSize][kTileSize + 1];

  const int b = blockIdx.z;
  const int n = blockIdx.y;
  const int s_base = blockIdx.x * kSeqTileSize;
  const int s = s_base + threadIdx.y;
  const int base_offset = (b * num_heads + n) * max_length * chunked_head_size;

  if (s < max_length) {
    const int in_offset = base_offset + s * chunked_head_size + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = in_buffer[in_offset];
  }

  __syncthreads();

  const int tidx = threadIdx.x + threadIdx.y * chunked_head_size;
  const int tidx_x = tidx % kSeqTileSize;
  const int tidx_y = tidx / kSeqTileSize;

  const int s2 = s_base + tidx_x;

  if (s2 < max_length) {
    const int out_offset = base_offset + tidx_y * max_length + s2;
    out_buffer[out_offset] = tile[tidx_x][tidx_y];
  }
}

void ReorderPastStatesKernelLauncher(void* out_buffer,
                                     const void* in_buffer,
                                     int batch_size,
                                     int num_heads,
                                     int max_length,
                                     int head_size,
                                     int chunk_size,
                                     cudaStream_t stream) {
  // [B, N, max_length, H2(head_size/chunk_size), equv_chunk_size] -> [B, N, H2(head_size/chunk_size), max_length, equv_chunk_size]
  const int chunked_head_size = head_size / chunk_size;
  const dim3 block(chunked_head_size, kSeqTileSize);
  const dim3 grid((max_length + kSeqTileSize - 1) / kSeqTileSize, num_heads, batch_size);
  if (chunk_size == 4 || chunk_size == 8) {
    ReorderPastStatesKernel<<<grid, block, 0, stream>>>(reinterpret_cast<float4*>(out_buffer),
                                                        reinterpret_cast<const float4*>(in_buffer),
                                                        batch_size,
                                                        num_heads,
                                                        max_length,
                                                        chunked_head_size);
  }
}

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

__global__ void ConvertFp32ToFp16(const float* src, half* dst, int count) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < count)
    dst[idx] = __float2half(src[idx]);
}

void LaunchFp32ToFp16(const float* fp32, uint16_t* fp16, int count, cudaStream_t stream) {
  int block_size = 256;
  int num_blocks = (count + block_size - 1) / block_size;
  ConvertFp32ToFp16<<<num_blocks, block_size, 0, stream>>>(fp32, reinterpret_cast<half*>(fp16), count);
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
