// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <limits>
#include <assert.h>
#include <stdio.h>

namespace Generators {
namespace cuda {

template <typename T>
__global__ void UpdatePositionIds(T* positions, int batch_beam_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch_beam_size)
    positions[i]++;
}

template <typename T>
__global__ void UpdatePositionIds(T* positions, int total_length, int new_kv_length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < new_kv_length)
    positions[i] = i + total_length - new_kv_length;
}

template <typename T>
void Launch_UpdatePositionIds(T* positions, int batch_beam_size, int total_length, int new_kv_length, cudaStream_t stream) {
  if (batch_beam_size == 1) {
    // For batch size == 1 we calculate position ids with total length and new kv length for continuous decoding
    int threads = std::min(256, new_kv_length);
    int blocks = (new_kv_length + threads - 1) / threads;
    UpdatePositionIds<T><<<blocks, threads, 0, stream>>>(positions, total_length, new_kv_length);
  } else {
    // For batch size > 1 we increment position ids by 1... continuous decoding is not supported
    UpdatePositionIds<T><<<(batch_beam_size + 255) / 256, 256, 0, stream>>>(positions, batch_beam_size);
  }
}

template void Launch_UpdatePositionIds(int32_t* positions, int batch_beam_size, int total_length, int new_kv_length, cudaStream_t stream);
template void Launch_UpdatePositionIds(int64_t* positions, int batch_beam_size, int total_length, int new_kv_length, cudaStream_t stream);

template <typename T>
__global__ void UpdateAttentionMask(T* mask_data, int batch_beam_size, int new_kv_length, int total_length, int max_length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id = i / new_kv_length;
  int seq_id = i % new_kv_length;
  if (i < new_kv_length * batch_beam_size) {
    mask_data[batch_id * max_length + total_length - seq_id] = 1;
  }
}

template <typename T>
__global__ void CopyAndUpdateAttentionMask(T* mask_data, const T* old_data, int batch_beam_size, int new_kv_length, int total_length, int max_length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id = i / total_length;
  int seq_id = i % total_length;
  if (i < total_length * batch_beam_size) {
    if (seq_id < total_length - new_kv_length) {
      mask_data[batch_id * max_length + seq_id] = old_data[batch_id * (total_length - new_kv_length) + seq_id];
    } else {
      mask_data[batch_id * max_length + seq_id] = 1;
    }
  }
}

template <typename T>
void Launch_UpdateAttentionMask(T* mask_data, const T* old_data, int batch_beam_size, int new_kv_length,
                                int total_length, int max_length, bool update_only, cudaStream_t stream) {
  if (update_only) {
    int threads = std::min(256, batch_beam_size * new_kv_length);
    int blocks = (batch_beam_size * new_kv_length + threads - 1) / threads;
    UpdateAttentionMask<T><<<blocks, threads, 0, stream>>>(mask_data, batch_beam_size, new_kv_length, total_length, max_length);
  } else {
    int threads = std::min(256, batch_beam_size * total_length);
    int blocks = (batch_beam_size * total_length + threads - 1) / threads;
    CopyAndUpdateAttentionMask<T><<<blocks, threads, 0, stream>>>(mask_data, old_data, batch_beam_size, new_kv_length, total_length, max_length);
  }
}

template void Launch_UpdateAttentionMask(int32_t* mask_data, const int32_t* old_data, int batch_beam_size, int new_kv_length, int total_length, int max_length, bool update_only, cudaStream_t stream);
template void Launch_UpdateAttentionMask(int64_t* mask_data, const int64_t* old_data, int batch_beam_size, int new_kv_length, int total_length, int max_length, bool update_only, cudaStream_t stream);

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

namespace {

struct ReorderPastStateParams {
  // Support head_size up to 128
  constexpr static unsigned int kTileSize = 32;
  constexpr static unsigned int kSeqTileSize = 16;
};

}  // namespace

__global__ void ReorderPastStatesKernel(float4* out_buffer,
                                        const float4* in_buffer,
                                        int batch_size,
                                        int num_heads,
                                        int max_length,
                                        int chunked_head_size) {
  __shared__ float4 tile[ReorderPastStateParams::kSeqTileSize][ReorderPastStateParams::kTileSize + 1];

  const int b = blockIdx.z;
  const int n = blockIdx.y;
  const int s_base = blockIdx.x * ReorderPastStateParams::kSeqTileSize;
  const int s = s_base + threadIdx.y;
  const int base_offset = (b * num_heads + n) * max_length * chunked_head_size;

  if (s < max_length) {
    const int in_offset = base_offset + s * chunked_head_size + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = in_buffer[in_offset];
  }

  __syncthreads();

  const int tidx = threadIdx.x + threadIdx.y * chunked_head_size;
  const int tidx_x = tidx % ReorderPastStateParams::kSeqTileSize;
  const int tidx_y = tidx / ReorderPastStateParams::kSeqTileSize;

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
  const dim3 block(chunked_head_size, ReorderPastStateParams::kSeqTileSize);
  const dim3 grid((max_length + ReorderPastStateParams::kSeqTileSize - 1) / ReorderPastStateParams::kSeqTileSize, num_heads, batch_size);
  if (chunk_size == 4 || chunk_size == 8) {
    ReorderPastStatesKernel<<<grid, block, 0, stream>>>(reinterpret_cast<float4*>(out_buffer),
                                                        reinterpret_cast<const float4*>(in_buffer),
                                                        batch_size,
                                                        num_heads,
                                                        max_length,
                                                        chunked_head_size);
  }
}

__global__ void UpdateCacheIndirectionKernel(int32_t* tgt_indir_cache,
                                             const int32_t* src_indir_cache,
                                             const int32_t* beam_ids,
                                             int batch_size,
                                             int beam_width,
                                             int input_seq_length,
                                             int max_seq_length,
                                             int current_length) {
  int time_step = threadIdx.x + blockIdx.x * blockDim.x;
  int bb_id = threadIdx.y + blockIdx.y * blockDim.y;
  const int batch_id = bb_id / beam_width;
  const int beam_id = bb_id % beam_width;

  if (bb_id >= beam_width * batch_size || time_step >= current_length) {
    return;
  }

  const int src_beam = beam_ids[batch_id * beam_width + beam_id] % beam_width;

  const int tgt_offset = batch_id * beam_width * max_seq_length + beam_id * max_seq_length + time_step;

  if (time_step < input_seq_length) {
    // For time steps that correspond to the input sequence,
    // the beam that it comes from is always 0.
    tgt_indir_cache[tgt_offset] = static_cast<int32_t>(0);
  } else if (time_step == (current_length - 1)) {
    // For the final (newly generated) time step,
    // the beam that it comes from is always the beam that we
    // are currently processing (i.e.) from this point on, these time-steps
    // form the new beams.
    tgt_indir_cache[tgt_offset] = static_cast<int32_t>(beam_id);
  } else {
    // For all other time-steps, we look up the source indirection, to
    // see which beam it came from based on the `src_beam`.
    const int src_offset = batch_id * beam_width * max_seq_length + src_beam * max_seq_length + time_step;
    tgt_indir_cache[tgt_offset] = src_indir_cache[src_offset];
  }
}

void UpdateCacheIndirectionKernelLauncher(int32_t* tgt_indir_cache,
                                          const int32_t* src_indir_cache,
                                          const int32_t* beam_ids,
                                          int batch_size,
                                          int beam_width,
                                          int input_seq_length,
                                          int max_seq_length,
                                          int current_length,
                                          cudaStream_t stream) {
  const dim3 block(32);
  const dim3 grid((current_length + block.x - 1) / block.x, batch_size * beam_width);
  UpdateCacheIndirectionKernel<<<grid, block, 0, stream>>>(tgt_indir_cache,
                                                           src_indir_cache,
                                                           beam_ids,
                                                           batch_size,
                                                           beam_width,
                                                           input_seq_length,
                                                           max_seq_length,
                                                           current_length);
}

template <typename T>
__global__ void CopyCrossQKSingleDecodeStepKernel(T* target,  // shape [batch_beam_size, num_alignment_heads, max_length, frames]
                                                  T** qk_layer_pointers,
                                                  int token_index,
                                                  int num_layers,
                                                  int num_heads,
                                                  const int* alignment_heads,
                                                  int frames,
                                                  int max_length) {
  const int pair = blockIdx.x;
  const int num_alignment_heads = gridDim.x;
  const int bbm = blockIdx.y;
  alignment_heads += (pair * 2);
  const int layer = *alignment_heads;
  const int head = *(alignment_heads + 1);

  target += ((int64_t)bbm * num_alignment_heads + pair) * max_length * frames + ((int64_t)token_index * frames);
  T* src = qk_layer_pointers[layer] + ((int64_t)bbm * num_heads + head) * frames;

  for (int tid = threadIdx.x; tid < frames; tid += blockDim.x) {
    target[tid] = src[tid];  // use vectorized read write in future if needed
  }
}

template <typename T>
void LaunchCopyCrossQKSingleDecodeStep(cudaStream_t stream,
                                       T* cross_qk_buffer_data,
                                       T** qk_layer_pointers,
                                       int token_index,
                                       int batch_beam_size,
                                       int num_layers,
                                       int num_heads,
                                       int num_alignment_heads,
                                       const int* alignment_heads,
                                       int frames,
                                       int max_length) {
  dim3 block(512);
  dim3 grid(num_alignment_heads, batch_beam_size);

  CopyCrossQKSingleDecodeStepKernel<<<grid, block, 0, stream>>>(cross_qk_buffer_data,
                                                                qk_layer_pointers,
                                                                token_index,
                                                                num_layers,
                                                                num_heads,
                                                                alignment_heads,
                                                                frames,
                                                                max_length);
}

template void LaunchCopyCrossQKSingleDecodeStep(cudaStream_t stream,
                                                float* cross_qk_buffer_data,
                                                float** qk_layer_pointers,
                                                int token_index,
                                                int batch_beam_size,
                                                int num_layers,
                                                int num_heads,
                                                int num_alignment_heads,
                                                const int* alignment_heads,
                                                int frames,
                                                int max_length);

template <typename T>
__global__ void CopyDecoderCrossQKAllStepsKernel(int context_decoding_len,
                                                 int num_beams,
                                                 int num_return_sequences,
                                                 int max_length,
                                                 int frames_of_k,
                                                 const T* cross_qk_buffer_data,  // [batch, num_beams, num_alignment_heads, max_length, frames]
                                                 T* cross_qk_output,             // [batch, num_return_sequences, num_alignment_heads, total_decoding_length, frames]
                                                 const int* cache_indir_data) {  // [batch, num_beams, max_length]
  const int pair = blockIdx.y;
  const int num_alignment_heads = gridDim.y;
  const int total_decoding_length = gridDim.x;
  const int token_decoding_index = blockIdx.x;
  const int br = blockIdx.z;
  const int batch = br / num_return_sequences;
  const int ret_seq_id = br % num_return_sequences;

  const int64_t offset_in_cache = ((int64_t)batch * num_return_sequences + ret_seq_id) * max_length + token_decoding_index + context_decoding_len;
  int bi_src = batch * num_beams + cache_indir_data[offset_in_cache];

  T* target = cross_qk_output + (((int64_t)br * num_alignment_heads + (int64_t)pair) * total_decoding_length + token_decoding_index) * frames_of_k;
  const T* src = cross_qk_buffer_data + (((int64_t)bi_src * num_alignment_heads + (int64_t)pair) * max_length + token_decoding_index) * frames_of_k;
  for (int tid = threadIdx.x; tid < frames_of_k; tid += blockDim.x) {
    target[tid] = src[tid];  // use vectorized read write in future if needed
  }
}

template <typename T>
void LaunchFinalizeCrossQK(cudaStream_t stream,
                           int iteration_number,
                           int context_decoding_len,
                           int batch_size,
                           int num_beams,
                           int max_length,
                           int num_alignment_heads,
                           int frames_of_k,
                           const T* cross_qk_buffer_data,
                           T* cross_qk_output,
                           int num_return_sequences,
                           const int* cache_indir_data) {
  int64_t br = (int64_t)batch_size * num_return_sequences;
  assert(br < 65536L && num_alignment_heads < 65536);

  const int total_decoding_length = iteration_number - 1;
  dim3 block(512);
  dim3 grid(total_decoding_length, num_alignment_heads, (unsigned)br);

  CopyDecoderCrossQKAllStepsKernel<<<grid, block, 0, stream>>>(context_decoding_len,
                                                               num_beams,
                                                               num_return_sequences,
                                                               max_length,
                                                               frames_of_k,
                                                               cross_qk_buffer_data,
                                                               cross_qk_output,
                                                               cache_indir_data);
}

template void LaunchFinalizeCrossQK(cudaStream_t stream,
                                    int iteration_number,
                                    int context_decoding_len,
                                    int batch_size,
                                    int num_beams,
                                    int max_length,
                                    int num_alignment_heads,
                                    int frames_of_k,
                                    const float* cross_qk_buffer_data,
                                    float* cross_qk_output,
                                    int num_return_sequences,
                                    const int* cache_indir_data);

}  // namespace cuda
}  // namespace Generators
