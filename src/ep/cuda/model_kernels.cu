// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <assert.h>
#include <stdio.h>
#include "cuda_common.h"
#include "kernels.h"

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
  CUDA_CHECK_LAUNCH();
}

template void Launch_UpdatePositionIds(int32_t* positions, int batch_beam_size, int total_length, int new_kv_length, cudaStream_t stream);
template void Launch_UpdatePositionIds(int64_t* positions, int batch_beam_size, int total_length, int new_kv_length, cudaStream_t stream);

template <typename T>
__global__ void UpdateAttentionMaskStatic(T* mask_data, int batch_beam_size, int new_kv_length, int total_length, int max_length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id = i / new_kv_length;
  int seq_id = (i % new_kv_length) + 1;
  if (i < new_kv_length * batch_beam_size) {
    mask_data[batch_id * max_length + total_length - seq_id] = 1;
  }
}

template <typename T>
__global__ void CopyAndUpdateAttentionMask(T* next_mask_data, const T* mask_data, int batch_beam_size, int new_kv_length, int total_length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id = i / total_length;
  int seq_id = i % total_length;
  if (i < total_length * batch_beam_size) {
    if (seq_id < total_length - new_kv_length) {
      next_mask_data[batch_id * total_length + seq_id] = mask_data[batch_id * (total_length - new_kv_length) + seq_id];
    } else {
      next_mask_data[batch_id * total_length + seq_id] = 1;
    }
  }
}

template <typename T>
void Launch_UpdateAttentionMask(T* next_mask_data, T* mask_data, int batch_beam_size, int new_kv_length,
                                int total_length, int max_length, bool update_only, cudaStream_t stream) {
  if (update_only) {
    int threads = std::min(256, batch_beam_size * new_kv_length);
    int blocks = (batch_beam_size * new_kv_length + threads - 1) / threads;
    UpdateAttentionMaskStatic<T><<<blocks, threads, 0, stream>>>(mask_data, batch_beam_size, new_kv_length, total_length, max_length);
  } else {
    int threads = std::min(256, batch_beam_size * total_length);
    int blocks = (batch_beam_size * total_length + threads - 1) / threads;
    CopyAndUpdateAttentionMask<T><<<blocks, threads, 0, stream>>>(next_mask_data, mask_data, batch_beam_size, new_kv_length, total_length);
  }
  CUDA_CHECK_LAUNCH();
}

template void Launch_UpdateAttentionMask(int32_t* next_mask_data, int32_t* mask_data, int batch_beam_size, int new_kv_length, int total_length, int max_length, bool update_only, cudaStream_t stream);
template void Launch_UpdateAttentionMask(int64_t* next_mask_data, int64_t* mask_data, int batch_beam_size, int new_kv_length, int total_length, int max_length, bool update_only, cudaStream_t stream);

__global__ void AddLogitsMask(float* batch_logits, int batch_beam_size, int vocab_size, const uint32_t* logits_mask) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= batch_beam_size * vocab_size)
    return;
  int batch_index = index / vocab_size;
  int vocab_index = index % vocab_size;
  const size_t words_per_row = (static_cast<size_t>(vocab_size) + 31) / 32;
  const size_t mask_index = static_cast<size_t>(batch_index) * words_per_row +
                            static_cast<size_t>(vocab_index) / 32;
  if (!(logits_mask[mask_index] & (uint32_t{1} << (vocab_index % 32))))
    batch_logits[index] = std::numeric_limits<float>::lowest();
}

void LaunchAddLogitsMask(float* batch_logits, int batch_beam_size, int vocab_size, const uint32_t* logits_mask, cudaStream_t stream) {
  int block_size = 256;
  int num_blocks = (batch_beam_size * vocab_size + block_size - 1) / block_size;
  AddLogitsMask<<<num_blocks, block_size, 0, stream>>>(batch_logits, batch_beam_size, vocab_size, logits_mask);
  CUDA_CHECK_LAUNCH();
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
  CUDA_CHECK_LAUNCH();
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
  CUDA_CHECK_LAUNCH();
}

__global__ void ConvertBf16ToFp32(const __nv_bfloat16* src, float* dst, int count) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < count)
    dst[idx] = __bfloat162float(src[idx]);
}

void LaunchBf16ToFp32(const uint16_t* bf16, float* fp32, int count, cudaStream_t stream) {
  constexpr int block_size = 256;
  const int num_blocks = (count + block_size - 1) / block_size;
  ConvertBf16ToFp32<<<num_blocks, block_size, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(bf16), fp32, count);
  CUDA_CHECK_LAUNCH();
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
  CUDA_CHECK_LAUNCH();
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
    CUDA_CHECK_LAUNCH();
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
  CUDA_CHECK_LAUNCH();
}

template <typename T>
__global__ void CopyCrossQKSingleDecodeStepKernel(T* target,  // shape [batch_beam_size, num_alignment_heads, max_length, frames]
                                                  void** qk_layer_pointers,
                                                  int token_index,
                                                  int num_layers,
                                                  int num_heads,
                                                  const int* alignment_heads,
                                                  int frames,
                                                  int max_length,
                                                  int sequence_length) {
  const int pair = blockIdx.x;
  const int num_alignment_heads = gridDim.x;
  const int bbm = blockIdx.y;
  alignment_heads += (pair * 2);
  const int layer = *alignment_heads;
  const int head = *(alignment_heads + 1);

  target += ((int64_t)bbm * num_alignment_heads + pair) * max_length * frames + ((int64_t)token_index * frames);
  T* src = reinterpret_cast<T*>(qk_layer_pointers[layer]) + ((int64_t)bbm * num_heads + head) * sequence_length * frames;

  for (int tid = threadIdx.x; tid < frames; tid += blockDim.x) {
    target[tid] = src[tid];  // use vectorized read write in future if needed
    for (int i = 1; i < sequence_length; i++) {
      target[i * frames + tid] = src[i * frames + tid];
    }
  }
}

template <typename T>
void LaunchCopyCrossQKSingleDecodeStep(cudaStream_t stream,
                                       T* cross_qk_buffer_data,
                                       void** qk_layer_pointers,
                                       int token_index,
                                       int batch_beam_size,
                                       int num_layers,
                                       int num_heads,
                                       int num_alignment_heads,
                                       const int* alignment_heads,
                                       int frames,
                                       int max_length,
                                       int sequence_length) {
  dim3 block(512);
  dim3 grid(num_alignment_heads, batch_beam_size);

  if (std::is_same<T, uint16_t>::value) {
    CopyCrossQKSingleDecodeStepKernel<<<grid, block, 0, stream>>>(reinterpret_cast<half*>(cross_qk_buffer_data),
                                                                  qk_layer_pointers,
                                                                  token_index,
                                                                  num_layers,
                                                                  num_heads,
                                                                  alignment_heads,
                                                                  frames,
                                                                  max_length,
                                                                  sequence_length);
  } else {
    CopyCrossQKSingleDecodeStepKernel<<<grid, block, 0, stream>>>(cross_qk_buffer_data,
                                                                  qk_layer_pointers,
                                                                  token_index,
                                                                  num_layers,
                                                                  num_heads,
                                                                  alignment_heads,
                                                                  frames,
                                                                  max_length,
                                                                  sequence_length);
  }
  CUDA_CHECK_LAUNCH();
}

template void LaunchCopyCrossQKSingleDecodeStep(cudaStream_t stream,
                                                float* cross_qk_buffer_data,
                                                void** qk_layer_pointers,
                                                int token_index,
                                                int batch_beam_size,
                                                int num_layers,
                                                int num_heads,
                                                int num_alignment_heads,
                                                const int* alignment_heads,
                                                int frames,
                                                int max_length,
                                                int sequence_length);

template void LaunchCopyCrossQKSingleDecodeStep(cudaStream_t stream,
                                                uint16_t* cross_qk_buffer_data,
                                                void** qk_layer_pointers,
                                                int token_index,
                                                int batch_beam_size,
                                                int num_layers,
                                                int num_heads,
                                                int num_alignment_heads,
                                                const int* alignment_heads,
                                                int frames,
                                                int max_length,
                                                int sequence_length);

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

  const int64_t offset_in_cache = ((int64_t)batch * num_return_sequences + ret_seq_id) * max_length + token_decoding_index;
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
                           int batch_beam_size,
                           int num_beams,
                           int max_length,
                           int num_alignment_heads,
                           int frames_of_k,
                           const T* cross_qk_buffer_data,
                           T* cross_qk_output,
                           int num_return_sequences,
                           const int* cache_indir_data) {
  int64_t br = (int64_t)batch_beam_size;
  assert(br < 65536L && num_alignment_heads < 65536);

  const int total_decoding_length = iteration_number;
  dim3 block(512);
  dim3 grid(total_decoding_length, num_alignment_heads, (unsigned)br);

  if (std::is_same<T, uint16_t>::value) {
    CopyDecoderCrossQKAllStepsKernel<<<grid, block, 0, stream>>>(context_decoding_len,
                                                                 num_beams,
                                                                 num_return_sequences,
                                                                 max_length,
                                                                 frames_of_k,
                                                                 reinterpret_cast<const half*>(cross_qk_buffer_data),
                                                                 reinterpret_cast<half*>(cross_qk_output),
                                                                 cache_indir_data);
  } else {
    CopyDecoderCrossQKAllStepsKernel<<<grid, block, 0, stream>>>(context_decoding_len,
                                                                 num_beams,
                                                                 num_return_sequences,
                                                                 max_length,
                                                                 frames_of_k,
                                                                 cross_qk_buffer_data,
                                                                 cross_qk_output,
                                                                 cache_indir_data);
  }
  CUDA_CHECK_LAUNCH();
}

template void LaunchFinalizeCrossQK(cudaStream_t stream,
                                    int iteration_number,
                                    int context_decoding_len,
                                    int batch_beam_size,
                                    int num_beams,
                                    int max_length,
                                    int num_alignment_heads,
                                    int frames_of_k,
                                    const float* cross_qk_buffer_data,
                                    float* cross_qk_output,
                                    int num_return_sequences,
                                    const int* cache_indir_data);

template void LaunchFinalizeCrossQK(cudaStream_t stream,
                                    int iteration_number,
                                    int context_decoding_len,
                                    int batch_beam_size,
                                    int num_beams,
                                    int max_length,
                                    int num_alignment_heads,
                                    int frames_of_k,
                                    const uint16_t* cross_qk_buffer_data,
                                    uint16_t* cross_qk_output,
                                    int num_return_sequences,
                                    const int* cache_indir_data);

namespace {

struct StateSlotDescGpu {
  uint8_t* base;
  uint64_t slot_bytes;
};

struct StateUpdateReplayDescGpu {
  const void* source_state;
  void* destination_state;
  const void* value;
  const float* decay;
  const float* key;
  const float* delta;
  const void* source_aux_state;
  void* destination_aux_state;
  const int32_t* source_lengths;
  int32_t* destination_lengths;
  uint64_t channel_count;
  uint64_t state_width;
  uint64_t key_width;
  uint64_t key_head_count;
  uint64_t state_capacity;
  uint64_t aux_capacity;
  uint32_t capacity;
  uint32_t kept_count;
  uint32_t element_size;
  uint32_t compress_ratio;
  uint32_t kind;
};

constexpr int kSlotCopyThreads = 256;
constexpr int kSlotCopyBlocksPerTensor = 128;

__global__ void CopyStateSlotsKernel(const StateSlotDescGpu* __restrict__ descs, int src_slot, int dst_slot) {
  const StateSlotDescGpu desc = descs[blockIdx.y];
  const uint64_t bytes = desc.slot_bytes;
  uint8_t* dst_bytes = desc.base + static_cast<uint64_t>(dst_slot) * bytes;
  const uint8_t* src_bytes = desc.base + static_cast<uint64_t>(src_slot) * bytes;

  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * kSlotCopyThreads;
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * kSlotCopyThreads + threadIdx.x;

  uint64_t copied_bytes = 0;
  if (((reinterpret_cast<uintptr_t>(src_bytes) | reinterpret_cast<uintptr_t>(dst_bytes)) & 0xF) == 0) {
    const uint64_t vec_count = bytes >> 4;
    auto* dst_vec = reinterpret_cast<uint4*>(dst_bytes);
    const auto* src_vec = reinterpret_cast<const uint4*>(src_bytes);
    for (uint64_t i = start; i < vec_count; i += stride) {
      dst_vec[i] = src_vec[i];
    }
    copied_bytes = vec_count << 4;
  }

  for (uint64_t i = copied_bytes + start; i < bytes; i += stride) {
    dst_bytes[i] = src_bytes[i];
  }
}

__global__ void ReplayStateUpdatesKernel(const StateUpdateReplayDescGpu* __restrict__ descs) {
  const StateUpdateReplayDescGpu descriptor = descs[blockIdx.y];
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (descriptor.kind == 3) {
    const uint64_t update_offset =
        static_cast<uint64_t>(descriptor.kept_count - 1) * descriptor.state_width;
    for (uint64_t index = start; index < descriptor.state_width; index += stride) {
      const uint64_t source_index = update_offset + index;
      if (descriptor.element_size == 8) {
        static_cast<uint64_t*>(descriptor.destination_state)[index] =
            static_cast<const uint64_t*>(descriptor.value)[source_index];
      } else if (descriptor.element_size == 2) {
        static_cast<uint16_t*>(descriptor.destination_state)[index] =
            static_cast<const uint16_t*>(descriptor.value)[source_index];
      } else if (descriptor.element_size == 4) {
        static_cast<uint32_t*>(descriptor.destination_state)[index] =
            static_cast<const uint32_t*>(descriptor.value)[source_index];
      }
    }
    return;
  }

  if (descriptor.kind == 4) {
    const int32_t old_key_length = descriptor.source_lengths[0];
    const int32_t old_buffer_length = descriptor.source_lengths[1];
    const uint64_t new_block_count =
        (static_cast<uint64_t>(old_buffer_length) + descriptor.kept_count) /
        descriptor.compress_ratio;
    const uint64_t new_buffer_length =
        (static_cast<uint64_t>(old_buffer_length) + descriptor.kept_count) %
        descriptor.compress_ratio;
    const uint64_t work_entries = descriptor.state_capacity > descriptor.aux_capacity
                                      ? descriptor.state_capacity
                                      : descriptor.aux_capacity;
    for (uint64_t index = start; index < work_entries * descriptor.state_width; index += stride) {
      const uint64_t entry = index / descriptor.state_width;
      const uint64_t component = index % descriptor.state_width;
      if (entry < descriptor.state_capacity) {
        uint64_t source_index = index;
        const void* source = descriptor.source_state;
        if (entry >= static_cast<uint64_t>(old_key_length) &&
            entry < static_cast<uint64_t>(old_key_length) + new_block_count) {
          const uint64_t completion_token =
              descriptor.compress_ratio - old_buffer_length - 1 +
              (entry - old_key_length) * descriptor.compress_ratio;
          source = descriptor.value;
          source_index = completion_token * descriptor.state_width + component;
        }
        if (descriptor.element_size == 2) {
          static_cast<uint16_t*>(descriptor.destination_state)[index] =
              static_cast<const uint16_t*>(source)[source_index];
        } else {
          static_cast<uint32_t*>(descriptor.destination_state)[index] =
              static_cast<const uint32_t*>(source)[source_index];
        }
      }
      if (entry < descriptor.aux_capacity) {
        uint64_t source_index = index;
        const void* source = descriptor.source_aux_state;
        if (entry < new_buffer_length) {
          const uint64_t virtual_position = new_block_count * descriptor.compress_ratio + entry;
          const bool from_old_buffer = virtual_position < static_cast<uint64_t>(old_buffer_length);
          source_index =
              (from_old_buffer ? virtual_position : virtual_position - old_buffer_length) *
                  descriptor.state_width +
              component;
          source = from_old_buffer ? descriptor.source_aux_state : descriptor.value;
        }
        if (descriptor.element_size == 2) {
          static_cast<uint16_t*>(descriptor.destination_aux_state)[index] =
              static_cast<const uint16_t*>(source)[source_index];
        } else {
          static_cast<uint32_t*>(descriptor.destination_aux_state)[index] =
              static_cast<const uint32_t*>(source)[source_index];
        }
      }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      descriptor.destination_lengths[0] = old_key_length + static_cast<int32_t>(new_block_count);
      descriptor.destination_lengths[1] = static_cast<int32_t>(new_buffer_length);
    }
    return;
  }

  const uint64_t state_elements =
      descriptor.channel_count * descriptor.state_width *
      (descriptor.kind == 1 ? uint64_t{1} : descriptor.key_width);

  for (uint64_t state_index = start; state_index < state_elements; state_index += stride) {
    if (descriptor.kind == 1) {
      const uint64_t channel = state_index / descriptor.state_width;
      const uint64_t position = state_index - channel * descriptor.state_width;
      const uint64_t shifted_position = position + descriptor.kept_count;
      const uint64_t source_index = shifted_position < descriptor.state_width
                                        ? channel * descriptor.state_width + shifted_position
                                        : (shifted_position - descriptor.state_width) *
                                                  descriptor.channel_count +
                                              channel;
      if (descriptor.element_size == 2) {
        const auto* source = shifted_position < descriptor.state_width
                                 ? static_cast<const uint16_t*>(descriptor.source_state)
                                 : static_cast<const uint16_t*>(descriptor.value);
        static_cast<uint16_t*>(descriptor.destination_state)[state_index] = source[source_index];
      } else if (descriptor.element_size == 4) {
        const auto* source = shifted_position < descriptor.state_width
                                 ? static_cast<const uint32_t*>(descriptor.source_state)
                                 : static_cast<const uint32_t*>(descriptor.value);
        static_cast<uint32_t*>(descriptor.destination_state)[state_index] = source[source_index];
      }
      continue;
    }

    const uint64_t key_index = state_index % descriptor.key_width;
    const uint64_t value_index =
        (state_index / descriptor.key_width) % descriptor.state_width;
    const uint64_t value_head =
        state_index / (descriptor.state_width * descriptor.key_width);
    const uint64_t key_head =
        value_head * descriptor.key_head_count / descriptor.channel_count;
    float state = static_cast<const float*>(descriptor.source_state)[state_index];
    for (uint32_t transition = 0; transition < descriptor.kept_count; ++transition) {
      state = __fmul_rn(
          state,
          descriptor.decay[static_cast<uint64_t>(transition) * descriptor.channel_count + value_head]);
      // Match GatedDeltaNet's separately rounded multiply and add. Contracting this update changes
      // the target state after replaying an accepted prefix and can alter later greedy tokens.
      const float update = __fmul_rn(
          descriptor.key[(static_cast<uint64_t>(transition) * descriptor.key_head_count + key_head) *
                             descriptor.key_width +
                         key_index],
          descriptor.delta[(static_cast<uint64_t>(transition) * descriptor.channel_count + value_head) *
                               descriptor.state_width +
                           value_index]);
      state = __fadd_rn(state, update);
    }
    static_cast<float*>(descriptor.destination_state)[state_index] = state;
  }
}

}  // namespace

// Fast compact replay for gated-delta-net states. One block owns kReplayRows value rows of one head:
// it stages that head's decays, the kept keys, and the rows' deltas in shared memory, and each warp
// then streams whole rows as float4 through registers. It computes exactly what
// ReplayStateUpdatesKernel does for these descriptors, without the per-element index arithmetic and
// global reloads of the transition factors that keep that kernel below the bandwidth roofline.
constexpr int kReplayThreads = 256;
constexpr int kReplayRows = 32;

__global__ void __launch_bounds__(kReplayThreads) ReplayGatedDeltaNetKernel(
    const StateUpdateReplayDescGpu* __restrict__ descs) {
  __shared__ float key_tile[kMaxFastReplayTransitions * kMaxFastReplayKeyWidth];
  __shared__ float delta_tile[kMaxFastReplayTransitions * kReplayRows];
  __shared__ float decay_tile[kMaxFastReplayTransitions];

  const StateUpdateReplayDescGpu d = descs[blockIdx.y];
  const int value_width = static_cast<int>(d.state_width);
  const int key_width = static_cast<int>(d.key_width);
  const int heads = static_cast<int>(d.channel_count);
  const int key_heads = static_cast<int>(d.key_head_count);
  const int kept = static_cast<int>(d.kept_count);
  const int chunks = (value_width + kReplayRows - 1) / kReplayRows;
  const int head = blockIdx.x / chunks;
  if (head >= heads) return;
  const int first_row = (blockIdx.x - head * chunks) * kReplayRows;
  const int key_head = head * key_heads / heads;

  for (int i = threadIdx.x; i < kept * key_width; i += blockDim.x) {
    const int t = i / key_width;
    const int k = i - t * key_width;
    key_tile[t * key_width + k] =
        d.key[(static_cast<size_t>(t) * key_heads + key_head) * key_width + k];
  }
  for (int i = threadIdx.x; i < kept * kReplayRows; i += blockDim.x) {
    const int t = i / kReplayRows;
    const int r = i - t * kReplayRows;
    const int v = first_row + r;
    delta_tile[i] = v < value_width
                        ? d.delta[(static_cast<size_t>(t) * heads + head) * value_width + v]
                        : 0.0f;
  }
  if (threadIdx.x < kept) {
    decay_tile[threadIdx.x] = d.decay[static_cast<size_t>(threadIdx.x) * heads + head];
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const auto* source = static_cast<const float*>(d.source_state);
  auto* destination = static_cast<float*>(d.destination_state);
  for (int r = warp; r < kReplayRows; r += kReplayThreads / 32) {
    const int v = first_row + r;
    if (v >= value_width) break;
    const size_t row = (static_cast<size_t>(head) * value_width + v) * key_width;
    for (int k = lane * 4; k < key_width; k += 128) {
      float4 s = *reinterpret_cast<const float4*>(source + row + k);
      for (int t = 0; t < kept; ++t) {
        const float decay = decay_tile[t];
        const float delta = delta_tile[t * kReplayRows + r];
        const float* key = key_tile + t * key_width + k;
        s.x = __fadd_rn(__fmul_rn(s.x, decay), __fmul_rn(key[0], delta));
        s.y = __fadd_rn(__fmul_rn(s.y, decay), __fmul_rn(key[1], delta));
        s.z = __fadd_rn(__fmul_rn(s.z, decay), __fmul_rn(key[2], delta));
        s.w = __fadd_rn(__fmul_rn(s.w, decay), __fmul_rn(key[3], delta));
      }
      *reinterpret_cast<float4*>(destination + row + k) = s;
    }
  }
}

void LaunchCopyStateSlots(const void* descs, int count, int src_slot, int dst_slot, cudaStream_t stream) {
  if (count <= 0 || src_slot == dst_slot) return;
  const dim3 grid(kSlotCopyBlocksPerTensor, static_cast<unsigned>(count));
  CopyStateSlotsKernel<<<grid, kSlotCopyThreads, 0, stream>>>(
      reinterpret_cast<const StateSlotDescGpu*>(descs), src_slot, dst_slot);
}

void LaunchReplayStateUpdates(const void* descs, int fast_count, int generic_count,
                              int fast_blocks_per_descriptor, cudaStream_t stream) {
  const auto* typed = reinterpret_cast<const StateUpdateReplayDescGpu*>(descs);
  if (fast_count > 0) {
    const dim3 grid(static_cast<unsigned>(fast_blocks_per_descriptor), static_cast<unsigned>(fast_count));
    ReplayGatedDeltaNetKernel<<<grid, kReplayThreads, 0, stream>>>(typed);
    CUDA_CHECK_LAUNCH();
  }
  if (generic_count > 0) {
    const dim3 grid(kSlotCopyBlocksPerTensor, static_cast<unsigned>(generic_count));
    ReplayStateUpdatesKernel<<<grid, kSlotCopyThreads, 0, stream>>>(typed + fast_count);
    CUDA_CHECK_LAUNCH();
  }
}

int ReplayGatedDeltaNetBlocks(int heads, int value_width) {
  return heads * ((value_width + kReplayRows - 1) / kReplayRows);
}

namespace small_copy {

template <int kBytes>
struct BytePayload {
  alignas(16) unsigned char bytes[kBytes];
};

template <int kBytes>
__global__ void StoreBytesKernel(BytePayload<kBytes> payload, unsigned char* __restrict__ destination,
                                 int count) {
  for (int i = threadIdx.x; i < count; i += blockDim.x) destination[i] = payload.bytes[i];
}

template <int kBytes>
void LaunchStoreBytesImpl(void* destination, const void* source, size_t count, cudaStream_t stream) {
  BytePayload<kBytes> payload;
  memcpy(payload.bytes, source, count);
  const auto kernel = StoreBytesKernel<kBytes>;
  const int threads = count < 256 ? 64 : 256;
  kernel<<<1, threads, 0, stream>>>(payload, static_cast<unsigned char*>(destination), static_cast<int>(count));
  CUDA_CHECK_LAUNCH();
}

__global__ void CopyBytesKernel(const unsigned char* __restrict__ source,
                                unsigned char* __restrict__ destination, size_t count) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (((reinterpret_cast<uintptr_t>(source) | reinterpret_cast<uintptr_t>(destination)) & 0xF) == 0) {
    const size_t vectors = count >> 4;
    const auto* source_vectors = reinterpret_cast<const uint4*>(source);
    auto* destination_vectors = reinterpret_cast<uint4*>(destination);
    for (size_t i = start; i < vectors; i += stride) destination_vectors[i] = source_vectors[i];
    for (size_t i = (vectors << 4) + start; i < count; i += stride) destination[i] = source[i];
    return;
  }
  for (size_t i = start; i < count; i += stride) destination[i] = source[i];
}

__global__ void ZeroBytesKernel(unsigned char* __restrict__ destination, size_t count) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if ((reinterpret_cast<uintptr_t>(destination) & 0xF) == 0) {
    const size_t vectors = count >> 4;
    auto* destination_vectors = reinterpret_cast<uint4*>(destination);
    for (size_t i = start; i < vectors; i += stride) destination_vectors[i] = make_uint4(0, 0, 0, 0);
    for (size_t i = (vectors << 4) + start; i < count; i += stride) destination[i] = 0;
    return;
  }
  for (size_t i = start; i < count; i += stride) destination[i] = 0;
}

__global__ void GatherStridedInt32Kernel(const int32_t* __restrict__ source, int stride,
                                         int32_t* __restrict__ destination, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) destination[i] = source[static_cast<size_t>(i) * stride];
}

unsigned CopyBlocks(size_t count) {
  constexpr size_t kBytesPerBlock = 256 * 16 * 4;
  return static_cast<unsigned>(std::min<size_t>(1024, (count + kBytesPerBlock - 1) / kBytesPerBlock));
}

}  // namespace small_copy

void LaunchStoreBytes(void* destination, const void* source, size_t count, cudaStream_t stream) {
  if (count == 0) return;
  if (count <= 64) {
    small_copy::LaunchStoreBytesImpl<64>(destination, source, count, stream);
  } else if (count <= 512) {
    small_copy::LaunchStoreBytesImpl<512>(destination, source, count, stream);
  } else if (count <= kMaxStoreBytes) {
    small_copy::LaunchStoreBytesImpl<kMaxStoreBytes>(destination, source, count, stream);
  } else {
    throw std::invalid_argument("LaunchStoreBytes payload exceeds kMaxStoreBytes.");
  }
}

void LaunchCopyBytes(void* destination, const void* source, size_t count, cudaStream_t stream) {
  if (count == 0) return;
  small_copy::CopyBytesKernel<<<small_copy::CopyBlocks(count), 256, 0, stream>>>(
      static_cast<const unsigned char*>(source), static_cast<unsigned char*>(destination), count);
  CUDA_CHECK_LAUNCH();
}

void LaunchZeroBytes(void* destination, size_t count, cudaStream_t stream) {
  if (count == 0) return;
  small_copy::ZeroBytesKernel<<<small_copy::CopyBlocks(count), 256, 0, stream>>>(
      static_cast<unsigned char*>(destination), count);
  CUDA_CHECK_LAUNCH();
}

void LaunchGatherStridedInt32(const int32_t* source, int stride, int32_t* destination, int count,
                              cudaStream_t stream) {
  if (count <= 0) return;
  small_copy::GatherStridedInt32Kernel<<<(count + 255) / 256, 256, 0, stream>>>(
      source, stride, destination, count);
  CUDA_CHECK_LAUNCH();
}

}  // namespace cuda
}  // namespace Generators
