// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
namespace Generators {

namespace cuda {

template <typename T>
void Launch_UpdatePositionIds(T* positions, int batch_beam_size, int total_length, int new_kv_length, cudaStream_t stream);
template <typename T>
void Launch_UpdateAttentionMask(T* mask_data, T* old_data, int batch_beam_size, int new_kv_length, int total_length, int max_length, bool update_only, cudaStream_t stream);

void LaunchAddLogitsMask(float* batch_logits, int batch_beam_size, int vocab_size, const uint32_t* logits_mask, cudaStream_t stream);
void LaunchFp16ToFp32(const uint16_t* fp16, float* fp32, int count, cudaStream_t stream);
void LaunchBf16ToFp32(const uint16_t* bf16, float* fp32, int count, cudaStream_t stream);
void LaunchFp32ToFp16(const float* fp32, uint16_t* fp16, int count, cudaStream_t stream);
void LaunchInt32ToInt64(const int32_t* src, int64_t* dst, int count, cudaStream_t stream);

template <typename T>
void BufferExpansionKernelLauncher(const T* input, T* output, int batch_size, int beam_width, int chunk_size, cudaStream_t stream);

void ReorderPastStatesKernelLauncher(void* out_buffer, const void* in_buffer, int batch_size, int num_heads,
                                     int max_length, int head_size, int chunk_size, cudaStream_t stream);

void UpdateCacheIndirectionKernelLauncher(int32_t* tgt_indir_cache,
                                          const int32_t* src_indir_cache,
                                          const int32_t* beam_ids,
                                          int batch_size,
                                          int beam_width,
                                          int input_seq_length,
                                          int max_seq_length,
                                          int current_length,
                                          cudaStream_t stream);

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
                                       int sequence_length);

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
                           const int* cache_indir_data);

// Copies window slot `src_slot` onto slot `dst_slot` for each of `count` state tensors described
// by `descs` (device memory, {base, slot_bytes} pairs). One launch replaces `count` memcpys.
void LaunchCopyStateSlots(const void* descs, int count, int src_slot, int dst_slot, cudaStream_t stream);

// Replays compact state updates for `fast_count + generic_count` descriptors in `descs` (device
// memory). The first `fast_count` are gated-delta-net descriptors eligible for the fast kernel (see
// kMaxFastReplayTransitions and kMaxFastReplayKeyWidth; 16-byte aligned states and a key width
// divisible by 4), launched with `fast_blocks_per_descriptor` blocks each (the largest
// ReplayGatedDeltaNetBlocks over them). The rest take the generic kernel.
constexpr int kMaxFastReplayTransitions = 8;
constexpr int kMaxFastReplayKeyWidth = 256;
void LaunchReplayStateUpdates(const void* descs, int fast_count, int generic_count,
                              int fast_blocks_per_descriptor, cudaStream_t stream);
int ReplayGatedDeltaNetBlocks(int heads, int value_width);

// Small copies as kernels. On WDDM every switch between a copy-engine operation (cudaMemcpyAsync,
// cudaMemsetAsync) and a kernel on the same stream costs tens of microseconds of device idle time,
// which dominates the few bytes these moves carry. Keeping them on the compute engine avoids it.
//
// Largest host payload LaunchStoreBytes accepts. Kernel parameters are limited to 4 KiB below
// Volta, and the payload shares that space with the other arguments.
constexpr size_t kMaxStoreBytes = 4032;
// Writes `count` bytes read from host memory `source` at launch time to device memory
// `destination`. `source` may be reused as soon as the call returns.
void LaunchStoreBytes(void* destination, const void* source, size_t count, cudaStream_t stream);
// Copies `count` bytes between any two addresses the device can access, including pinned host
// memory mapped through unified addressing.
void LaunchCopyBytes(void* destination, const void* source, size_t count, cudaStream_t stream);
void LaunchZeroBytes(void* destination, size_t count, cudaStream_t stream);
// destination[i] = source[i * stride] for i < count; either side may be mapped pinned host memory.
void LaunchGatherStridedInt32(const int32_t* source, int stride, int32_t* destination, int count,
                              cudaStream_t stream);

}  // namespace cuda
}  // namespace Generators
