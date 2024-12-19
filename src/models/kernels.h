// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
namespace Generators {

namespace cuda {

template <typename T>
void Launch_UpdatePositionIds(T* positions, int batch_beam_size, int total_length, int new_kv_length, cudaStream_t stream);
template <typename T>
void Launch_UpdateAttentionMask(T* mask_data, const T* old_data, int batch_beam_size, int new_kv_length, int total_length, int max_length, bool update_only, cudaStream_t stream);

void LaunchHandleEOSArray(float* batch_logits, int batch_beam_size, int vocab_size, const int32_t* eos_token_ids, int eos_token_ids_count, cudaStream_t stream);
void LaunchAddLogitsMask(float* batch_logits, int batch_beam_size, int vocab_size, const uint32_t* logits_mask, cudaStream_t stream);

void LaunchFp16ToFp32(const uint16_t* fp16, float* fp32, int count, cudaStream_t stream);
void LaunchFp32ToFp16(const float* fp32, uint16_t* fp16, int count, cudaStream_t stream);
void LaunchInt32ToInt64(const int32_t* src, int64_t* dst, int count, cudaStream_t stream);
void LaunchExpandAndInt32ToInt64(const int32_t* src, int64_t* dst, int num_beams, int batch_size, int sequence_length, cudaStream_t stream);
void LaunchExpand(const int32_t* src, int32_t* dst, int num_beams, int batch_size, int sequence_length, cudaStream_t stream);

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
                                       T** qk_layer_pointers,
                                       int token_index,
                                       int batch_beam_size,
                                       int num_layers,
                                       int num_heads,
                                       int num_alignment_heads,
                                       const int* alignment_heads,
                                       int frames,
                                       int max_length);

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
                           const int* cache_indir_data);

}  // namespace cuda
}  // namespace Generators
