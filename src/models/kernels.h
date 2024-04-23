// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
namespace Generators {

namespace cuda {

template <typename T>
void Launch_UpdatePositionIds(T* positions, int batch_beam_size, cudaStream_t stream);
template <typename T>
void Launch_UpdateAttentionMask(T* mask_data, const T* old_mask_data, int batch_beam_size, int current_length,
                                int max_length, bool update_only, cudaStream_t stream);

void LaunchHandleEOSArray(float* batch_logits, int batch_beam_size, int vocab_size, const int32_t* eos_token_ids, int eos_token_ids_count, cudaStream_t stream);

void LaunchFp16ToFp32(const uint16_t* fp16, float* fp32, int count, cudaStream_t stream);
void LaunchInt32ToInt64(const int32_t* src, int64_t* dst, int count, cudaStream_t stream);
}  // namespace cuda

}  // namespace Generators
