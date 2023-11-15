#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace Generators {
namespace cuda {

__global__ void Gpt_InitAttentionMask(int32_t* mask_data, int32_t* position_data, int32_t* sequence_lengths, const int32_t* input_ids,
  int batch_size, int num_beams, int sequence_length, int pad_token_id)
{
  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
  const int32_t* word_id = input_ids;
  bool init_mask = mask_data;
  int32_t* mask = mask_data;
  int32_t* position = position_data;
  for (int i = 0; i < batch_size; i++) {
    int32_t abs_position = 0;
    for (int j = 0; j < sequence_length; j++, word_id++, mask++, position++) {
      if (*word_id == pad_token_id) {
        if (init_mask)
          *mask = 0;
        *position = 0;
      } else {
        if (init_mask)
          *mask = 1;
        *position = abs_position;
        abs_position++;
      }
    }
    for (int k = 0; k < num_beams; k++) {
      sequence_lengths[i * num_beams + k] = abs_position;
    }
  }
}

void LaunchGpt_InitAttentionMask(int32_t * mask_data, int32_t * position_data, int32_t* sequence_lengths, const int32_t* input_ids, 
  int batch_size, int num_beams, int sequence_length, int pad_token_id, cudaStream_t stream) {
    Gpt_InitAttentionMask<<<1, 1, 0, stream>>>(mask_data, position_data, sequence_lengths, input_ids, batch_size, num_beams, sequence_length, pad_token_id);
}

__global__ void Gpt_UpdatePositionIds(int32_t* positions, int batch_beam_size, int current_length) {
    for (int i = 0; i < batch_beam_size; i++) {
      positions[i] = current_length - 1;
      }
}

void LaunchGpt_UpdatePositionIds(int32_t* positions, int batch_beam_size, int current_length, cudaStream_t stream) {
      Gpt_UpdatePositionIds<<<1, 1, 0, stream>>>(positions, batch_beam_size, current_length);
}

__global__ void Gpt_UpdateMask(int32_t* mask_data, const int32_t* old_mask_data, int batch_beam_size, int current_length) {
  for (int i = 0; i < batch_beam_size; i++) {
        for (int j = 0; j < current_length - 1; j++) {
        mask_data[i * current_length + j] = old_mask_data[i * (current_length - 1) + j];
        }
        mask_data[i * current_length + current_length - 1] = 1;
  }
}

void LaunchGpt_UpdateMask(int32_t* mask_data, const int32_t* old_mask_data, int batch_beam_size, int current_length, cudaStream_t stream) {
  Gpt_UpdateMask<<<1, 1, 0, stream>>>(mask_data, old_mask_data, batch_beam_size, current_length);
}

__global__ void Gpt_InitAttentionMask(int64_t* mask_data, int64_t* position_data, int32_t* sequence_lengths, const int64_t* input_ids,
  int batch_size, int num_beams, int sequence_length, int pad_token_id)
{
  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
  const int64_t* word_id = input_ids;
  bool init_mask = mask_data;
  int64_t* mask = mask_data;
  int64_t* position = position_data;
  for (int i = 0; i < batch_size; i++) {
    int64_t abs_position = 0;
    for (int j = 0; j < sequence_length; j++, word_id++, mask++, position++) {
      if (*word_id == pad_token_id) {
        if (init_mask)
          *mask = 0;
        *position = 0;
      } else {
        if (init_mask)
          *mask = 1;
        *position = abs_position;
        abs_position++;
      }
    }
    for (int k = 0; k < num_beams; k++) {
      sequence_lengths[i * num_beams + k] = abs_position;
    }
  }
}

void LaunchGpt_InitAttentionMask(int64_t * mask_data, int64_t * position_data, int32_t* sequence_lengths, const int64_t* input_ids, 
  int batch_size, int num_beams, int sequence_length, int pad_token_id, cudaStream_t stream) {
  Gpt_InitAttentionMask<<<1, 1, 0, stream>>>(mask_data, position_data, sequence_lengths, input_ids, batch_size, num_beams, sequence_length, pad_token_id);
}

__global__ void Gpt_UpdatePositionIds(int64_t* positions, int batch_beam_size, int current_length) {
  for (int i = 0; i < batch_beam_size; i++) {
    positions[i] = current_length - 1;
  }
}

void LaunchGpt_UpdatePositionIds(int64_t* positions, int batch_beam_size, int current_length, cudaStream_t stream) {
  Gpt_UpdatePositionIds<<<1, 1, 0, stream>>>(positions, batch_beam_size, current_length);
}

__global__ void Gpt_UpdateMask(int64_t* mask_data, const int64_t* old_mask_data, int batch_beam_size, int current_length) {
  for (int i = 0; i < batch_beam_size; i++) {
    for (int j = 0; j < current_length - 1; j++) {
      mask_data[i * current_length + j] = old_mask_data[i * (current_length - 1) + j];
    }
    mask_data[i * current_length + current_length - 1] = 1;
  }
}

void LaunchGpt_UpdateMask(int64_t* mask_data, const int64_t* old_mask_data, int batch_beam_size, int current_length, cudaStream_t stream) {
  Gpt_UpdateMask<<<1, 1, 0, stream>>>(mask_data, old_mask_data, batch_beam_size, current_length);
}

__global__ void ConvertFp16ToFp32(const half* src, float* dst, int count) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < count) {
    dst[idx] = __half2float(src[idx]);
  }
}

void LaunchFp16ToFp32(const uint16_t* fp16, float* fp32, int count, cudaStream_t stream) {
  int block_size = 256;
  int num_blocks = (count + block_size - 1) / block_size;
  ConvertFp16ToFp32<<<num_blocks, block_size, 0, stream>>>(reinterpret_cast<const half*>(fp16), fp32, count);
}

}
}
