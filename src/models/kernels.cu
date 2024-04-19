#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace Generators {
namespace cuda {

template <typename T> __global__ void UpdatePositionIds(T *positions, int batch_beam_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_beam_size)
        positions[i]++;
}

template <typename T> void Launch_UpdatePositionIds(T *positions, int batch_beam_size, cudaStream_t stream) {
    UpdatePositionIds<T><<<(batch_beam_size + 255) / 256, 256, 0, stream>>>(positions, batch_beam_size);
}

template void Launch_UpdatePositionIds(int32_t *positions, int batch_beam_size, cudaStream_t stream);
template void Launch_UpdatePositionIds(int64_t *positions, int batch_beam_size, cudaStream_t stream);

template <typename T>
__global__ void CopyAndUpdateAttentionMask(T *mask_data, const T *old_mask_data, int batch_beam_size,
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
__global__ void UpdateAttentionMask(T *mask_data, int batch_beam_size, int current_length, int max_length) {
    int i = blockIdx.x;
    if (i < batch_beam_size) {
        mask_data[i * max_length + current_length] = 1;
    }
}

template <typename T>
void Launch_UpdateAttentionMask(T *mask_data, const T *old_mask_data, int batch_beam_size, int current_length,
                                int max_length, bool update_only, cudaStream_t stream) {
    if (update_only) {
        UpdateAttentionMask<T>
            <<<batch_beam_size, 1, 0, stream>>>(mask_data, batch_beam_size, current_length, max_length);
    } else {
        CopyAndUpdateAttentionMask<T><<<(batch_beam_size * max_length + 255) / 256, 256, 0, stream>>>(
            mask_data, old_mask_data, batch_beam_size, current_length, max_length);
    }
}

template void Launch_UpdateAttentionMask(int32_t *mask_data, const int32_t *old_mask_data, int batch_beam_size,
                                         int current_length, int max_length, bool update_only, cudaStream_t stream);
template void Launch_UpdateAttentionMask(int64_t *mask_data, const int64_t *old_mask_data, int batch_beam_size,
                                         int current_length, int max_length, bool update_only, cudaStream_t stream);

__global__ void ConvertFp16ToFp32(const half *src, float *dst, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count)
        dst[idx] = __half2float(src[idx]);
}

void LaunchFp16ToFp32(const uint16_t *fp16, float *fp32, int count, cudaStream_t stream) {
    int block_size = 256;
    int num_blocks = (count + block_size - 1) / block_size;
    ConvertFp16ToFp32<<<num_blocks, block_size, 0, stream>>>(reinterpret_cast<const half *>(fp16), fp32, count);
}

__global__ void ConvertInt32ToInt64(const int32_t *src, int64_t *dst, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        dst[idx] = src[idx];
    }
}

void LaunchInt32ToInt64(const int32_t *src, int64_t *dst, int count, cudaStream_t stream) {
    int block_size = 256;
    int num_blocks = (count + block_size - 1) / block_size;
    ConvertInt32ToInt64<<<num_blocks, block_size, 0, stream>>>(src, dst, count);
}

} // namespace cuda
} // namespace Generators
