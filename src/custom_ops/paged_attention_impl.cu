#include "paged_attention_impl.h"
#include "utils.cuh"
#include "device_prop.cuh"
#ifdef OCOS_USE_FLASH_ATTENTION
#include "attention_lib/flash_attention/flash_api.h"
#endif
#ifdef OCOS_USE_MEMORY_EFFICIENT_ATTENTION
#include "attention_lib/cutlass_fmha/memory_efficient_attention.h"
#endif
#include <vector>
#include <cassert>

namespace cuda {

namespace vllm {

template <typename scalar_t>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,      // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,    // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key_cache,      // [num_blocks, block_size, num_heads, head_size]
    scalar_t* __restrict__ value_cache,    // [num_blocks, block_size, num_heads, head_size]
    const int* __restrict__ slot_mapping,  // [num_tokens]
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size) {
  const int token_idx = blockIdx.x;
  const int slot_idx = slot_mapping[token_idx];
  const int block_idx = slot_idx / block_size;
  const int block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int src_key_idx = token_idx * key_stride + i;
    const int src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;

    const int tgt_value_idx = block_idx * num_heads * head_size * block_size + block_offset * num_heads * head_size + head_idx * head_size + head_offset;
    const int tgt_key_idx = tgt_value_idx;
    //{
    //  if (key_cache[tgt_key_idx] - key[src_key_idx] > half(0.1)) {
    //    printf("key error find, %d,%d ", tgt_key_idx, src_key_idx);
    //  }
    //  if (value_cache[tgt_value_idx] - value[src_value_idx] > half(0.1)) {
    //    printf("key error find, %d %d", tgt_value_idx, src_value_idx);
    //  }
    //}
    key_cache[tgt_key_idx] = __ldg(&key[src_key_idx]);
    value_cache[tgt_value_idx] = __ldg(&value[src_value_idx]);
  }
}

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
    scalar_t* __restrict__ arr,
    const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr,
    int rot_offset,
    int embed_dim) {
  int x_index, y_index;
  scalar_t cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = __ldg(cos_ptr + x_index);
    sin = __ldg(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = __ldg(cos_ptr + x_index / 2);
    sin = __ldg(sin_ptr + x_index / 2);
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template <typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
    const int32_t* __restrict__ positions,       // [batch_size, seq_len] or [num_tokens]
    scalar_t* __restrict__ query,                // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key,                  // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim // 2]
    const int rot_dim,
    const int query_stride,
    const int key_stride,
    const int num_heads,
    const int num_kv_heads,
    const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int32_t pos = positions[token_idx];
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int token_head = token_idx * query_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    apply_rotary_embedding<scalar_t, IS_NEOX>(query + token_head, cos_ptr,
                                              sin_ptr, rot_offset, embed_dim);
  }

  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int token_head = token_idx * key_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    apply_rotary_embedding<scalar_t, IS_NEOX>(key + token_head, cos_ptr,
                                              sin_ptr, rot_offset, embed_dim);
  }
}
}   // namespace vllm

void rotary_embedding_neox(
    const cudaStream_t stream,
    const int32_t* positions,  // [num_tokens]
    void* query,               // [num_tokens, num_heads * head_size]
    void* key,                 // [num_tokens, num_kv_heads * head_size]
    int head_size,
    const void* cos_sin_cache,  // [max_position, rot_dim]
    int num_tokens,
    int rot_dim,
    int num_heads,
    int num_kv_heads) {
  const bool is_neox = true;
  int query_stride = num_heads * head_size;
  int key_stride = num_kv_heads * head_size;
  // TORCH_CHECK(stride == key.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * rot_dim / 2, 512));

  // half
  using scalar_t = half;
  if (is_neox) {
    vllm::rotary_embedding_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
        positions,
        static_cast<scalar_t*>(query),
        static_cast<scalar_t*>(key),
        static_cast<const scalar_t*>(cos_sin_cache),
        rot_dim,
        query_stride,
        key_stride,
        num_heads,
        num_kv_heads,
        head_size);
  } else {
    vllm::rotary_embedding_kernel<scalar_t, false><<<grid, block, 0, stream>>>(
        positions,
        static_cast<scalar_t*>(query),
        static_cast<scalar_t*>(key),
        static_cast<const scalar_t*>(cos_sin_cache),
        rot_dim,
        query_stride,
        key_stride,
        num_heads,
        num_kv_heads,
        head_size);
  }
}

void reshape_and_cache(
    const cudaStream_t stream,
    const void* key,          // [num_tokens, num_heads, head_size]
    const void* value,        // [num_tokens, num_heads, head_size]
    const void* key_cache,    // [num_blocks, block_size, num_heads, head_size]
    const void* value_cache,  // [num_blocks, block_size, num_heads, head_size]
    const int* slot_mapping,  // [num_tokens]
    const int32_t* key_shapes,
    const int32_t* value_shapes,
    const int64_t block_size) {
  int num_tokens = key_shapes[0];
  int num_heads = key_shapes[1];
  int head_size = key_shapes[2];
  // int block_size = key_cache.size(3);

  int key_stride = key_shapes[1] * key_shapes[2];
  int value_stride = value_shapes[1] * value_shapes[2];

  // static_assert(std::is_same_v<T, MLFloat16>, "Unsupported data type: ");

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
 
  vllm::reshape_and_cache_kernel<half><<<grid, block, 0, stream>>>(
      (const half*)key,
      (const half*)value,
      (half*)key_cache,
      (half*)value_cache,
      slot_mapping,
      key_stride,
      value_stride,
      num_heads,
      head_size,
      block_size);
}

#if OCOS_USE_FLASH_ATTENTION
template <typename T>
OrtStatusPtr FlashAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    PackedAttentionParameters& parameters,
    PackedMultiHeadAttentionData<T>& data) {
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int num_kv_heads = parameters.num_kv_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;

  // Q, K and V pointers
  const int model_dimension_qk = num_heads * qk_head_size;
  const int model_dimension_v = num_kv_heads * v_head_size;
  const size_t elements_qk = static_cast<size_t>(parameters.token_count) * static_cast<size_t>(model_dimension_qk);
  const size_t elements_v = static_cast<size_t>(parameters.token_count) * static_cast<size_t>(model_dimension_v);

  // When separated Q, K, V is used, we can directly use them in Cutlass FMHA. Otherwise, transpose BSN3H to 3BSNH
  // TODO(leca): 
//  if (!data.no_qkv_workspace) {
//    LaunchTranspose(data.query, data.key, data.value, data.bias, data.workspace,
//                    batch_size, sequence_length,
//                    num_heads, qk_head_size, v_head_size,
//                    data.source_qkv_format, AttentionQkvFormat::Q_K_V_TNH,
//                    data.token_offset, parameters.token_count, stream);
//  }

  float scale = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(qk_head_size))
                                         : parameters.scale;
  int32_t* cu_seqlens_q = const_cast<int32_t*>(data.cumulative_sequence_length);
  int32_t* cu_seqlens_k = const_cast<int32_t*>(data.cumulative_sequence_length);
  const void* query = data.no_qkv_workspace ? data.query : data.workspace;
  const void* key = data.no_qkv_workspace ? data.key : (data.workspace + elements_qk);
  const void* value = data.no_qkv_workspace ? data.value : (data.workspace + elements_qk + elements_qk);
  void* softmax_lse_buffer = data.no_qkv_workspace
                                 ? data.workspace
                                 : (data.workspace + elements_qk + elements_v + elements_v);

  ORTX_RETURN_IF_ERROR(
      flash::mha_varlen_fwd(
          device_prop,
          stream,
          const_cast<void*>(query),
          const_cast<void*>(key),
          const_cast<void*>(value),
          data.output,
          cu_seqlens_q,
          cu_seqlens_k,
          softmax_lse_buffer,
          batch_size,
          num_heads,
          num_kv_heads,  // num_heads_k
          qk_head_size,
          sequence_length,
          sequence_length,
          scale,
          parameters.causal,  // is causal
          false    // is_bf16 TODO(leca)
          ));

  return nullptr;
}
#endif

template <typename T>
OrtStatusPtr QkvToContext(
    cudaStream_t stream,
    PackedAttentionParameters& parameters,
    PackedMultiHeadAttentionData<T>& data) {
  const cudaDeviceProp& device_prop = DeviceProp::GetCudaDeviceProp();
#if OCOS_USE_FLASH_ATTENTION
  return FlashAttention(device_prop, stream, parameters, data);
#endif
#if OCOS_USE_MEMORY_EFFICIENT_ATTENTION
  // TODO(leca):
  //return FusedAttentionCutlass(device_prop, stream, parameters, data);
#endif
  return nullptr;
}

//template OrtStatusPtr QkvToContext<BFloat16>(
//    cudaStream_t stream,
//    PackedAttentionParameters& parameters,
//    PackedMultiHeadAttentionData<BFloat16>& data);

template OrtStatusPtr QkvToContext<half>(
    cudaStream_t stream,
    PackedAttentionParameters& parameters,
    PackedMultiHeadAttentionData<half>& data);

constexpr size_t kCUDAMemoryAlignment = 256;

size_t GetAttentionScratchSize(
    size_t element_size,
    size_t batch_size,
    size_t num_heads,
    size_t sequence_length) {
  const size_t bytes = element_size * batch_size * num_heads * sequence_length * sequence_length;
  return ((bytes + kCUDAMemoryAlignment - 1) / kCUDAMemoryAlignment) * kCUDAMemoryAlignment;
}

size_t GetAttentionWorkspaceSize(
    size_t element_size,
    size_t batch_size,
    size_t num_heads,
    size_t qk_head_size,
    size_t v_head_size,
    size_t sequence_length,
    void* fused_runner,
    bool use_flash_attention,
    bool use_memory_efficient_attention,
    bool no_qkv_workspace) {
  // Note that q, k and v might need alignment for fused attention kernels.
  const size_t qkv_bytes = no_qkv_workspace ? 0 : (element_size * batch_size * num_heads * sequence_length * (qk_head_size + qk_head_size + v_head_size));

  // Use portion of workspace for softmax buffer.
  if (use_flash_attention) {
    size_t flash_buffer_bytes = flash::get_softmax_lse_size(sequence_length, batch_size, num_heads);
    return qkv_bytes + flash_buffer_bytes;
  }

  if (fused_runner != nullptr) {
    return qkv_bytes;
  }

//#if USE_MEMORY_EFFICIENT_ATTENTION
//  if (use_memory_efficient_attention) {
//    size_t fmha_buffer_bytes = 0;
//    if (MemoryEfficientAttentionParams::need_workspace(v_head_size, element_size == sizeof(float))) {
//      fmha_buffer_bytes = batch_size * sequence_length * num_heads * v_head_size * sizeof(float);
//    }
//    return qkv_bytes + fmha_buffer_bytes;
//  }
//#else
//  ORT_UNUSED_PARAMETER(use_memory_efficient_attention);
//#endif

  return qkv_bytes + 2 * GetAttentionScratchSize(element_size, batch_size, num_heads, sequence_length);
}

}   // namespace cuda