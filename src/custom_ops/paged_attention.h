// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ocos.h"
#include "cuda_type.h"
#include "paged_attention_impl.h"
#include "device_prop.cuh"
#ifdef OCOS_USE_FLASH_ATTENTION
#include "attention_lib/flash_attention/flash_api.h"
#endif

template <typename T>
using UniquePtrWithDeletor = std::unique_ptr<T, std::function<void(T*)>>;

template <typename T>
inline UniquePtrWithDeletor<T> GetScratchBuffer(void* p, OrtAllocator* allocator) {
  return UniquePtrWithDeletor<T>{static_cast<T*>(p), [allocator = std::move(allocator)](T* p) {
                                  allocator->Free(allocator, p);
                                }};
}

template <typename T>
OrtStatusPtr CheckInputs(const cudaStream_t stream, OrtAllocator* allocator, const ortc::Tensor<T>& query, const ortc::Tensor<int32_t>& context_lens, 
                         int32_t num_heads, int32_t num_kv_heads, int32_t head_size, float scale, bool prompt_mode, PackedAttentionParameters& parameters) {
  const std::vector<int64_t>& query_shape = query.Shape();
  if (query_shape.size() < 2 || query_shape.size() > 3) {
    return OrtW::CreateStatus(MakeString("Invalid query shape, expect 2 or 3 dimensions"), ORT_INVALID_ARGUMENT);
  }
  if (query_shape.back() != num_heads * head_size) {
    return OrtW::CreateStatus(MakeString("Hidden size should equal to num_heads_ * head_size_"), ORT_INVALID_ARGUMENT);
  }

  parameters.batch_size = context_lens.NumberOfElement();
  parameters.sequence_length = 1;
  parameters.token_count = 0;
  parameters.valid_token_count = query_shape[0];
  parameters.causal = true;
  parameters.head_size = head_size;
  parameters.num_heads = num_heads;
  parameters.num_kv_heads = num_kv_heads;
  parameters.scale = scale;
  parameters.hidden_size = static_cast<int>(head_size * num_heads);
  parameters.v_hidden_size = static_cast<int>(head_size * num_kv_heads);
  parameters.v_head_size = static_cast<int>(parameters.head_size);
  return nullptr;
}

template<typename T>
struct PagedAttention {
  static OrtMemType GetInputMemoryType(size_t input_index) {
    if (input_index == 7 || input_index == 8) return OrtMemType::OrtMemTypeCPUInput;  // make context_lens and is_prompt CPU input
    return OrtMemType::OrtMemTypeDefault;
  }

  using TT = typename contrib::CudaT<T>::MappedType;
  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info) {
    int64_t num_heads = 0, head_size = 0;
    ORTX_RETURN_IF_ERROR(api.KernelInfoGetAttribute_int64(&info, "num_heads", &num_heads));
    assert(num_heads > 0);
    num_heads_ = static_cast<int32_t>(num_heads);
    num_kv_heads_ = static_cast<int32_t>(OrtW::GetOpAttributeOrDefault<int64_t>(info, "num_kv_heads", num_heads));
    
    ORTX_RETURN_IF_ERROR(api.KernelInfoGetAttribute_int64(&info, "head_size", &head_size));
    assert(head_size > 0);
    head_size_ = static_cast<int32_t>(head_size);

    ORTX_RETURN_IF_ERROR(api.KernelInfoGetAttribute_float(&info, "scale", &scale_));
    assert(scale_ >= 0);

    num_queries_per_kv_ = num_heads_ / num_kv_heads_;
    OrtAllocator* allocator = nullptr;
    ORTX_RETURN_IF_ERROR(api.KernelInfoGetAllocator(&info, OrtMemType::OrtMemTypeDefault, &allocator));
    allocator_ = UniquePtrWithDeletor<OrtAllocator>{allocator, [&api](OrtAllocator* p){api.ReleaseAllocator(p);}};
    return nullptr;
  }

  OrtStatusPtr RunMultiHeadAttention(Ort::Custom::CUDAKernelContext* ctx, const ortc::Tensor<T>& query, const ortc::Tensor<T>& key, const ortc::Tensor<T>& value,
                                     T* output, PackedAttentionParameters& parameters, const int32_t* seqinfo) const {
    PackedMultiHeadAttentionData<TT> data;
    data.use_flash_attention = false; 
    data.use_memory_efficient_attention = false;
#if OCOS_USE_FLASH_ATTENTION
    data.use_flash_attention = true;
#endif
#if OCOS_USE_MEMORY_EFFICIENT_ATTENTION
    data.use_memory_efficient_attention = true;
#endif
    data.query = reinterpret_cast<const TT*>(query.DataRaw());
    data.key = reinterpret_cast<const TT*>(key.DataRaw());
    data.value = reinterpret_cast<const TT*>(value.DataRaw());

    // TODO(leca):
//    // broadcast key,value for GQA
//    TensorShape key_shape({parameters.valid_token_count, parameters.num_kv_heads, parameters.head_size});
//    size_t kv_repeat_space = key_shape.Size() * (num_queries_per_kv_ > 0 ? num_queries_per_kv_ : 0);
//    IAllocatorUniquePtr<CudaT> key_out = GetScratchBuffer<CudaT>(kv_repeat_space, context->GetComputeStream());
//    IAllocatorUniquePtr<CudaT> value_out = GetScratchBuffer<CudaT>(kv_repeat_space, context->GetComputeStream());
//    if (num_queries_per_kv_ > 1 && !ParseEnvironmentVariableWithDefault<bool>("repeat_kv_tile", false)) {
//      // repeat key and value
//      LaunchRepeatKeyValue<CudaT>(Stream(context), key_out.get(), value_out.get(),
//                                  data.key, data.value, key_shape.GetDims().data(), num_queries_per_kv_);
//      CHECK_CUDA_ERROR();
//      data.key = key_out.get();
//      data.value = value_out.get();
//      parameters.num_kv_heads = parameters.num_heads;
//      DumpTensor(Stream(context), data.key, "repeat_key", kv_repeat_space * sizeof(CudaT));
//    }

    size_t workSpaceSize = cuda::GetAttentionWorkspaceSize(sizeof(T), parameters.batch_size, parameters.num_heads, parameters.head_size, parameters.v_head_size,
                                                           parameters.sequence_length, nullptr, data.use_flash_attention, data.use_memory_efficient_attention, true);
    UniquePtrWithDeletor<T> workspace_unique = GetScratchBuffer<T>(allocator_->Alloc(allocator_.get(), workSpaceSize), allocator_.get());
    data.workspace = reinterpret_cast<TT*>(workspace_unique.get());
    data.cumulative_sequence_length = seqinfo;
    data.output = reinterpret_cast<TT*>(output);
    data.fused_runner = nullptr;
    data.no_qkv_workspace = data.fused_runner == nullptr || data.use_flash_attention || data.use_memory_efficient_attention;
    data.source_qkv_format = data.key == nullptr ? AttentionQkvFormat::QKV_TN3H : AttentionQkvFormat::Q_K_V_TNH;
    return cuda::QkvToContext<TT>(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()), parameters, data);
  }

  OrtStatusPtr Compute(Ort::Custom::CUDAKernelContext* ctx, const ortc::Tensor<T>& query, const ortc::Tensor<T>& key,
                       const ortc::Tensor<T>& value, const ortc::Tensor<T>& key_cache, const ortc::Tensor<T>& value_cache,
                       const ortc::Tensor<int32_t>& block_tables, const ortc::Tensor<int32_t>& slot_mappings, 
                       const ortc::Tensor<int32_t>& context_lens, const ortc::Tensor<int32_t>& is_prompt,
                       std::optional<const ortc::Tensor<T>*> cos_sin_cache,
                       std::optional<const ortc::Tensor<int32_t>*> positions, ortc::Tensor<T>& attn_out) const {
    bool prompt_mode = *(is_prompt.Data()) == 1;
    PackedAttentionParameters parameters;
    ORTX_RETURN_IF_ERROR(CheckInputs<T>(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()), allocator_.get(), query, 
                         context_lens, num_heads_, num_kv_heads_, head_size_, scale_, prompt_mode, parameters));

    UniquePtrWithDeletor<int32_t> seqinfo;
    UniquePtrWithDeletor<int32_t> position_ids;
    if (prompt_mode) {
        parameters.token_count = parameters.valid_token_count;
  
        std::vector<int32_t> seqstart(context_lens.NumberOfElement() + 1, 0);
        for (int64_t i = 0; i < context_lens.NumberOfElement(); i++) {
          int32_t seqlen_i = *(context_lens.Data()+i);
          if (seqlen_i > parameters.sequence_length) parameters.sequence_length = seqlen_i;
          seqstart[i+1] = seqstart[i] + seqlen_i;
        }
        seqinfo = GetScratchBuffer<int32_t>(allocator_.get()->Alloc(allocator_.get(), seqstart.size() * sizeof(int32_t)), allocator_.get());
        cudaMemcpy(seqinfo.get(), seqstart.data(), seqstart.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    } else {
        seqinfo = GetScratchBuffer<int32_t>(allocator_.get()->Alloc(allocator_.get(), context_lens.SizeInBytes()), allocator_.get());
        cudaMemcpy(seqinfo.get(), context_lens.DataRaw(), context_lens.SizeInBytes(), cudaMemcpyHostToDevice);
    }
  
    if (cos_sin_cache.has_value() && !positions.has_value()) {
      std::vector<int32_t> position_ids_host;
      if (prompt_mode) {
        for (int64_t i = 0; i < context_lens.NumberOfElement(); i++) {
          int32_t seqlen_i = *(context_lens.Data()+i);
          if (seqlen_i == 0) continue;
          std::vector<int32_t> position_id(seqlen_i);
          std::iota(position_id.begin(), position_id.end(), 0);   // fill position_id with [0, 1, 2, ...seqlen_i)
          position_ids_host.insert(position_ids_host.end(), position_id.begin(), position_id.end());
        }
      } else position_ids_host.assign(parameters.batch_size, 0);  // TODO(leca): Does decoding case support seqlen_knew > 1?
    
      position_ids = GetScratchBuffer<int32_t>(allocator_.get()->Alloc(allocator_.get(), position_ids_host.size() * sizeof(int32_t)), allocator_.get());
      cudaMemcpy(position_ids.get(), position_ids_host.data(), position_ids_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    const std::vector<int64_t>& query_shape = query.Shape();
    T* output_data = attn_out.Allocate(query_shape);

    if (cos_sin_cache.has_value()) {
      int64_t rot_dim = (*cos_sin_cache)->Shape()[1];
      assert(rot_dim == head_size_);
      cuda::rotary_embedding_neox(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()), positions.has_value() ? (*positions)->Data() : position_ids.get(), 
                                  const_cast<void*>(query.DataRaw()), const_cast<void*>(key.DataRaw()), head_size_, (*cos_sin_cache)->DataRaw(), parameters.valid_token_count, rot_dim, num_heads_, num_kv_heads_);
    }

    const std::vector<int64_t>& key_cache_shape = key_cache.Shape();
    int block_size = key_cache_shape[1] / (num_kv_heads_ * head_size_);
    if (parameters.valid_token_count > 0) {
      int32_t key_shape_r[3] = {parameters.valid_token_count, num_kv_heads_, head_size_};
      int32_t value_shape_r[3] = {parameters.valid_token_count, num_kv_heads_, head_size_};
      // TODO(leca): or we just pass num_valid_tokens, num_kv_head, head_size and block_size as parameter?
      cuda::reshape_and_cache(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()), key.DataRaw(), value.DataRaw(), key_cache.DataRaw(), value_cache.DataRaw(), slot_mappings.Data(),
                              key_shape_r, value_shape_r, block_size);
    }

    if (prompt_mode) {
      return RunMultiHeadAttention(ctx, query, key, value, output_data, parameters, seqinfo.get()); // Don't handle prompt with decoding case for now
    }

#ifdef OCOS_USE_FLASH_ATTENTION
    int seqlen_knew = 1;  // TODO(leca): Decoding case, the sequence of k will always be 1?
    int max_num_blocks_per_seq = block_tables.Shape()[1];
    int seqlen_k = max_num_blocks_per_seq * block_size;
    parameters.causal = false;  // flash code: if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    size_t workSpaceSize = cuda::GetAttentionWorkspaceSize(sizeof(T), parameters.batch_size, parameters.num_heads, parameters.head_size, parameters.v_head_size,
                                                           seqlen_knew, nullptr, true/*data.use_flash_attention*/, false/*data.use_memory_efficient_attention*/, true);
    UniquePtrWithDeletor<T> workspace_unique = GetScratchBuffer<T>(allocator_->Alloc(allocator_.get(), workSpaceSize), allocator_.get()); // for softmax_lse
    const cudaDeviceProp& device_prop = DeviceProp::GetCudaDeviceProp();
    return flash::mha_fwd_kvcache(device_prop, reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()), const_cast<void*>(query.DataRaw()), const_cast<void*>(key_cache.DataRaw()),
                                                const_cast<void*>(value_cache.DataRaw()), const_cast<void*>(key.DataRaw()), const_cast<void*>(value.DataRaw()), output_data,
                                                workspace_unique.get(), seqinfo.get(), 
                                                nullptr, nullptr, // rotary_sin and rotary_cos. TODO(leca): Do we still split the input cos_sin_cache as there is a seperate step to do rotary embedding
                                                query_shape[0], num_heads_, num_kv_heads_, head_size_, 1 /*seqlen_q*/, seqlen_k, seqlen_knew, 1.0f/sqrt(head_size_), parameters.causal, false, true,
                                                1 /*num_splits*/, nullptr, nullptr, -1 /*local_window_size*/, false, false, const_cast<int32_t*>(block_tables.Data()), max_num_blocks_per_seq, block_size);
#endif
  }

private:
  int32_t num_heads_;                  // number of attention heads
  int32_t num_kv_heads_;                  // number of attention kv_heads
  int32_t head_size_;                      // number of attention heads
  float scale_;                            // sqrt(head_size_)
  int32_t num_queries_per_kv_;
  UniquePtrWithDeletor<OrtAllocator> allocator_;  // make allocator_ declared first in order to release it last
};