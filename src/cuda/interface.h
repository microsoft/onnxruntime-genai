// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

struct GenaiInterface {
#if _WIN32
  virtual void* HeapAllocate(size_t size) = 0;
  virtual void HeapFree(void*) = 0;
#endif

  virtual Generators::LogItems& GetLogItems() = 0;
  virtual std::ostream& operator_leftshift(std::ostream& stream, Generators::SGR sgr_code) = 0;
  virtual std::ostream& Log(std::string_view label, std::string_view text = {}) = 0;

  virtual void DumpSpan(std::ostream& stream, std::span<const float> values) = 0;
  virtual void DumpSpan(std::ostream& stream, std::span<const int> values) = 0;

  virtual void Sequences_AfterAppendNextTokens(Generators::Sequences* p_this, Generators::DeviceSpan<int32_t> next_tokens) = 0;
};

namespace Generators {
LogItems& GetLogItems();

#if USE_CUDA
DeviceInterface& GetCudaDeviceInterface();

struct CudaInterface : DeviceInterface {
  virtual void Int32ToInt64(const int32_t* input, int64_t* output, int count, cudaStream_t stream) = 0;
  virtual void Fp16ToFp32(const uint16_t* input, float* output, int count, cudaStream_t stream) = 0;
  virtual void Fp32ToFp16(const float* input, uint16_t* output, int count, cudaStream_t stream) = 0;

  virtual void Launch_UpdatePositionIds(int32_t* position_ids, int batch_beam_size, cudaStream_t stream) = 0;
  virtual void Launch_UpdatePositionIds(int64_t* position_ids, int batch_beam_size, cudaStream_t stream) = 0;
  virtual void Launch_UpdateAttentionMask(int32_t* mask_data, const int32_t* old_mask_data, int batch_beam_size, int current_length, int max_length, bool update_only, cudaStream_t stream) = 0;
  virtual void Launch_UpdateAttentionMask(int64_t* mask_data, const int64_t* old_mask_data, int batch_beam_size, int current_length, int max_length, bool update_only, cudaStream_t stream) = 0;
  virtual void LaunchHandleEOSArray(float* batch_logits, int batch_beam_size, int vocab_size, const int32_t* eos_token_ids, int eos_token_ids_count, cudaStream_t stream) = 0;
  virtual void UpdateCacheIndirectionKernelLauncher(int32_t* tgt_indir_cache, const int32_t* src_indir_cache, const int32_t* beam_ids, int batch_size, int beam_width, int input_seq_length, int max_seq_length, int current_length, cudaStream_t stream) = 0;
  virtual void ReorderPastStatesKernelLauncher(void* out_buffer, const void* in_buffer, int batch_size, int num_heads, int max_length, int head_size, int chunk_size, cudaStream_t stream) = 0;
  virtual void LaunchCopyCrossQKSingleDecodeStep(cudaStream_t stream, float* cross_qk_buffer_data, float** qk_layer_pointers, int token_index, int batch_beam_size, int num_layers, int num_heads, int num_alignment_heads, const int* alignment_heads, int frames, int max_length) = 0;
  virtual void LaunchFinalizeCrossQK(cudaStream_t stream, int iteration_number, int context_decoding_len, int batch_size, int num_beams, int max_length, int num_alignment_heads, int frames_of_k, const float* cross_qk_buffer_data, float* cross_qk_output, int num_return_sequences, const int* cache_indir_data) = 0;

  virtual cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) = 0;
  virtual cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) = 0;
  virtual cudaError_t cudaMemsetAsync(void* ptr, int value, size_t count, cudaStream_t stream) = 0;
  virtual cudaError_t cudaMemset(void* ptr, int value, size_t count) = 0;
};
#endif
}  // namespace Generators
