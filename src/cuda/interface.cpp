// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "ort_genai_c.h"  // For OGA_EXPORT
#include "interface.h"
#include "../search.h"
#include "search_cuda.h"
#include "../models/kernels.h"
#include <cstdarg>

namespace Generators {

const char* label_cuda = "cuda";
const char* label_cuda_cpu = "cuda_cpu";

struct HostMemory : DeviceMemoryBase {
  HostMemory(size_t size) {
    size_in_bytes_ = size;
    ::cudaMallocHost(&p_device_, size);
    p_cpu_ = p_device_;  // CPU & GPU both access the same memory here
  }

  ~HostMemory() override {
    ::cudaFreeHost(p_device_);
  }

  const char* GetType() const override { return label_cuda_cpu; }
  bool IsCpuAccessible() const override { return true; }
  void GetOnCpu() override { assert(false); }  // Should never be called, as p_cpu_ is always valid
  void CopyFromDevice(size_t begin_dest, DeviceMemoryBase& source, size_t begin_source, size_t size_in_bytes) override {
    if (source.GetType() == label_cuda_cpu)
      ::memcpy(p_cpu_ + begin_dest, source.p_cpu_ + begin_source, size_in_bytes);
    else if (source.GetType() == label_cuda)
      ::cudaMemcpy(p_device_ + begin_dest, source.p_device_ + begin_source, size_in_bytes, ::cudaMemcpyDeviceToHost);
    else
      throw std::runtime_error("Cuda HostMemory::CopyFromDevice not implemented for " + std::string(source.GetType()));
  }
};

struct GpuMemory : DeviceMemoryBase {
  GpuMemory(size_t size) {
    size_in_bytes_ = size;
    ::cudaMalloc(&p_device_, size);
  }

  ~GpuMemory() override {
    ::cudaFree(p_device_);
    if (p_cpu_)
      ::cudaFreeHost(p_cpu_);
  }

  const char* GetType() const override { return label_cuda; }
  bool IsCpuAccessible() const override { return false; }
  void GetOnCpu() override {
    assert(!p_cpu_);
    ::cudaMallocHost(&p_cpu_, size_in_bytes_);
    ::cudaMemcpy(p_cpu_, p_device_, size_in_bytes_, ::cudaMemcpyDeviceToHost);
  }

  void CopyFromDevice(size_t begin_source, DeviceMemoryBase& source, size_t begin_dest, size_t size_in_bytes) override {
    if (source.GetType() == label_cuda_cpu)
      ::cudaMemcpy(p_device_ + begin_source, source.p_device_ + begin_dest, size_in_bytes, ::cudaMemcpyHostToDevice);
    else if (source.GetType() == label_cuda)
      ::cudaMemcpy(p_device_ + begin_source, source.p_device_ + begin_dest, size_in_bytes, ::cudaMemcpyDeviceToDevice);
    else
      throw std::runtime_error("Cuda GpuMemory::CopyFromDevice not implemented for " + std::string(source.GetType()));
  }
};

struct CudaInterfaceImpl : CudaInterface {
  ~CudaInterfaceImpl() {
  }

  std::shared_ptr<DeviceMemoryBase> AllocateBase(size_t size, bool cpu_accessible) override {
    if (cpu_accessible)
      return std::make_shared<HostMemory>(size);
    return std::make_shared<GpuMemory>(size);
  }

  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override {
    return std::make_unique<GreedySearch_Cuda>(params);
  }

  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override {
    return std::make_unique<BeamSearch_Cuda>(params);
  }

  void Int32ToInt64(const int32_t* input, int64_t* output, int count, cudaStream_t stream) override {
    cuda::LaunchInt32ToInt64(input, output, count, stream);
  }

  void Fp16ToFp32(const uint16_t* input, float* output, int count, cudaStream_t stream) override {
    cuda::LaunchFp16ToFp32(input, output, count, stream);
  }

  void Fp32ToFp16(const float* input, uint16_t* output, int count, cudaStream_t stream) override {
    cuda::LaunchFp32ToFp16(input, output, count, stream);
  }

  void Launch_UpdatePositionIds(int32_t* position_ids, int batch_beam_size, cudaStream_t stream) override {
    cuda::Launch_UpdatePositionIds(position_ids, batch_beam_size, stream);
  }

  void Launch_UpdatePositionIds(int64_t* position_ids, int batch_beam_size, cudaStream_t stream) override {
    cuda::Launch_UpdatePositionIds(position_ids, batch_beam_size, stream);
  }

  void Launch_UpdateAttentionMask(int32_t* mask_data, const int32_t* old_mask_data, int batch_beam_size, int current_length, int max_length, bool update_only, cudaStream_t stream) override {
    cuda::Launch_UpdateAttentionMask(mask_data, old_mask_data, batch_beam_size, current_length, max_length, update_only, stream);
  }

  void Launch_UpdateAttentionMask(int64_t* mask_data, const int64_t* old_mask_data, int batch_beam_size, int current_length, int max_length, bool update_only, cudaStream_t stream) override {
    cuda::Launch_UpdateAttentionMask(mask_data, old_mask_data, batch_beam_size, current_length, max_length, update_only, stream);
  }

  void LaunchHandleEOSArray(float* batch_logits, int batch_beam_size, int vocab_size, const int32_t* eos_token_ids, int eos_token_ids_count, cudaStream_t stream) override {
    cuda::LaunchHandleEOSArray(batch_logits, batch_beam_size, vocab_size, eos_token_ids, eos_token_ids_count, stream);
  }

  void UpdateCacheIndirectionKernelLauncher(int32_t* tgt_indir_cache, const int32_t* src_indir_cache, const int32_t* beam_ids, int batch_size, int beam_width, int input_seq_length, int max_seq_length, int current_length, cudaStream_t stream) override {
    cuda::UpdateCacheIndirectionKernelLauncher(tgt_indir_cache, src_indir_cache, beam_ids, batch_size, beam_width, input_seq_length, max_seq_length, current_length, stream);
  }

  void ReorderPastStatesKernelLauncher(void* out_buffer, const void* in_buffer, int batch_size, int num_heads, int max_length, int head_size, int chunk_size, cudaStream_t stream) override {
    cuda::ReorderPastStatesKernelLauncher(out_buffer, in_buffer, batch_size, num_heads, max_length, head_size, chunk_size, stream);
  }

  void LaunchCopyCrossQKSingleDecodeStep(cudaStream_t stream, float* cross_qk_buffer_data, float** qk_layer_pointers, int token_index, int batch_beam_size, int num_layers, int num_heads, int num_alignment_heads, const int* alignment_heads, int frames, int max_length) override {
    cuda::LaunchCopyCrossQKSingleDecodeStep(stream, cross_qk_buffer_data, qk_layer_pointers, token_index, batch_beam_size, num_layers, num_heads, num_alignment_heads, alignment_heads, frames, max_length);
  }

  void LaunchFinalizeCrossQK(cudaStream_t stream, int iteration_number, int context_decoding_len, int batch_size, int num_beams, int max_length, int num_alignment_heads, int frames_of_k, const float* cross_qk_buffer_data, float* cross_qk_output, int num_return_sequences, const int* cache_indir_data) override {
    cuda::LaunchFinalizeCrossQK(stream, iteration_number, context_decoding_len, batch_size, num_beams, max_length, num_alignment_heads, frames_of_k, cross_qk_buffer_data, cross_qk_output, num_return_sequences, cache_indir_data);
  }

  cudaError_t cudaStreamCreate(cudaStream_t* stream) override {
    return ::cudaStreamCreate(stream);
  }

  cudaError_t cudaStreamDestroy(cudaStream_t stream) override {
    return ::cudaStreamDestroy(stream);
  }

  cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) override {
    return ::cudaMemcpyAsync(dst, src, count, kind, stream);
  }

  cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) override {
    return ::cudaMemcpy(dst, src, count, kind);
  }

  cudaError_t cudaMemsetAsync(void* ptr, int value, size_t count, cudaStream_t stream) override {
    return ::cudaMemsetAsync(ptr, value, count, stream);
  }

  cudaError_t cudaMemset(void* ptr, int value, size_t count) override {
    return ::cudaMemset(ptr, value, count);
  }

  cudaError_t cudaMalloc(void** ptr, size_t size) override {
    return ::cudaMalloc(ptr, size);
  }

  cudaError_t cudaFree(void* ptr) override {
    return ::cudaFree(ptr);
  }

  cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) override {
    return ::cudaHostAlloc(ptr, size, flags);
  }

  cudaError_t cudaFreeHost(void* ptr) override {
    return ::cudaFreeHost(ptr);
  }
} g_cuda_device;

DeviceInterface& GetCudaDeviceInterface() { return g_cuda_device; }
GenaiInterface* gp_genai{};

LogItems& GetLogItems() { return gp_genai->GetLogItems(); }
std::ostream& operator<<(std::ostream& stream, SGR sgr_code) { return gp_genai->operator_leftshift(stream, sgr_code); }
std::ostream& Log(std::string_view label, std::string_view text) { return gp_genai->Log(label, text); }

// Duplicate of logging.cpp function
std::ostream& Log(std::string_view label, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  va_list args_copy;
  va_copy(args_copy, args);
  size_t len = vsnprintf(0, 0, fmt, args_copy);
  if (len <= 0) {
    throw std::runtime_error("Invalid format");
  }
  std::unique_ptr<char[]> buf(new char[len + 1]);
  vsnprintf(buf.get(), len + 1, fmt, args);
  va_end(args);
  return Log(label, std::string(buf.get(), buf.get() + len));
}

template <>
void DumpSpan<float>(std::ostream& stream, std::span<const float> values) { return gp_genai->DumpSpan(stream, values); }
template <>
void DumpSpan<int>(std::ostream& stream, std::span<const int> values) { return gp_genai->DumpSpan(stream, values); }

}  // namespace Generators

#ifdef _WIN32
// Override default new/delete so that we match the host's allocator
_Ret_notnull_ _Post_writable_byte_size_(n) void* operator new(size_t n) { return Generators::gp_genai->HeapAllocate(n); }
void operator delete(void* p) noexcept { Generators::gp_genai->HeapFree(p); }
void operator delete(void* p, size_t /*size*/) noexcept { Generators::gp_genai->HeapFree(p); }
#endif

extern "C" {
Generators::CudaInterface* GetInterface(GenaiInterface* p_genai) {
  Generators::gp_genai = p_genai;
  return &Generators::g_cuda_device;
}
}
