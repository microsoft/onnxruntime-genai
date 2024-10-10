#include "generators.h"
#include "interface.h"
#include "..\search.h"
#include "search_cuda.h"
#include "..\models\kernels.h"

namespace Generators {

struct HostMemory : DeviceMemory {
  HostMemory(size_t size) : DeviceMemory{size} {
    cudaMallocHost(&p_, size);
  }

  ~HostMemory() {
    cudaFreeHost(p_);
  }

  const char* GetType() const override { return "cuda_cpu"; }
  bool IsCpuAccessible() const override { return true; }
};

struct GpuMemory : DeviceMemory {
  GpuMemory(size_t size) : DeviceMemory{size} {
    ::cudaMalloc(&p_, size);
  }

  ~GpuMemory() {
    ::cudaFree(p_);
  }

  const char* GetType() const override { return "cuda"; }
  bool IsCpuAccessible() const override { return false; }
};

struct CudaInterfaceImpl : CudaInterface {
  ~CudaInterfaceImpl() {
  }

  std::unique_ptr<DeviceMemory> Allocate(size_t size, bool cpu_accessible) override {
    if (cpu_accessible)
      return std::make_unique<HostMemory>(size);
    return std::make_unique<DeviceMemory>(size);
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
};


GenaiInterface* gp_genai{};

LogItems& GetLogItems() { return gp_genai->GetLogItems(); }
std::ostream& operator<<(std::ostream& stream, SGR sgr_code) { return gp_genai->operator_leftshift(stream, sgr_code); }
std::ostream& Log(std::string_view label, std::string_view text) { return gp_genai->Log(label, text);}

template<>
void DumpSpan<float>(std::ostream& stream, std::span<const float> values) { return gp_genai->DumpSpan(stream, values); }
template <>
void DumpSpan<int>(std::ostream& stream, std::span<const int> values) { return gp_genai->DumpSpan(stream, values); }

}  // namespace Generators

extern "C" {
__declspec(dllexport) Generators::CudaInterface* CreateCudaInterface(GenaiInterface* p_genai) {
  Generators::gp_genai=p_genai;
  return new Generators::CudaInterfaceImpl();
}
}
