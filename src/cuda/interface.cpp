// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "ort_genai_c.h"  // For OGA_EXPORT
#include "interface.h"
#include "../search.h"
#include "search_cuda.h"
#include "kernels.h"
#include <cstdarg>

#if defined(_WIN32) || defined(_WIN64)
#define strcasecmp _stricmp
#endif

namespace Generators {

GenaiInterface* gp_genai{};
Ort::Allocator* ort_allocator_{};
const char* device_label = "cuda";

cuda_stream_holder g_stream;
cudaStream_t GetStream() { return g_stream.get(); }

struct GpuMemory final : DeviceBuffer {
  GpuMemory(size_t size) : owned_{true} {
    size_in_bytes_ = size;
    p_device_ = static_cast<uint8_t*>(ort_allocator_->Alloc(size));
  }

  GpuMemory(void* p, size_t size) : owned_{false} {
    size_in_bytes_ = size;
    p_device_ = static_cast<uint8_t*>(p);
  }

  ~GpuMemory() override {
    if (owned_)
      ort_allocator_->Free(p_device_);
    if (p_cpu_)
      ::cudaFreeHost(p_cpu_);
  }

  const char* GetType() const override { return device_label; }

  void AllocateCpu() override {
    if (!p_cpu_)
      ::cudaHostAlloc(&p_cpu_, size_in_bytes_, 0);
  }

  void CopyDeviceToCpu() override {
    AllocateCpu();
    ::cudaMemcpyAsync(p_cpu_, p_device_, size_in_bytes_, ::cudaMemcpyDeviceToHost, GetStream());
    ::cudaStreamSynchronize(GetStream());
  }

  void CopyCpuToDevice() override {
    assert(p_cpu_);
    ::cudaMemcpyAsync(p_device_, p_cpu_, size_in_bytes_, ::cudaMemcpyHostToDevice, GetStream());
  }

  void CopyFrom(size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) override {
    if (source.GetType() == device_label)
      ::cudaMemcpyAsync(p_device_ + begin_dest, source.p_device_ + begin_source, size_in_bytes, ::cudaMemcpyDeviceToDevice, GetStream());
    else
      gp_genai->CopyThroughCpu(*this, begin_dest, source, begin_source, size_in_bytes);
  }

  void Zero() override {
    ::cudaMemsetAsync(p_device_, 0, size_in_bytes_, GetStream());
  }

  bool owned_;  // If we own the memory, we delete it on destruction
};

struct CudaInterfaceImplBase : DeviceInterface {
  CudaInterfaceImplBase() {
    g_stream.Create();
  }

  ~CudaInterfaceImplBase() {
  }

  void InitOrt(const OrtApi& api, Ort::Allocator& allocator) override {
    Ort::api = &api;
    assert(!ort_allocator_);
    ort_allocator_ = &allocator;
  }

  Ort::Allocator& GetAllocator() override {
    return *ort_allocator_;
  }

  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return std::make_shared<GpuMemory>(size);
  }

  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* p, size_t size) override {
    return std::make_shared<GpuMemory>(p, size);
  }

  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override {
    return std::make_unique<GreedySearch_Cuda>(params);
  }

  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override {
    return std::make_unique<BeamSearch_Cuda>(params);
  }

  void Synchronize() override {
    ::cudaStreamSynchronize(GetStream());
  }

  void* GetCudaStream() override {
    return GetStream();
  }

  bool Cast(void* input_data, void* output_data, ONNXTensorElementDataType input_type, ONNXTensorElementDataType output_type, size_t element_count) override {
    if (input_type == output_type)
      throw std::runtime_error("Cast - input and output types are the same");

    if (input_type == Ort::TypeToTensorType<float> && output_type == Ort::TypeToTensorType<Ort::Float16_t>) {
      cuda::LaunchFp32ToFp16(reinterpret_cast<const float*>(input_data), reinterpret_cast<uint16_t*>(output_data), static_cast<int>(element_count), GetStream());
    } else if (input_type == Ort::TypeToTensorType<Ort::Float16_t> && output_type == Ort::TypeToTensorType<float>) {
      cuda::LaunchFp16ToFp32(reinterpret_cast<const uint16_t*>(input_data), reinterpret_cast<float*>(output_data), static_cast<int>(element_count), GetStream());
    } else if (input_type == Ort::TypeToTensorType<int32_t> && output_type == Ort::TypeToTensorType<int64_t>) {
      cuda::LaunchInt32ToInt64(reinterpret_cast<const int32_t*>(input_data), reinterpret_cast<int64_t*>(output_data), static_cast<int>(element_count), GetStream());
    } else
      return false;
    return true;
  }

  bool UpdatePositionIds(void* position_ids, int batch_beam_size, int total_length, int new_kv_length, ONNXTensorElementDataType type) override {
    if (type == Ort::TypeToTensorType<int32_t>)
      cuda::Launch_UpdatePositionIds(static_cast<int32_t*>(position_ids), batch_beam_size, total_length, new_kv_length, GetStream());
    else
      cuda::Launch_UpdatePositionIds(static_cast<int64_t*>(position_ids), batch_beam_size, total_length, new_kv_length, GetStream());
    return true;
  }

  bool UpdateAttentionMask(void* next_mask_data, void* mask_data, int batch_beam_size, int new_kv_length, int total_length, int max_length, bool update_only, ONNXTensorElementDataType type) override {
    if (type == Ort::TypeToTensorType<int32_t>)
      cuda::Launch_UpdateAttentionMask(static_cast<int32_t*>(next_mask_data), static_cast<int32_t*>(mask_data), batch_beam_size, new_kv_length, total_length, max_length, update_only, GetStream());
    else
      cuda::Launch_UpdateAttentionMask(static_cast<int64_t*>(next_mask_data), static_cast<int64_t*>(mask_data), batch_beam_size, new_kv_length, total_length, max_length, update_only, GetStream());
    return true;
  }

  void UpdateCacheIndirection(int32_t* tgt_indir_cache, const int32_t* src_indir_cache, const int32_t* beam_ids, int batch_size, int beam_width, int input_seq_length, int max_seq_length, int current_length) override {
    cuda::UpdateCacheIndirectionKernelLauncher(tgt_indir_cache, src_indir_cache, beam_ids, batch_size, beam_width, input_seq_length, max_seq_length, current_length, GetStream());
  }

  void ReorderPastStates(void* out_buffer, const void* in_buffer, int batch_size, int num_heads, int max_length, int head_size, int chunk_size) override {
    cuda::ReorderPastStatesKernelLauncher(out_buffer, in_buffer, batch_size, num_heads, max_length, head_size, chunk_size, GetStream());
  }

  void CopyCrossQK(float* cross_qk_buffer_data, void** qk_layer_pointers, int token_index, int batch_beam_size, int num_layers, int num_heads, int num_alignment_heads, const int* alignment_heads, int frames, int max_length, int sequence_length) override {
    cuda::LaunchCopyCrossQKSingleDecodeStep(GetStream(), cross_qk_buffer_data, qk_layer_pointers, token_index, batch_beam_size, num_layers, num_heads, num_alignment_heads, alignment_heads, frames, max_length, sequence_length);
  }

  void CopyCrossQK(Ort::Float16_t* cross_qk_buffer_data, void** qk_layer_pointers, int token_index, int batch_beam_size, int num_layers, int num_heads, int num_alignment_heads, const int* alignment_heads, int frames, int max_length, int sequence_length) override {
    cuda::LaunchCopyCrossQKSingleDecodeStep(GetStream(), reinterpret_cast<uint16_t*>(cross_qk_buffer_data), qk_layer_pointers, token_index, batch_beam_size, num_layers, num_heads, num_alignment_heads, alignment_heads, frames, max_length, sequence_length);
  }

  void FinalizeCrossQK(int iteration_number, int context_decoding_len, int batch_size, int num_beams, int max_length, int num_alignment_heads, int frames_of_k, const float* cross_qk_buffer_data, float* cross_qk_output, int num_return_sequences, const int* cache_indir_data) override {
    cuda::LaunchFinalizeCrossQK(GetStream(), iteration_number, context_decoding_len, batch_size, num_beams, max_length, num_alignment_heads, frames_of_k, cross_qk_buffer_data, cross_qk_output, num_return_sequences, cache_indir_data);
  }

  void FinalizeCrossQK(int iteration_number, int context_decoding_len, int batch_size, int num_beams, int max_length, int num_alignment_heads, int frames_of_k, const Ort::Float16_t* cross_qk_buffer_data, Ort::Float16_t* cross_qk_output, int num_return_sequences, const int* cache_indir_data) override {
    cuda::LaunchFinalizeCrossQK(GetStream(), iteration_number, context_decoding_len, batch_size, num_beams, max_length, num_alignment_heads, frames_of_k, reinterpret_cast<const uint16_t*>(cross_qk_buffer_data), reinterpret_cast<uint16_t*>(cross_qk_output), num_return_sequences, cache_indir_data);
  }

  void LaunchAddLogitsMask(float* batch_logits, int batch_beam_size, int vocab_size, const uint32_t* logits_mask) override {
    cuda::LaunchAddLogitsMask(batch_logits, batch_beam_size, vocab_size, logits_mask, GetStream());
  }
};

struct CudaInterfaceImpl final : CudaInterfaceImplBase {
  DeviceType GetType() const override { return DeviceType::CUDA; }
};

struct NvTensorRtRtxInterfaceImpl final : CudaInterfaceImplBase {
  DeviceType GetType() const override { return DeviceType::NvTensorRtRtx; }
};

std::unique_ptr<DeviceInterface> g_cuda_device;

DeviceInterface& GetCudaDeviceInterface() { return *g_cuda_device; }

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

void Sequences::AfterAppendNextTokens(DeviceSpan<int32_t>& next_tokens, size_t batch_beam_size) { return gp_genai->Sequences_AfterAppendNextTokens(this, next_tokens, batch_beam_size); }
void Sequences::RewindTo(size_t new_length) { return gp_genai->Sequences_RewindTo(this, new_length); }
}  // namespace Generators

#ifdef _WIN32
// Override default new/delete so that we match the host's allocator
// Previous implementation calls Generators::gp_genai->HeapAllocate(n) or HeapFree(p).
// But memory allocation might be called before gp_genai created so gp_genai might be nullptr, which causes crash.
// Here we just copy the implementation of HeapAllocate and HeapFree to avoid initialization order issue.
_Ret_notnull_ _Post_writable_byte_size_(n) void* operator new(size_t n) {
  return std::malloc(n);
}
void operator delete(void* p) noexcept {
  return std::free(p);
}

void operator delete(void* p, size_t /*size*/) noexcept {
  return std::free(p);
}
#endif

extern "C" {
Generators::DeviceInterface* GetInterface(GenaiInterface* p_genai, const char* deviceType) {
  Generators::gp_genai = p_genai;
  if (strcasecmp(deviceType, "NvTensorRtRtx") == 0) {
    Generators::g_cuda_device = std::make_unique<Generators::NvTensorRtRtxInterfaceImpl>();
  } else {
    Generators::g_cuda_device = std::make_unique<Generators::CudaInterfaceImpl>();
  }
  return Generators::g_cuda_device.get();
}
}
