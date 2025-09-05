// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <assert.h>
#include <atomic>
#include <memory>
#include "span.h"
#include "models/onnxruntime_api.h" // for ONNXTensorElementDataType
namespace Ort {
struct Allocator;
}

namespace Generators {
struct Search;
struct Sequences;
struct GeneratorParams;

// A DeviceBuffer is an abstract interface to a block of device memory (can be cuda/dml/cpu memory)
// Note: For a CPU DeviceBuffer, there's only one block of memory on CPU, the copy methods are no-ops
// Do not use DeviceBuffer directly, use a DeviceSpan (the Allocate/WrapMemory methods return DeviceSpans)
struct DeviceBuffer : std::enable_shared_from_this<DeviceBuffer> {
  virtual ~DeviceBuffer() {}
  virtual const char* GetType() const = 0;  // Returns "cuda" "cuda_cpu" "directml" etc

  virtual void AllocateCpu() = 0;      // Allocates p_cpu_ if necessary (using appropriate memory type for interop)
  virtual void CopyDeviceToCpu() = 0;  // Allocates p_cpu_ if necessary and copies p_device_ memory into it
  virtual void CopyCpuToDevice() = 0;
  virtual void CopyFrom(size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) = 0;
  virtual void Zero() = 0;  // Zero out the device memory

  uint8_t* p_device_{};
  uint8_t* p_cpu_{};
  size_t size_in_bytes_{};
};

// A DeviceSpan is how a DeviceBuffer is used. It can be thought of as a std::span for device memory with
// utilities to interop with CPU memory. It is what Allocate<T> returns and what should be passed around by value.
template <typename T>
struct DeviceSpan {
  DeviceSpan() = default;
  DeviceSpan(std::shared_ptr<DeviceBuffer>&& memory)
      : p_device_memory_{std::move(memory)}, begin_{}, length_{p_device_memory_->size_in_bytes_ / sizeof(T)} {}

  bool empty() const { return length_ == 0; }
  size_t size() const { return length_; }

  operator DeviceSpan<const T>() const { return DeviceSpan<const T>(*p_device_memory_, begin_, length_); }

  DeviceSpan<T> subspan(size_t begin, size_t length) { return DeviceSpan<T>(*p_device_memory_, begin_ + begin, length); }

  // Return the device accessible memory. Should only be done in device specific code, as it's not CPU accessible
  std::span<T> Span() { return std::span<T>{reinterpret_cast<T*>(p_device_memory_->p_device_) + begin_, length_}; }

  // Return the CPU accessible memory, allocating if necessary (note, to get the current device memory on CPU, use 'CopyDeviceToCpu' instead)
  std::span<T> CpuSpan() {
    p_device_memory_->AllocateCpu();
    return std::span<T>{reinterpret_cast<T*>(p_device_memory_->p_cpu_) + begin_, length_};
  }

  // Copy device memory to CPU memory and return the CPU accessible memory
  std::span<T> CopyDeviceToCpu() {
    p_device_memory_->CopyDeviceToCpu();
    return std::span<T>{reinterpret_cast<T*>(p_device_memory_->p_cpu_) + begin_, length_};
  }

  // Copy CPU memory to device memory, typically used after calling CpuSpan or CopyDeviceToCpu to update the device memory with the modifications made
  void CopyCpuToDevice() { p_device_memory_->CopyCpuToDevice(); }

  // Zero out the device memory
  void Zero() { p_device_memory_->Zero(); }

  void CopyFrom(const DeviceSpan<const T>& source) {
    assert(source.size() == size());  // Spans must be the same size to copy
    p_device_memory_->CopyFrom(begin_ * sizeof(T), *source.p_device_memory_, source.begin_ * sizeof(T), length_ * sizeof(T));
  }

 private:
  DeviceSpan(DeviceBuffer& memory, size_t begin, size_t length)
      : p_device_memory_{memory.shared_from_this()}, begin_{begin}, length_{length} {}

  std::shared_ptr<DeviceBuffer> p_device_memory_;
  size_t begin_{}, length_{};  // Subspan of p_device_memory_, relative to original memory block
  template <typename U>
  friend struct DeviceSpan;  // All DeviceSpans are friends
};

enum struct DeviceType {
  CPU,
  CUDA,
  DML,
  WEBGPU,
  QNN,
  OpenVINO,
  NvTensorRtRtx,
  MAX
};

struct DeviceInterface {
  virtual ~DeviceInterface() {}

  virtual DeviceType GetType() const = 0;
  virtual void InitOrt(const OrtApi& api, Ort::Allocator& allocator) = 0;
  virtual Ort::Allocator& GetAllocator() = 0;

  template <typename T>
  DeviceSpan<T> Allocate(size_t count) { return DeviceSpan<T>(AllocateBase(sizeof(T) * count)); }
  virtual std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) = 0;

  // Wraps an existing memory block, useful for tensors. Use WrapTensor for OrtValue vs calling this directly
  template <typename T>
  DeviceSpan<T> WrapMemory(std::span<T> memory) { return DeviceSpan<T>(WrapMemoryBase(const_cast<std::remove_const_t<T>*>(memory.data()), memory.size_bytes())); }
  virtual std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* memory, size_t size) = 0;

  virtual std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) = 0;
  virtual std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) = 0;

  virtual void Synchronize() = 0;  // Synchronize the device, typically used for timing or debugging

  virtual bool Cast(void* /*input*/, void* /*output*/, ONNXTensorElementDataType /*input_type*/, ONNXTensorElementDataType /*output_type*/, size_t /*element_count*/) { return false; }

  virtual bool UpdatePositionIds(void* /*position_ids*/, int /*batch_beam_size*/, int /*total_length*/, int /*new_kv_length*/, ONNXTensorElementDataType /*type*/) { return false; }
  virtual bool UpdateAttentionMask(void* /*next_mask_data*/, void* /*mask_data*/, int /*batch_beam_size*/, int /*new_kv_length*/, int /*total_length*/, int /*max_length*/, bool /*update_only*/, ONNXTensorElementDataType /*type*/) { return false; }
  virtual void LaunchAddLogitsMask(float* /*batch_logits*/, int /*batch_beam_size*/, int /*vocab_size*/, const uint32_t* /*logits_mask*/) { assert(false); }

  virtual void UpdateCacheIndirection(int32_t* /*tgt_indir_cache*/, const int32_t* /*src_indir_cache*/, const int32_t* /*beam_ids*/, int /*batch_size*/, int /*beam_width*/, int /*input_seq_length*/, int /*max_seq_length*/, int /*current_length*/) { assert(false); }
  virtual void ReorderPastStates(void* /*out_buffer*/, const void* /*in_buffer*/, int /*batch_size*/, int /*num_heads*/, int /*max_length*/, int /*head_size*/, int /*chunk_size*/) { assert(false); }
  virtual void CopyCrossQK(float* /*cross_qk_buffer_data*/, void** /*qk_layer_pointers*/, int /*token_index*/, int /*batch_beam_size*/, int /*num_layers*/, int /*num_heads*/, int /*num_alignment_heads*/, const int* /*alignment_heads*/, int /*frames*/, int /*max_length*/, int /*sequence_length*/) { assert(false); }
  virtual void CopyCrossQK(Ort::Float16_t* /*cross_qk_buffer_data*/, void** /*qk_layer_pointers*/, int /*token_index*/, int /*batch_beam_size*/, int /*num_layers*/, int /*num_heads*/, int /*num_alignment_heads*/, const int* /*alignment_heads*/, int /*frames*/, int /*max_length*/, int /*sequence_length*/) { assert(false); }
  virtual void FinalizeCrossQK(int /*iteration_number*/, int /*context_decoding_len*/, int /*batch_size*/, int /*num_beams*/, int /*max_length*/, int /*num_alignment_heads*/, int /*frames_of_k*/, const float* /*cross_qk_buffer_data*/, float* /*cross_qk_output*/, int /*num_return_sequences*/, const int* /*cache_indir_data*/) { assert(false); }
  virtual void FinalizeCrossQK(int /*iteration_number*/, int /*context_decoding_len*/, int /*batch_size*/, int /*num_beams*/, int /*max_length*/, int /*num_alignment_heads*/, int /*frames_of_k*/, const Ort::Float16_t* /*cross_qk_buffer_data*/, Ort::Float16_t* /*cross_qk_output*/, int /*num_return_sequences*/, const int* /*cache_indir_data*/) { assert(false); }

  virtual void* GetCudaStream() {
    assert(false);
    return nullptr;
  }  // Temporary until we fully factor out providers
};

// A shared_ptr based type that we expose through our C API should inherit from this type.
// ExternalAddRef must be called when returning an object through the C API
// ExternalRelease must be called on the C API destroy method
template <typename T>
struct ExternalRefCounted {
  void ExternalAddRef() {
    if (++ref_count_ == 1)  // First reference?
      external_owner_ = static_cast<T*>(this)->shared_from_this();
  }
  void ExternalRelease() {
    if (--ref_count_ == 0)
      external_owner_ = nullptr;
  }

 private:
  std::shared_ptr<T> external_owner_;  // shared_ptr to ourselves to keep us alive
  std::atomic<int> ref_count_{};       // C API refcount (can't use only the shared_ptr)
};

namespace Location {
struct CPU {};
struct GPU {};
}  // namespace Location

template <typename T>
struct cpu_span : std::span<T> {
  using std::span<T>::span;
  explicit cpu_span(std::span<T> v) : std::span<T>(v) {}
};
template <typename T>
struct gpu_span : std::span<T> {
  using std::span<T>::span;
  explicit gpu_span(std::span<T> v) : std::span<T>(v) {}
};

template <typename T>
void copy(std::span<const T> source, std::span<T> dest) {
  assert(source.size() == dest.size());
  std::copy(source.begin(), source.end(), dest.begin());
}

template <typename T>
std::unique_ptr<T[]> AllocateArray(size_t count, std::span<T>* p_span = nullptr) {
  T* p = new T[count];
  if (p_span)
    *p_span = std::span<T>(p, count);
  return std::unique_ptr<T[]>{p};
}

}  // namespace Generators
