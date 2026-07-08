// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <algorithm>  // for std::copy
#include <assert.h>
#include <atomic>
#include <cstring>  // for std::strcmp
#include <memory>
#include <type_traits>  // for std::remove_const_t
#include "span.h"
#include "models/onnxruntime_api.h"  // for ONNXTensorElementDataType
#include "provider_options.h"        // for ProviderOptions
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
  RyzenAI,
  MAX
};

// §6: Step 1 — find the OrtEpDevice entries on `env` whose EP name matches one of `ep_names`. This
// is pure name filtering and is independent of allocators: a non-empty result means the EP is
// registered as a plugin EP (the presence of an OrtEpDevice is the plugin-EP signal). `env` is the
// caller's own env (in-process EPs pass GetOrtEnv(); the CUDA add-on passes the env handed to it at
// load). Defined inline so the CUDA add-on library (which links neither onnxruntime nor genai core,
// only Ort::api-based inline wrappers) can use it too.
inline std::vector<const OrtEpDevice*> FindEpDevicesByName(OrtEnv& env, std::span<const char* const> ep_names) {
  std::vector<const OrtEpDevice*> devices;
  const OrtEpDevice* const* device_ptrs = nullptr;
  size_t num_devices = 0;
  Ort::GetEpDevices(&env, &device_ptrs, &num_devices);
  for (const auto* device : std::span{device_ptrs, num_devices}) {
    const char* dev_name = Ort::api->EpDevice_EpName(device);
    for (const char* n : ep_names)
      if (std::strcmp(dev_name, n) == 0) {
        devices.push_back(device);
        break;
      }
  }
  return devices;
}

// §6/§11: The shared allocators an execution provider exposes on an OrtEnv, resolved (step 2) from a
// device list produced by FindEpDevicesByName (step 1). "Availability" is decided by whether
// Ort::GetSharedAllocator returns an allocator (not by whether EpDevice_MemoryInfo returns a
// mem-info).
//
// Generic allocator policy — one model for every plugin EP:
//   * device_allocator is the allocator genai uses for a DeviceBuffer's device memory. It is the
//     EP's DEFAULT (device-local) shared allocator when it advertises one (e.g. CUDA, real WebGPU);
//     otherwise it is the HOST_ACCESSIBLE shared allocator. EPs that do not allocate separate
//     device memory (e.g. QNN's QnnHtpShared, OpenVINO) expose only CPU-accessible shared memory
//     that serves as both device and host memory (a DeviceBuffer then has p_device_ == p_cpu_).
//   * host_allocator is the EP's HOST_ACCESSIBLE (pinned / mappable) shared allocator, reported
//     whenever the EP advertises one — independent of whether a DEFAULT allocator also exists. It
//     is used for host-side staging (e.g. copying a CPU input into EP-visible memory). When the EP
//     has no DEFAULT allocator, device_allocator is set to this same host-accessible allocator (so
//     device == host) but host_allocator still reports it, so callers that specifically need
//     host-accessible memory behave the same regardless of whether a DEFAULT allocator exists.
struct EpSharedAllocators {
  Ort::Allocator* device_allocator{};      // Device-memory allocator: DEFAULT, or HOST_ACCESSIBLE when no DEFAULT exists.
  const OrtMemoryInfo* device_mem_info{};  // Its mem-info, or null.
  Ort::Allocator* host_allocator{};        // HOST_ACCESSIBLE staging allocator, whenever the EP advertises one, else null.
  const OrtMemoryInfo* host_mem_info{};    // Its mem-info, or null.

  bool HasDeviceAllocator() const { return device_allocator != nullptr; }
  bool HasHostAccessibleAllocator() const { return host_allocator != nullptr; }
};

// §6/§11: Step 2 — resolve the shared allocators the given `devices` (from FindEpDevicesByName)
// expose on `env`.
inline EpSharedAllocators ResolveEpSharedAllocators(OrtEnv& env, std::span<const OrtEpDevice* const> devices) {
  EpSharedAllocators result;
  for (const auto* device : devices) {
    // Device-local (DEFAULT) allocator. Signal = a shared allocator is actually available.
    if (!result.device_allocator) {
      if (const OrtMemoryInfo* mi = Ort::GetMemoryInfo(device, OrtDeviceMemoryType_DEFAULT)) {
        if (Ort::Allocator* a = Ort::GetSharedAllocator(&env, mi)) {
          result.device_allocator = a;
          result.device_mem_info = mi;
        }
      }
    }

    // Host-accessible allocator, usable only when this device is non-CPU (§11 gate): a CPU device
    // needs no special host-accessible allocator (plain CPU memory suffices).
    if (!result.host_allocator) {
      const OrtHardwareDevice* hw = Ort::api->EpDevice_Device(device);
      if (hw && Ort::api->HardwareDevice_Type(hw) != OrtHardwareDeviceType_CPU) {
        if (const OrtMemoryInfo* mi = Ort::GetMemoryInfo(device, OrtDeviceMemoryType_HOST_ACCESSIBLE)) {
          if (Ort::Allocator* a = Ort::GetSharedAllocator(&env, mi)) {
            result.host_allocator = a;
            result.host_mem_info = mi;
          }
        }
      }
    }
  }

  // Generic policy: if the EP advertises no DEFAULT device allocator, use its HOST_ACCESSIBLE
  // allocator as the device allocator — device memory is host-accessible shared memory (QNN,
  // OpenVINO). host_allocator is left populated so GetHostAccessibleAllocator() still returns it:
  // code that specifically needs host-accessible memory (e.g. staging a CPU input for the EP)
  // behaves the same whether or not the EP also advertised a DEFAULT allocator.
  if (!result.device_allocator && result.host_allocator) {
    result.device_allocator = result.host_allocator;
    result.device_mem_info = result.host_mem_info;
  }

  return result;
}

struct DeviceInterface {
  virtual ~DeviceInterface() {}

  virtual DeviceType GetType() const = 0;
  virtual void InitOrt(const OrtApi& api, Ort::Allocator& allocator) = 0;
  virtual Ort::Allocator& GetAllocator() = 0;

  // §11 host-accessible allocator. EPs that expose a pinned / mappable host allocator (e.g. CUDA
  // host-pinned staging) return the env's shared allocator for their HOST_ACCESSIBLE mem-info.
  // Default nullptr => callers take the EP-agnostic fallback (a plain host staging buffer with
  // device<->CPU copies).
  virtual Ort::Allocator* GetHostAccessibleAllocator() { return nullptr; }

  // §6: Returns the OrtEpDevice entries for this interface's EP on genai's env.
  // Non-empty => plugin mode (shared allocator fetched on demand).
  // Empty => legacy mode (EnsureDeviceOrtInit bootstrap).
  // Default returns empty (most EPs have no plugin path yet).
  virtual std::vector<const OrtEpDevice*> FindMyEpDevices() const { return {}; }

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
  virtual void GetAvailableMemory(size_t& /* free_bytes */, size_t& /* total_bytes */) { assert(false); }

  // Allow each EP to shape the trivial init-session ProviderOptions used by EnsureDeviceOrtInit.
  // The default does nothing; EPs that need global singletons configured (e.g. WebGPU) or
  // allocator gating options (e.g. QNN) override this. `user_options` is the user-supplied entry
  // for this provider from config.model.decoder.session_options.provider_options, or nullptr if
  // the user did not provide one.
  virtual void ShapeInitSessionProviderOptions(ProviderOptions& /*init_options*/,
                                               const ProviderOptions* /*user_options*/) const {}

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
