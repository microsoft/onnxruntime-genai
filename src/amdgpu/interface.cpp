// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// GPU-resident KV cache for the AMDGPU device.
//
// p_device_ is an opaque device-buffer handle, so no pointer arithmetic or
// dereference. All copies go through ORT's CopyTensors. Offset copies fall
// back to CPU staging via CopyThroughCpu.

#include "../generators.h"
#include "../search.h"
#include "interface.h"

#include <stdexcept>

namespace Generators {
namespace AMDGPU {

const char* device_label = "amdgpu";
const char* label_cpu = "cpu";

struct GpuMemory final : DeviceBuffer {
  GpuMemory(size_t size, Ort::Allocator* allocator, const OrtMemoryInfo* memory_info)
      : owned_{true}, ort_allocator_{allocator}, ort_memory_info_{memory_info} {
    size_in_bytes_ = size;
    p_device_ = static_cast<uint8_t*>(ort_allocator_->Alloc(size_in_bytes_));
  }

  GpuMemory(void* p, size_t size, Ort::Allocator* allocator, const OrtMemoryInfo* memory_info)
      : owned_{false}, ort_allocator_{allocator}, ort_memory_info_{memory_info} {
    size_in_bytes_ = size;
    p_device_ = static_cast<uint8_t*>(p);
  }

  ~GpuMemory() override {
    if (owned_)
      ort_allocator_->Free(p_device_);
    if (p_cpu_)
      free(p_cpu_);
  }

  const char* GetType() const override { return device_label; }

  void AllocateCpu() override {
    if (!p_cpu_)
      p_cpu_ = static_cast<uint8_t*>(malloc(size_in_bytes_));
  }

  void CopyDeviceToCpu() override {
    if (!ort_allocator_)
      throw std::runtime_error("AMDGPU allocator not initialized");

    AllocateCpu();

    int64_t shape_val = static_cast<int64_t>(size_in_bytes_);
    std::span<const int64_t> shape{&shape_val, 1};
    auto src_tensor = OrtValue::CreateTensor(*ort_memory_info_, p_device_, size_in_bytes_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    auto cpu_mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto dst_tensor = OrtValue::CreateTensor(*cpu_mem_info, p_cpu_, size_in_bytes_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    const std::vector<const OrtValue*> src_ptrs = {src_tensor.get()};
    const std::vector<OrtValue*> dst_ptrs = {dst_tensor.get()};
    GetOrtEnv().CopyTensors(src_ptrs, dst_ptrs, nullptr);
  }

  void CopyCpuToDevice() override {
    if (!ort_allocator_)
      throw std::runtime_error("AMDGPU allocator not initialized");
    assert(p_cpu_);

    int64_t shape_val = static_cast<int64_t>(size_in_bytes_);
    std::span<const int64_t> shape{&shape_val, 1};
    auto cpu_mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto src_tensor = OrtValue::CreateTensor(*cpu_mem_info, p_cpu_, size_in_bytes_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    auto dst_tensor = OrtValue::CreateTensor(*ort_memory_info_, p_device_, size_in_bytes_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    const std::vector<const OrtValue*> src_ptrs = {src_tensor.get()};
    const std::vector<OrtValue*> dst_ptrs = {dst_tensor.get()};
    GetOrtEnv().CopyTensors(src_ptrs, dst_ptrs, nullptr);
  }

  void CopyFrom(size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) override {
    if (!ort_allocator_)
      throw std::runtime_error("AMDGPU allocator not initialized");

    // Opaque handle, so wrap the whole buffer and let CopyTensors do the copy.
    if (strcmp(source.GetType(), device_label) == 0 && begin_source == 0 && begin_dest == 0) {
      // Full-buffer device-to-device copy.
      int64_t shape_val = static_cast<int64_t>(size_in_bytes);
      std::span<const int64_t> shape{&shape_val, 1};
      auto src_tensor = OrtValue::CreateTensor(*ort_memory_info_, source.p_device_, size_in_bytes, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
      auto dst_tensor = OrtValue::CreateTensor(*ort_memory_info_, p_device_, size_in_bytes, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

      const std::vector<const OrtValue*> src_ptrs = {src_tensor.get()};
      const std::vector<OrtValue*> dst_ptrs = {dst_tensor.get()};
      GetOrtEnv().CopyTensors(src_ptrs, dst_ptrs, nullptr);
    } else if (strcmp(source.GetType(), label_cpu) == 0 && begin_source == 0 && begin_dest == 0) {
      // Full-buffer CPU-to-device copy.
      int64_t shape_val = static_cast<int64_t>(size_in_bytes);
      std::span<const int64_t> shape{&shape_val, 1};
      auto cpu_mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
      auto src_tensor = OrtValue::CreateTensor(*cpu_mem_info, source.p_device_, size_in_bytes, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
      auto dst_tensor = OrtValue::CreateTensor(*ort_memory_info_, p_device_, size_in_bytes, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

      const std::vector<const OrtValue*> src_ptrs = {src_tensor.get()};
      const std::vector<OrtValue*> dst_ptrs = {dst_tensor.get()};
      GetOrtEnv().CopyTensors(src_ptrs, dst_ptrs, nullptr);
    } else {
      // Offset copies can't sub-buffer-view an opaque handle, so stage through CPU.
      CopyThroughCpu(*this, begin_dest, source, begin_source, size_in_bytes);
    }
  }

  void Zero() override {
    if (!ort_allocator_)
      throw std::runtime_error("AMDGPU allocator not initialized");

    // TODO: device-side zero to avoid the host staging buffer. Off the decode hot path for now.
    std::vector<uint8_t> zero_buffer(size_in_bytes_, 0);

    int64_t shape_val = static_cast<int64_t>(size_in_bytes_);
    std::span<const int64_t> shape{&shape_val, 1};
    auto cpu_mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto src_tensor = OrtValue::CreateTensor(*cpu_mem_info, zero_buffer.data(), size_in_bytes_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    auto dst_tensor = OrtValue::CreateTensor(*ort_memory_info_, p_device_, size_in_bytes_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    const std::vector<const OrtValue*> src_ptrs = {src_tensor.get()};
    const std::vector<OrtValue*> dst_ptrs = {dst_tensor.get()};
    GetOrtEnv().CopyTensors(src_ptrs, dst_ptrs, nullptr);
  }

  bool owned_;  // If we own the memory, we free it on destruction
  Ort::Allocator* ort_allocator_;
  const OrtMemoryInfo* ort_memory_info_;
};

const char* pinned_device_label = "amdgpu_pinned";

// DeviceBuffer over a host-accessible allocation: one mapped pointer is both CPU-writable
// and GPU-readable, so p_cpu_ aliases p_device_ and the copy methods are no-ops. Used for
// decode inputs so the CPU updates them in place with no roundtrip.
struct PinnedMemory final : DeviceBuffer {
  PinnedMemory(size_t size, Ort::Allocator* allocator) : owned_{true}, ort_allocator_{allocator} {
    size_in_bytes_ = size;
    p_device_ = static_cast<uint8_t*>(ort_allocator_->Alloc(size_in_bytes_));
    p_cpu_ = p_device_;  // mapped: one pointer for both
  }

  PinnedMemory(void* p, size_t size, Ort::Allocator* allocator) : owned_{false}, ort_allocator_{allocator} {
    size_in_bytes_ = size;
    p_device_ = static_cast<uint8_t*>(p);
    p_cpu_ = p_device_;
  }

  ~PinnedMemory() override {
    if (owned_)
      ort_allocator_->Free(p_device_);
    // p_cpu_ aliases p_device_, do not free it separately.
  }

  const char* GetType() const override { return pinned_device_label; }
  void AllocateCpu() override {}       // p_cpu_ already valid (== p_device_)
  void CopyDeviceToCpu() override {}   // same memory: nothing to copy
  void CopyCpuToDevice() override {}

  void CopyFrom(size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) override {
    // Pinned memory is CPU-addressable; source may be on any device.
    CopyThroughCpu(*this, begin_dest, source, begin_source, size_in_bytes);
  }

  void Zero() override { memset(p_device_, 0, size_in_bytes_); }

  bool owned_;
  Ort::Allocator* ort_allocator_;
};

struct InterfaceImpl : DeviceInterface {
  DeviceType GetType() const override { return DeviceType::AMDGPU; }

  void InitOrt(const OrtApi& api, Ort::Allocator& allocator) override {
    Ort::api = &api;
    assert(!ort_allocator_);
    ort_allocator_ = &allocator;
    // Cache the memory info so tensors wrapping p_device_ carry the allocator's device attributes.
    ort_memory_info_ = &ort_allocator_->GetInfo();
  }

  Ort::Allocator& GetAllocator() override {
    return *ort_allocator_;
  }

  void InitHostAccessible(Ort::Allocator& allocator) override {
    ort_pinned_allocator_ = &allocator;
  }

  Ort::Allocator* GetHostAccessibleAllocator() override {
    return ort_pinned_allocator_;
  }

  Ort::Allocator* PinnedAllocator() const { return ort_pinned_allocator_; }

  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return std::make_shared<GpuMemory>(size, ort_allocator_, ort_memory_info_);
  }

  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* p, size_t size) override {
    return std::make_shared<GpuMemory>(p, size, ort_allocator_, ort_memory_info_);
  }

  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override {
    return GetDeviceInterface(DeviceType::CPU)->CreateGreedy(params);
  }

  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override {
    return GetDeviceInterface(DeviceType::CPU)->CreateBeam(params);
  }

  void Synchronize() override {}

 private:
  Ort::Allocator* ort_allocator_{};
  const OrtMemoryInfo* ort_memory_info_{};
  // Host-accessible allocator, set by InitHostAccessible when one is available.
  Ort::Allocator* ort_pinned_allocator_{};
};

// Inputs-only interface: allocations come from the host-accessible allocator, everything else
// delegates to the base AMDGPU interface. Updates run on the CPU in place.
struct PinnedInputsImpl : DeviceInterface {
  explicit PinnedInputsImpl(InterfaceImpl& base) : base_{base} {}

  DeviceType GetType() const override { return DeviceType::AMDGPU; }
  void InitOrt(const OrtApi& api, Ort::Allocator& allocator) override { base_.InitOrt(api, allocator); }
  Ort::Allocator& GetAllocator() override { return *base_.PinnedAllocator(); }
  Ort::Allocator* GetHostAccessibleAllocator() override { return base_.PinnedAllocator(); }

  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return std::make_shared<PinnedMemory>(size, base_.PinnedAllocator());
  }
  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* p, size_t size) override {
    return std::make_shared<PinnedMemory>(p, size, base_.PinnedAllocator());
  }

  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override { return base_.CreateGreedy(params); }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override { return base_.CreateBeam(params); }
  void Synchronize() override { base_.Synchronize(); }

  bool Cast(void* input, void* output, ONNXTensorElementDataType input_type,
            ONNXTensorElementDataType output_type, size_t element_count) override {
    return base_.Cast(input, output, input_type, output_type, element_count);
  }

  // In-place CPU updates on the pinned buffer. Delegate to the CPU interface and report
  // success so the caller skips its copy-back path.
  bool UpdatePositionIds(void* position_ids, int batch_beam_size, int total_length,
                         int new_kv_length, ONNXTensorElementDataType type) override {
    return GetDeviceInterface(DeviceType::CPU)->UpdatePositionIds(position_ids, batch_beam_size, total_length, new_kv_length, type);
  }
  bool UpdateAttentionMask(void* next_mask_data, void* mask_data, int batch_beam_size, int new_kv_length,
                           int total_length, int max_length, bool update_only,
                           ONNXTensorElementDataType type) override {
    return GetDeviceInterface(DeviceType::CPU)->UpdateAttentionMask(next_mask_data, mask_data, batch_beam_size, new_kv_length,
                                                  total_length, max_length, update_only, type);
  }

  InterfaceImpl& base_;
};

}  // namespace AMDGPU

static std::unique_ptr<AMDGPU::InterfaceImpl> g_amdgpu_device;
static std::unique_ptr<AMDGPU::PinnedInputsImpl> g_amdgpu_pinned_inputs;

DeviceInterface* GetAMDGPUInterface() {
  if (!g_amdgpu_device)
    g_amdgpu_device = std::make_unique<AMDGPU::InterfaceImpl>();
  return g_amdgpu_device.get();
}

DeviceInterface* GetAMDGPUPinnedInputsInterface() {
  auto* base = static_cast<AMDGPU::InterfaceImpl*>(GetAMDGPUInterface());
  if (!base->GetHostAccessibleAllocator())
    return nullptr;  // no pinned allocator, caller falls back to the device interface
  if (!g_amdgpu_pinned_inputs)
    g_amdgpu_pinned_inputs = std::make_unique<AMDGPU::PinnedInputsImpl>(*base);
  return g_amdgpu_pinned_inputs.get();
}

}  // namespace Generators
