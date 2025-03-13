// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "../search.h"
#include "../models/utils.h"
#include "interface.h"

namespace Generators {

static Ort::Allocator* ort_allocator_{};
const char* label_cpu = "cpu";

struct CpuMemory final : DeviceBuffer {
  CpuMemory(size_t size) : owned_{true} {
    size_in_bytes_ = size;
    p_cpu_ = p_device_ = static_cast<uint8_t*>(ort_allocator_->Alloc(size_in_bytes_));
  }

  CpuMemory(void* p, size_t size) : owned_{false} {
    size_in_bytes_ = size;
    p_cpu_ = p_device_ = static_cast<uint8_t*>(p);
  }

  ~CpuMemory() override {
    if (owned_)
      ort_allocator_->Free(p_device_);
  }

  const char* GetType() const override { return label_cpu; }
  void AllocateCpu() override {}      // Nothing to do, device is also CPU
  void CopyDeviceToCpu() override {}  // Nothing to do, device is also CPU
  void CopyCpuToDevice() override {}  // Nothing to do, device is also CPU
  void CopyFrom(size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) override {
    CopyThroughCpu(*this, begin_dest, source, begin_source, size_in_bytes);
  }

  void Zero() override {
    memset(p_device_, 0, size_in_bytes_);
  }

  bool owned_;
};

struct CpuInterface : DeviceInterface {
  CpuInterface() {
  }

  DeviceType GetType() const override { return DeviceType::CPU; }

  void InitOrt(const OrtApi& /*api*/, Ort::Allocator& allocator) override {
    assert(!ort_allocator_);
    ort_allocator_ = &allocator;
  }

  Ort::Allocator& GetAllocator() override {
    return *ort_allocator_;
  }

  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return std::make_shared<CpuMemory>(size);
  }

  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* p, size_t size) override {
    return std::make_shared<CpuMemory>(p, size);
  }

  bool Cast(void* input_data, void* output_data, ONNXTensorElementDataType input_type, ONNXTensorElementDataType output_type, size_t element_count) override {
    if (input_type == output_type)
      throw std::runtime_error("Cast - input and output types are the same");

    if (input_type == Ort::TypeToTensorType<float> && output_type == Ort::TypeToTensorType<Ort::Float16_t>) {
      auto* fp32 = static_cast<float*>(input_data);
      auto* fp16 = static_cast<uint16_t*>(output_data);
      for (size_t i = 0; i < element_count; i++)
        fp16[i] = FastFloat32ToFloat16(fp32[i]);
    } else if (input_type == Ort::TypeToTensorType<Ort::Float16_t> && output_type == Ort::TypeToTensorType<float>) {
      auto* fp16 = static_cast<uint16_t*>(input_data);
      auto* fp32 = static_cast<float*>(output_data);
      for (size_t i = 0; i < element_count; i++)
        fp32[i] = FastFloat16ToFloat32(fp16[i]);
    } else if (input_type == Ort::TypeToTensorType<int32_t> && output_type == Ort::TypeToTensorType<int64_t>) {
      auto* int32 = static_cast<int32_t*>(input_data);
      auto* int64 = static_cast<int64_t*>(output_data);
      for (size_t i = 0; i < element_count; i++)
        int64[i] = int32[i];
    } else
      throw std::runtime_error("Cast - Unimplemented cast");
    return true;
  }

  template <typename T>
  void UpdatePositionIds(T* position_ids, int batch_beam_size, int total_length, int new_kv_length) {
    if (batch_beam_size == 1) {
      // For batch size == 1 we calculate position ids with total length and new kv length for continuous decoding
      for (int i = 0; i < new_kv_length; i++)
        position_ids[i] = i + total_length - new_kv_length;
    } else {
      // For batch size > 1 we increment position ids by 1... continuous decoding is not supported
      for (int i = 0; i < batch_beam_size; i++)
        position_ids[i]++;
    }
  }

  bool UpdatePositionIds(void* position_ids, int batch_beam_size, int total_length, int new_kv_length, ONNXTensorElementDataType type) override {
    type == Ort::TypeToTensorType<int32_t>
        ? UpdatePositionIds<int32_t>(static_cast<int32_t*>(position_ids), batch_beam_size, total_length, new_kv_length)
        : UpdatePositionIds<int64_t>(static_cast<int64_t*>(position_ids), batch_beam_size, total_length, new_kv_length);
    return true;
  }

  template <typename T>
  void UpdateAttentionMask(T* next_mask_data, T* mask_data, int batch_beam_size, int total_length) {
    if (batch_beam_size == 1) {
      // For batch size == 1 we assume no padding. We make this explicit for continuous decoding.
      for (int i = 0; i < total_length; i++)
        next_mask_data[i] = 1;
    } else {
      // For batch size > 1 we increment attention mask by 1... continuous decoding is not supported
      for (int i = 0; i < batch_beam_size; i++) {
        for (int j = 0; j < total_length - 1; j++) {
          next_mask_data[i * total_length + j] = mask_data[i * (total_length - 1) + j];
        }
        next_mask_data[i * total_length + total_length - 1] = 1;
      }
    }
  }

  template <typename T>
  void UpdateAttentionMaskStatic(T* mask_data, int batch_beam_size, int new_kv_length, int total_length, int max_length) {
    for (int i = 0; i < batch_beam_size; i++) {
      for (int j = total_length - new_kv_length; j < total_length; j++) {
        mask_data[i * max_length + j] = 1;
      }
    }
  }

  bool UpdateAttentionMask(void* next_mask_data, void* mask_data, int batch_beam_size, int new_kv_length, int total_length, int max_length, bool update_only, ONNXTensorElementDataType type) override {
    if (update_only) {
      if (type == Ort::TypeToTensorType<int32_t>)
        UpdateAttentionMaskStatic(static_cast<int32_t*>(mask_data), batch_beam_size, new_kv_length, total_length, max_length);
      else
        UpdateAttentionMaskStatic(static_cast<int64_t*>(mask_data), batch_beam_size, new_kv_length, total_length, max_length);
    } else {
      if (type == Ort::TypeToTensorType<int32_t>)
        UpdateAttentionMask(static_cast<int32_t*>(next_mask_data), static_cast<int32_t*>(mask_data), batch_beam_size, total_length);
      else
        UpdateAttentionMask(static_cast<int64_t*>(next_mask_data), static_cast<int64_t*>(mask_data), batch_beam_size, total_length);
    }
    return true;
  }

  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override { return std::make_unique<GreedySearch_Cpu>(params); }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override { return std::make_unique<BeamSearch_Cpu>(params); }

  void Synchronize() override {}  // Nothing to do as CPU is always in sync with itself
};

DeviceInterface* GetCpuInterface() {
  static std::unique_ptr<CpuInterface> g_cpu = std::make_unique<CpuInterface>();
  return g_cpu.get();
}

}  // namespace Generators
