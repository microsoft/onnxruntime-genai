// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "../search.h"
#include "../models/utils.h"
#include "interface.h"
#include <cstring>
#include <functional>
#include <unordered_map>

namespace Generators {

const char* label_cpu = "cpu";

struct CpuMemory final : DeviceBuffer {
  CpuMemory(Ort::Allocator& allocator, size_t size) 
    : allocator_(&allocator), owned_(true) {
    size_in_bytes_ = size;
    p_cpu_ = p_device_ = static_cast<uint8_t*>(allocator_->Alloc(size_in_bytes_));
    if (!p_device_) {
      throw std::bad_alloc();
    }
  }

  CpuMemory(void* p, size_t size) : owned_(false) {
    if (reinterpret_cast<uintptr_t>(p) % 64 != 0) {
      throw std::invalid_argument("Pointer must be 64-byte aligned");
    }
    size_in_bytes_ = size;
    p_cpu_ = p_device_ = static_cast<uint8_t*>(p);
  }

  ~CpuMemory() noexcept override {
    if (owned_ && allocator_) {
      try {
        allocator_->Free(p_device_);
      } catch (...) {
        // Log or handle exception if needed
      }
    }
  }

  const char* GetType() const noexcept override { return label_cpu; }
  void AllocateCpu() noexcept override {}      // Nothing to do, device is also CPU
  void CopyDeviceToCpu() noexcept override {}  // Nothing to do, device is also CPU
  void CopyCpuToDevice() noexcept override {}  // Nothing to do, device is also CPU
  
  void CopyFrom(size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) override {
    if (begin_dest + size_in_bytes > size_in_bytes_ || 
        begin_source + size_in_bytes > source.GetSize()) {
      throw std::out_of_range("CopyFrom: Source or destination out of range");
    }
    
    void* src_ptr = source.GetCpuMemory() + begin_source;
    std::memcpy(p_device_ + begin_dest, src_ptr, size_in_bytes);
  }

  void Zero() noexcept override {
    std::memset(p_device_, 0, size_in_bytes_);
  }

  Ort::Allocator* allocator_ = nullptr;
  bool owned_;
};

struct CpuInterface : DeviceInterface {
  CpuInterface() {
    RegisterCast<float, Ort::Float16_t>(FastFloat32ToFloat16);
    RegisterCast<Ort::Float16_t, float>(FastFloat16ToFloat32);
    RegisterCast<int32_t, int64_t>([](int32_t v) -> int64_t { return v; });
  }

  DeviceType GetType() const noexcept override { return DeviceType::CPU; }

  void InitOrt(const OrtApi& /*api*/, Ort::Allocator& allocator) override {
    if (ort_allocator_) {
      throw std::runtime_error("Ort allocator already initialized");
    }
    ort_allocator_ = &allocator;
  }

  Ort::Allocator& GetAllocator() override {
    if (!ort_allocator_) {
      throw std::logic_error("Ort allocator not initialized");
    }
    return *ort_allocator_;
  }

  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return std::make_shared<CpuMemory>(GetAllocator(), size);
  }

  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* p, size_t size) override {
    return std::make_shared<CpuMemory>(p, size);
  }

  template <typename From, typename To>
  void RegisterCast(std::function<To(From)> converter) {
    auto key = std::make_pair(Ort::TypeToTensorType<From>(), 
                             Ort::TypeToTensorType<To>());
    cast_handlers_[key] = [converter](void* src, void* dst, size_t n) {
      auto* src_ptr = static_cast<From*>(src);
      auto* dst_ptr = static_cast<To*>(dst);
      for (size_t i = 0; i < n; ++i) {
        dst_ptr[i] = converter(src_ptr[i]);
      }
    };
  }

  bool Cast(void* input_data, void* output_data, 
            ONNXTensorElementDataType input_type, 
            ONNXTensorElementDataType output_type, 
            size_t element_count) override {
    if (input_type == output_type) {
      if (input_data != output_data) {
        std::memcpy(output_data, input_data, element_count * GetElementSize(input_type));
      }
      return true;
    }

    auto key = std::make_pair(input_type, output_type);
    auto it = cast_handlers_.find(key);
    if (it != cast_handlers_.end()) {
      it->second(input_data, output_data, element_count);
      return true;
    }

    throw std::runtime_error("Cast - Unimplemented cast from " + 
                             std::to_string(input_type) + " to " + 
                             std::to_string(output_type));
  }

  template <typename T>
  void UpdatePositionIds(T* position_ids, int batch_beam_size, int seq_length, int total_length, int new_kv_length) noexcept {
    if (seq_length == 1) {
      // For batch size == 1 we calculate position ids with total length and new kv length
      for (int i = 0; i < new_kv_length; i++) {
        position_ids[i] = i + total_length - new_kv_length;
      }
    } else {
      // For batch size > 1 we update each sequence position
      for (int i = 0; i < batch_beam_size; i++) {
        auto* pos = position_ids + i * seq_length;
        for (int j = 0; j < new_kv_length; j++) {
          pos[j] = total_length - new_kv_length + j;
        }
      }
    }
  }

  bool UpdatePositionIds(void* position_ids, int batch_beam_size, int seq_length, int total_length, int new_kv_length, ONNXTensorElementDataType type) noexcept override {
    if (type == Ort::TypeToTensorType<int32_t>()) {
      UpdatePositionIds(static_cast<int32_t*>(position_ids), batch_beam_size, seq_length, total_length, new_kv_length);
    } else if (type == Ort::TypeToTensorType<int64_t>()) {
      UpdatePositionIds(static_cast<int64_t*>(position_ids), batch_beam_size, seq_length, total_length, new_kv_length);
    } else {
      return false;
    }
    return true;
  }

  template <typename T>
  void UpdateAttentionMask(T* next_mask_data, T* mask_data, int batch_beam_size, int total_length) noexcept {
    if (batch_beam_size == 1) {
      // For batch size == 1 we assume no padding
      std::fill_n(next_mask_data, total_length, static_cast<T>(1));
    } else {
      // For batch size > 1 we increment attention mask
      for (int i = 0; i < batch_beam_size; i++) {
        auto* src_row = mask_data + i * (total_length - 1);
        auto* dst_row = next_mask_data + i * total_length;
        std::copy(src_row, src_row + total_length - 1, dst_row);
        dst_row[total_length - 1] = static_cast<T>(1);
      }
    }
  }

  template <typename T>
  void UpdateAttentionMaskStatic(T* mask_data, int batch_beam_size, int new_kv_length, int total_length, int max_length) noexcept {
    const int start_col = total_length - new_kv_length;
    for (int i = 0; i < batch_beam_size; i++) {
      auto* mask_row = mask_data + i * max_length;
      std::fill_n(mask_row + start_col, new_kv_length, static_cast<T>(1));
    }
  }

  bool UpdateAttentionMask(void* next_mask_data, void* mask_data, int batch_beam_size, int new_kv_length, int total_length, int max_length, bool update_only, ONNXTensorElementDataType type) noexcept override {
    if (update_only) {
      if (type == Ort::TypeToTensorType<int32_t>()) {
        UpdateAttentionMaskStatic(static_cast<int32_t*>(mask_data), batch_beam_size, new_kv_length, total_length, max_length);
      } else if (type == Ort::TypeToTensorType<int64_t>()) {
        UpdateAttentionMaskStatic(static_cast<int64_t*>(mask_data), batch_beam_size, new_kv_length, total_length, max_length);
      } else {
        return false;
      }
    } else {
      if (type == Ort::TypeToTensorType<int32_t>()) {
        UpdateAttentionMask(static_cast<int32_t*>(next_mask_data), static_cast<int32_t*>(mask_data), batch_beam_size, total_length);
      } else if (type == Ort::TypeToTensorType<int64_t>()) {
        UpdateAttentionMask(static_cast<int64_t*>(next_mask_data), static_cast<int64_t*>(mask_data), batch_beam_size, total_length);
      } else {
        return false;
      }
    }
    return true;
  }

  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override { 
    return std::make_unique<GreedySearch_Cpu>(params); 
  }
  
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override { 
    return std::make_unique<BeamSearch_Cpu>(params); 
  }

  void Synchronize() noexcept override {}  // Nothing to do for CPU

private:
  Ort::Allocator* ort_allocator_ = nullptr;
  
  using CastHandler = std::function<void(void*, void*, size_t)>;
  std::unordered_map<std::pair<int, int>, CastHandler> cast_handlers_;

  static size_t GetElementSize(ONNXTensorElementDataType type) {
    switch (type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return sizeof(float);
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return sizeof(uint16_t);
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return sizeof(int32_t);
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return sizeof(int64_t);
      default: throw std::runtime_error("Unsupported data type");
    }
  }
};

DeviceInterface* GetCpuInterface() {
  static CpuInterface g_cpu;
  return &g_cpu;
}

}  // namespace Generators