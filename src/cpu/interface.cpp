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

  bool Cast(OrtValue& input, OrtValue& output) override {
    auto input_info = input.GetTensorTypeAndShapeInfo();
    auto output_info = output.GetTensorTypeAndShapeInfo();

    auto input_type = input_info->GetElementType();
    auto output_type = output_info->GetElementType();

    auto element_count = input_info->GetElementCount();
    if (element_count != output_info->GetElementCount())
      throw std::runtime_error("Cast - input and output element counts do not match");
    if (input_type == output_type)
      throw std::runtime_error("Cast - input and output types are the same");

    if (input_type == Ort::TypeToTensorType<float> && output_type == Ort::TypeToTensorType<Ort::Float16_t>) {
      auto* fp32 = input.GetTensorData<float>();
      auto* fp16 = output.GetTensorMutableData<uint16_t>();
      for (size_t i = 0; i < element_count; i++)
        fp16[i] = FastFloat32ToFloat16(fp32[i]);
    } else if (input_type == Ort::TypeToTensorType<Ort::Float16_t> && output_type == Ort::TypeToTensorType<float>) {
      auto* fp16 = input.GetTensorData<uint16_t>();
      auto* fp32 = output.GetTensorMutableData<float>();
      for (size_t i = 0; i < element_count; i++)
        fp32[i] = FastFloat16ToFloat32(fp16[i]);
    } else if (input_type == Ort::TypeToTensorType<int32_t> && output_type == Ort::TypeToTensorType<int64_t>) {
      auto* input_data = input.GetTensorData<int32_t>();
      auto* output_data = output.GetTensorMutableData<int64_t>();
      for (size_t i = 0; i < element_count; i++)
        output_data[i] = input_data[i];
    } else
      throw std::runtime_error("Cast - Unimplemented cast");
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
