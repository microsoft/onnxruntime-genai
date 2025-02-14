// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "../search.h"
#include "interface.h"

namespace Generators {
namespace WebGPU {

static Ort::Allocator* ort_allocator_{};
const char* device_label = "WebGPU";

struct WebGPUMemory final : DeviceBuffer {
  WebGPUMemory(size_t size) : owned_{true} {
    size_in_bytes_ = size;
    p_cpu_ = p_device_ = static_cast<uint8_t*>(ort_allocator_->Alloc(size_in_bytes_));
  }

  WebGPUMemory(void* p, size_t size) : owned_{false} {
    size_in_bytes_ = size;
    p_cpu_ = p_device_ = static_cast<uint8_t*>(p);
  }

  ~WebGPUMemory() override {
    if (owned_)
      ort_allocator_->Free(p_device_);
  }

  const char* GetType() const override { return device_label; }
  void AllocateCpu() override { throw std::runtime_error("CPU can't access WebGPU memory"); }
  void CopyDeviceToCpu() override { throw std::runtime_error("CPU can't access WebGPU memory"); }
  void CopyCpuToDevice() override { throw std::runtime_error("CPU can't access WebGPU memory"); }
  void CopyFrom(size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) override {
    throw std::runtime_error("CPU can't access WebGPU memory");
  }

  void Zero() override {
    throw std::runtime_error("Zeroing not implemented for WebGPU memory");
  }

  bool owned_;
};

struct InterfaceImpl : DeviceInterface {
  InterfaceImpl() {
  }

  DeviceType GetType() const override { return DeviceType::WEBGPU; }

  void InitOrt(const OrtApi& /*api*/, Ort::Allocator& allocator) override {
    assert(!ort_allocator_);
    ort_allocator_ = &allocator;
  }

  Ort::Allocator& GetAllocator() override {
    return *ort_allocator_;
  }

  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return std::make_shared<WebGPUMemory>(size);
  }

  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* p, size_t size) override {
    return std::make_shared<WebGPUMemory>(p, size);
  }

  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override { return std::make_unique<GreedySearch_Cpu>(params); }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override { return std::make_unique<BeamSearch_Cpu>(params); }

  void Synchronize() override {}  // Nothing to do?
};

}  // namespace WebGPU

DeviceInterface* GetWebGPUInterface() {
  static std::unique_ptr<DeviceInterface> g_device = std::make_unique<WebGPU::InterfaceImpl>();
  return g_device.get();
}

}  // namespace Generators
