// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "../search.h"
#include "interface.h"

namespace Generators {

static Ort::Allocator* ort_allocator_{};
const char* label_cpu = "cpu";

struct CpuMemory final : DeviceBuffer {
  CpuMemory(size_t size) : owned_{true} {
    size_in_bytes_ = size;
    p_cpu_ = p_device_ = new uint8_t[size_in_bytes_];
  }

  CpuMemory(void* p, size_t size) : owned_{false} {
    size_in_bytes_ = size;
    p_cpu_ = p_device_ = static_cast<uint8_t*>(p);
  }

  ~CpuMemory() override {
    if (owned_)
      delete[] p_device_;
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
  void InitAllocator(Ort::Allocator& allocator) override {
    assert(!ort_allocator_);
    ort_allocator_ = &allocator;
  }

  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return std::make_shared<CpuMemory>(size);
  }

  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* p, size_t size) override {
    return std::make_shared<CpuMemory>(p, size);
  }

  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override { return std::make_unique<GreedySearch_Cpu>(params); }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override { return std::make_unique<BeamSearch_Cpu>(params); }

  void Synchronize() override {}  // Nothing to do as CPU is always in sync with itself
} g_cpu;

DeviceInterface* GetCpuInterface() { return &g_cpu; }

}  // namespace Generators
