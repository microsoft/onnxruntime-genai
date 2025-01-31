// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "../search.h"
#include "interface.h"

namespace Generators {
namespace QNN {

static Ort::Allocator* ort_allocator_{};
const char* device_label = "qnn";

struct QnnMemory final : DeviceBuffer {
  QnnMemory(size_t size) : owned_{true} {
    size_in_bytes_ = size;
    p_cpu_ = p_device_ = static_cast<uint8_t*>(ort_allocator_->Alloc(size_in_bytes_));
  }

  QnnMemory(void* p, size_t size) : owned_{false} {
    size_in_bytes_ = size;
    p_cpu_ = p_device_ = static_cast<uint8_t*>(p);
  }

  ~QnnMemory() override {
    if (owned_)
      ort_allocator_->Free(p_device_);
  }

  const char* GetType() const override { return device_label; }
  void AllocateCpu() override {}      // Nothing to do, device memory is CPU accessible
  void CopyDeviceToCpu() override {}  // Nothing to do, device memory is CPU accessible
  void CopyCpuToDevice() override {}  // Nothing to do, device memory is CPU accessible
  void CopyFrom(size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) override {
    CopyThroughCpu(*this, begin_dest, source, begin_source, size_in_bytes);
  }

  void Zero() override {
    memset(p_device_, 0, size_in_bytes_);
  }

  bool owned_;
};

struct InterfaceImpl : DeviceInterface {
  InterfaceImpl() {
  }

  DeviceType GetType() const override { return DeviceType::QNN; }

  void InitOrt(const OrtApi& /*api*/, Ort::Allocator& allocator) override {
    assert(!ort_allocator_);
    ort_allocator_ = &allocator;
  }

  Ort::Allocator& GetAllocator() override {
    return *ort_allocator_;
  }

  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return std::make_shared<QnnMemory>(size);
  }

  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* p, size_t size) override {
    return std::make_shared<QnnMemory>(p, size);
  }

  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override { return std::make_unique<GreedySearch_Cpu>(params); }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override { return std::make_unique<BeamSearch_Cpu>(params); }

  void Synchronize() override {}  // Nothing to do
};

}  // namespace QNN

DeviceInterface* GetQNNInterface() {
  static std::unique_ptr<DeviceInterface> g_device = std::make_unique<QNN::InterfaceImpl>();
  return g_device.get();
}

}  // namespace Generators
