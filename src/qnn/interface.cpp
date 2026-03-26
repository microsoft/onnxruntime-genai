// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "../search.h"
#include "../models/model.h"
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

bool IsQNNStatefulModel(const Model& model) {
  // Check for both QNN and CPU device types
  // When using QNN EP with genai_model=True, the model is stateful regardless of device type (QNN/CPU)
  // For QNN models with enable_htp_shared_memory_allocator=1, p_device_ will be QNN type
  // For QNN models without shared memory allocator, p_device_ will be CPU type
  // Both cases need to be handled the same way for stateful models where KV cache is managed internally
  if (model.p_device_->GetType() == DeviceType::QNN || model.p_device_->GetType() == DeviceType::CPU) {
    const auto& provider_options = model.config_->model.decoder.session_options.provider_options;
    for (const auto& po : provider_options) {
      if (po.name == "QNN") {
        for (const auto& option : po.options) {
          // For QNN, if session option 'genie_model' is set to true, the session will encapsulate
          // a stateful model, so KVCache will be managed internally.
          if (option.first == "genie_model") {
            std::string lower_value(option.second);
            std::transform(lower_value.begin(), lower_value.end(), lower_value.begin(),
              [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });
            return lower_value == "true";
          }
        }
      }
    }
  }

  return false;
}

}  // namespace Generators
