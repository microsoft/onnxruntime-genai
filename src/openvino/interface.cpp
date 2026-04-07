// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "../search.h"
#include "interface.h"
#include "../cpu/interface.h"
#include "../models/model.h"

namespace Generators {
namespace OpenVINO {

struct InterfaceImpl : DeviceInterface {
  InterfaceImpl() {
  }

  DeviceType GetType() const override { return DeviceType::OpenVINO; }

  void InitOrt(const OrtApi& /*api*/, Ort::Allocator& allocator) override {
    // since we use the CPU interface for allocation (right now), InitOrt should not be getting called.
    assert(false);
  }

  Ort::Allocator& GetAllocator() override {
    return GetCpuInterface()->GetAllocator();
  }

  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return GetCpuInterface()->AllocateBase(size);
  }

  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* p, size_t size) override {
    return GetCpuInterface()->WrapMemoryBase(p, size);
  }

  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override { return std::make_unique<GreedySearch_Cpu>(params); }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override { return std::make_unique<BeamSearch_Cpu>(params); }

  void Synchronize() override {}  // Nothing to do
};

}  // namespace OpenVINO

DeviceInterface* GetOpenVINOInterface() {
  static std::unique_ptr<DeviceInterface> g_device = std::make_unique<OpenVINO::InterfaceImpl>();
  return g_device.get();
}

bool IsOpenVINOStatefulModel(const Model& model) {
  if (model.p_device_->GetType() == DeviceType::OpenVINO) {
    const auto& provider_options = model.config_->model.decoder.session_options.provider_options;
    for (auto& po : provider_options) {
      if (po.name == "OpenVINO") {
        const auto& openvino_options = po.options;
        for (auto& option : openvino_options) {
          // For OpenVINO, if session option 'enable_causallm' is set, the session will encapsulate
          // a stateful model, so KVCache will be managed internally.
          if (option.first == "enable_causallm" && option.second == "True") {
            return true;
          }
        }
      }
    }
  }

  return false;
}

}  // namespace Generators
