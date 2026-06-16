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

  std::unique_ptr<OrtMemoryInfo> GetMemoryInfo() const override {
    // Note: "QnnHtpShared" allocator is the correct name even when using the GPU backend. Eventually, the plan is to
    // migrate to "QnnShared".
    return OrtMemoryInfo::Create("QnnHtpShared", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
  }

  Config::ProviderOptions GetProviderOptionsForAllocatorSession(const Config& config) const override {
    auto provider_options = Config::ProviderOptions{"QNN", {}};

    const auto& config_providers = config.model.decoder.session_options.providers;
    const auto& config_provider_options = config.model.decoder.session_options.provider_options;

    // Copy the config QNN provider options to the allocator session.
    // Certain options (e.g. "enable_htp_shared_memory_allocator") are required for QNN EP to expose an allocator.
    auto it = std::find_if(config_providers.begin(), config_providers.end(), [](const std::string& p) { return p == "QNN"; });
    if (it != config_providers.end()) {
      const auto i = std::distance(config_providers.begin(), it);
      if (config_provider_options.size() > static_cast<size_t>(i)) {
        for (const auto& pair : config_provider_options[i].options) {
          provider_options.options.emplace_back(pair);
        }
      }
    }

    return provider_options;
  }

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

bool IsQNNGPUBackend(const Config& config) {
  const auto& provider_options = config.model.decoder.session_options.provider_options;
  for (const auto& po : provider_options) {
    if (po.name == "QNN") {
      if (po.device_filtering_options) {
        const auto device_type = po.device_filtering_options->hardware_device_type;
        return device_type == OrtHardwareDeviceType_GPU;
      }

      for (const auto& option : po.options) {
        if (option.first == "backend_type") {
          return option.second == "gpu";
        }
      }
    }
  }
  return false;
}

bool IsQNNStatefulModel(const Model& model) {
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
  return false;
}

}  // namespace Generators
