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

struct QnnInterfaceBase : DeviceInterface {
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

struct HtpInterfaceImpl : QnnInterfaceBase {
  HtpInterfaceImpl() {}
  DeviceType GetType() const override { return DeviceType::QnnHtp; }

  void ShapeInitSessionProviderOptions(Config::ProviderOptions& init_options,
                                       const Config::ProviderOptions* user_options) const override {
    // NOTE: QNN EP currently exposes two allocators (HTP shared memory allocator and DX12 shared memory allocator), with
    //       the first only being supported with the HTP backend and the second only supported by the GPU backend.
    //       As a result, note the following:
    //         1.) We only look to copy over the "enable_htp_shared_memory_allocator" option here, and likewise only look for
    //             "enable_dx12_shared_memory_allocator" in `GpuInterfaceImpl::ShapeInitSessionProviderOptions`.
    //         2.) Oga keeps one global allocator for HTP and one for GPU, each of which is created once. Because each device
    //             only supports once allocator, the fact that each device's chosen allocator is sticky is ok.
    for (const auto& opt : user_options->options) {
      if (opt.first == "enable_htp_shared_memory_allocator") {
        init_options.options.emplace_back(opt);
      }
    }

    init_options.device_filtering_options = Generators::DeviceFilteringOptions{OrtHardwareDeviceType_NPU};
  }
};

struct GpuInterfaceImpl : QnnInterfaceBase {
  GpuInterfaceImpl() {}
  DeviceType GetType() const override { return DeviceType::QnnGpu; }

  void ShapeInitSessionProviderOptions(Config::ProviderOptions& init_options,
                                       const Config::ProviderOptions* user_options) const override {
    for (const auto& opt : user_options->options) {
      if (opt.first == "enable_dx12_shared_memory_allocator") {
        init_options.options.emplace_back(opt);
      }
    }

    init_options.device_filtering_options = Generators::DeviceFilteringOptions{OrtHardwareDeviceType_GPU};
  }
};

}  // namespace QNN

DeviceInterface* GetQNNInterface(DeviceType device_type) {
  assert(device_type == DeviceType::QnnHtp || device_type == DeviceType::QnnGpu);

  static std::unique_ptr<DeviceInterface> g_htp_device = std::make_unique<QNN::HtpInterfaceImpl>();
  static std::unique_ptr<DeviceInterface> g_gpu_device = std::make_unique<QNN::GpuInterfaceImpl>();
  switch (device_type) {
    case DeviceType::QnnHtp:
      return g_htp_device.get();
    case DeviceType::QnnGpu:
      return g_gpu_device.get();
    default:
      return nullptr;
  }
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
