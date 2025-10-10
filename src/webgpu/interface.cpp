// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "../search.h"
#include "interface.h"

#ifdef USE_WEBGPU
// USE_WEBGPU path: Build our own Dawn (legacy mode, will be deprecated)
#include "webgpu_update_mask_kernel.h"
#include "webgpu_update_position_ids_kernel.h"
#include "webgpu_cast_kernel.h"

#include <dawn/webgpu_cpp.h>
#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#else
// Preferred path: Use WebGPU EP's Dawn instance
// This enables graph capture and other WebGPU features without building Dawn twice

// Forward declare the C API functions to avoid header conflicts
extern "C" {
  const void* OrtWebGpuGetDawnProcTable(int context_id);
  void* OrtWebGpuGetInstance(int context_id);
  void* OrtWebGpuGetDevice(int context_id);
}

#include "webgpu_update_mask_kernel.h"
#include "webgpu_update_position_ids_kernel.h"
#include "webgpu_cast_kernel.h"

#include <dawn/webgpu_cpp.h>
#include <dawn/dawn_proc.h>
#endif

#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>

namespace Generators {
namespace WebGPU {

namespace {
constexpr size_t NormalizeBufferSize(size_t size) {
  return (size + 15) / 16 * 16;
}
}  // namespace
static Ort::Allocator* ort_allocator_{};
// These pointers will be initialized from WebGPU EP's Dawn when USE_WEBGPU=OFF
// or from our own Dawn when USE_WEBGPU=ON
static wgpu::Device* webgpu_device_{};
static wgpu::Queue* webgpu_queue_{};
static wgpu::Instance* webgpu_instance_{};
const char* device_label = "WebGPU";

struct WebGPUMemory final : DeviceBuffer {
  WebGPUMemory(size_t size) : owned_{true} {
    size_in_bytes_ = size;
    p_device_ = static_cast<uint8_t*>(ort_allocator_->Alloc(size_in_bytes_));
  }

  WebGPUMemory(void* p, size_t size) : owned_{false} {
    size_in_bytes_ = size;
    p_device_ = static_cast<uint8_t*>(p);
  }

  ~WebGPUMemory() override {
    if (owned_)
      ort_allocator_->Free(p_device_);
    if (p_cpu_)
      free(p_cpu_);
  }

  const char* GetType() const override { return device_label; }

  void AllocateCpu() override {
    if (!p_cpu_)
      p_cpu_ = static_cast<uint8_t*>(malloc(size_in_bytes_));
  }

  void CopyDeviceToCpu() override {
#ifdef USE_WEBGPU
    // USE_WEBGPU=ON: webgpu_device_ points to our own Dawn device
    if (!webgpu_device_) {
      throw std::runtime_error("WebGPU device not initialized");
    }
#else
    // USE_WEBGPU=OFF: webgpu_device_ points to WebGPU EP's Dawn device
    if (!webgpu_device_) {
      throw std::runtime_error("WebGPU device not initialized. Ensure WebGPU EP is enabled and InitOrt() was called.");
    }
#endif
    AllocateCpu();
    WGPUBuffer src_buf = reinterpret_cast<WGPUBuffer>(p_device_);
    wgpu::BufferDescriptor desc{};
    desc.size = size_in_bytes_;
    desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;

    auto staging_buffer = webgpu_device_->CreateBuffer(&desc);
    auto command_encoder = webgpu_device_->CreateCommandEncoder();
    command_encoder.CopyBufferToBuffer(src_buf, 0, staging_buffer, 0, size_in_bytes_);
    wgpu::CommandBuffer command_buffer1 = command_encoder.Finish();
    webgpu_queue_->Submit(1, &command_buffer1);

    webgpu_instance_->WaitAny(
        staging_buffer.MapAsync(
            wgpu::MapMode::Read,
            0, size_in_bytes_,
            wgpu::CallbackMode::WaitAnyOnly,
            [&](wgpu::MapAsyncStatus status, wgpu::StringView message) {
              if (status == wgpu::MapAsyncStatus::Success) {
                // Get a pointer to the mapped data
                const void* mapped_data = staging_buffer.GetConstMappedRange(0, size_in_bytes_);
                // Copy the data to our vector
                memcpy(p_cpu_, mapped_data, size_in_bytes_);
              } else {
                std::cerr << "Failed to map staging buffer: " << static_cast<int>(status) << std::endl;
              }
            }),
        UINT64_MAX);
    staging_buffer.Unmap();
    staging_buffer.Destroy();
  }

  void CopyCpuToDevice() override {
    if (!webgpu_queue_) {
      throw std::runtime_error("WebGPU queue not initialized");
    }
    WGPUBuffer dst_buf = reinterpret_cast<WGPUBuffer>(p_device_);
    webgpu_queue_->WriteBuffer(dst_buf, 0, p_cpu_, size_in_bytes_);
  }

  void CopyFrom(size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) override {
    if (!webgpu_device_) {
      throw std::runtime_error("WebGPU device not initialized");
    }
    if (source.GetType() == device_label) {
      // WebGPU buffer-to-buffer copy
      WGPUBuffer src_buf = reinterpret_cast<WGPUBuffer>(source.p_device_);
      WGPUBuffer dst_buf = reinterpret_cast<WGPUBuffer>(p_device_);

      auto command_encoder = webgpu_device_->CreateCommandEncoder();
      command_encoder.CopyBufferToBuffer(src_buf, begin_source, dst_buf, begin_dest, size_in_bytes);
      wgpu::CommandBuffer command_buffer = command_encoder.Finish();
      webgpu_queue_->Submit(1, &command_buffer);
    } else {
      CopyThroughCpu(*this, begin_dest, source, begin_source, size_in_bytes);
    }
  }

  void Zero() override {
    if (!webgpu_queue_) {
      throw std::runtime_error("WebGPU queue not initialized");
    }
    // Clear buffer by writing zeros
    WGPUBuffer dst_buf = reinterpret_cast<WGPUBuffer>(p_device_);
    std::vector<uint8_t> zero_data(size_in_bytes_, 0);
    webgpu_queue_->WriteBuffer(dst_buf, 0, zero_data.data(), size_in_bytes_);
  }

  bool owned_;
};

struct InterfaceImpl : DeviceInterface {
  InterfaceImpl() {
#ifdef USE_WEBGPU
    // USE_WEBGPU=ON: Initialize our own Dawn (legacy mode)
    InitializeDawn();
#else
    // USE_WEBGPU=OFF: Will retrieve Dawn from WebGPU EP later
    webgpu_ep_context_id_ = 0;  // Default WebGPU EP context
    dawn_initialized_from_ep_ = false;
#endif
  }

  DeviceType GetType() const override { return DeviceType::WEBGPU; }

  void InitOrt(const OrtApi& /*api*/, Ort::Allocator& allocator) override {
    assert(!ort_allocator_);
    ort_allocator_ = &allocator;
#ifdef USE_WEBGPU
    // USE_WEBGPU=ON: Use our own Dawn objects
    webgpu_device_ = &device_;
    webgpu_queue_ = &queue_;
    webgpu_instance_ = &instance_;
#else
    // USE_WEBGPU=OFF: Retrieve Dawn from WebGPU EP and initialize static pointers
    if (!dawn_initialized_from_ep_) {
      InitializeDawnFromWebGpuEP();
    }
#endif
  }

#ifdef USE_WEBGPU
  wgpu::Device GetDevice() const { return device_; }
  wgpu::Instance GetInstance() const { return instance_; }
#endif

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
  
  // WebGPU-accelerated operations (available in both USE_WEBGPU=ON and OFF modes)
  bool UpdateAttentionMask(void* next_mask_data, void* mask_data, int batch_beam_size, int new_kv_length, int total_length, int max_length, bool update_only, ONNXTensorElementDataType type) override {
    if (!device_) {
      // Fall back to CPU implementation if WebGPU context is not initialized
      return false;
    }

    // Only support static mask updates (update_only = true)
    if (!update_only) {
      return false;  // Fall back to CPU for dynamic mask handling
    }

    try {
      if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        webgpu_update_mask_kernel_int32_.UpdateMask(
            device_, queue_,
            static_cast<int32_t*>(next_mask_data), static_cast<int32_t*>(mask_data),
            batch_beam_size, new_kv_length, total_length, max_length, update_only);
      } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        webgpu_update_mask_kernel_int64_.UpdateMask(
            device_, queue_,
            static_cast<int64_t*>(next_mask_data), static_cast<int64_t*>(mask_data),
            batch_beam_size, new_kv_length, total_length, max_length, update_only);
      } else {
        return false;  // Unsupported data type
      }
      return true;
    } catch (const std::exception&) {
      return false;  // Fall back to CPU on any error
    }
  }

  bool Cast(void* input_data, void* output_data, ONNXTensorElementDataType input_type, ONNXTensorElementDataType output_type, size_t element_count) override {
    if (!device_) {
      return false;  // Fall back to CPU if WebGPU context is not initialized
    }

    if (input_type == output_type) {
      throw std::runtime_error("Cast - input and output types are the same");
    }

    try {
      // Handle supported type conversions
      if (input_type == Ort::TypeToTensorType<int32_t> && output_type == Ort::TypeToTensorType<int64_t>) {
        return webgpu_cast_kernel_.CastInt32ToInt64(
            input_data,
            output_data,
            element_count);
      } else if (input_type == Ort::TypeToTensorType<Ort::Float16_t> && output_type == Ort::TypeToTensorType<float>) {
        return webgpu_cast_kernel_.CastFloat16ToFloat32(
            input_data,
            output_data,
            element_count);
      } else {
        // Unsupported type conversion, fall back to CPU
        return false;
      }
    } catch (const std::exception&) {
      return false;  // Fall back to CPU on any error
    }
  }

  bool UpdatePositionIds(void* position_ids, int batch_beam_size, int total_length, int new_kv_length, ONNXTensorElementDataType type) override {
    if (!device_) {
      return false;  // Fall back to CPU if WebGPU context is not initialized
    }

    // Only support batch_beam_size == 1 for graph capture (continuous decoding)
    if (batch_beam_size != 1) {
      return false;  // Fall back to CPU for batch_size > 1
    }

    try {
      if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        webgpu_update_position_ids_kernel_int32_.UpdatePositionIds(
            device_, queue_,
            static_cast<int32_t*>(position_ids),
            batch_beam_size, total_length, new_kv_length);
      } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        webgpu_update_position_ids_kernel_int64_.UpdatePositionIds(
            device_, queue_,
            static_cast<int64_t*>(position_ids),
            batch_beam_size, total_length, new_kv_length);
      } else {
        return false;  // Unsupported data type
      }
      return true;
    } catch (const std::exception&) {
      return false;  // Fall back to CPU on any error
    }
  }
  void Synchronize() override {}  // Nothing to do

 private:
#ifdef USE_WEBGPU
  // USE_WEBGPU=ON: Own Dawn objects (legacy mode)
  wgpu::Device device_;
  wgpu::Instance instance_;
  wgpu::Queue queue_;
#else
  // USE_WEBGPU=OFF: Retrieve and wrap WebGPU EP's Dawn objects
  int webgpu_ep_context_id_;
  bool dawn_initialized_from_ep_;
  wgpu::Device device_;        // Wrapped WebGPU EP device
  wgpu::Instance instance_;    // Wrapped WebGPU EP instance
  wgpu::Queue queue_;          // Wrapped WebGPU EP queue

  void InitializeDawnFromWebGpuEP() {
    // Retrieve Dawn objects from WebGPU EP
    const void* dawn_proc_table_ptr = OrtWebGpuGetDawnProcTable(webgpu_ep_context_id_);
    void* instance_ptr = OrtWebGpuGetInstance(webgpu_ep_context_id_);
    void* device_ptr = OrtWebGpuGetDevice(webgpu_ep_context_id_);

    if (!device_ptr || !instance_ptr) {
      throw std::runtime_error(
          "Failed to retrieve Dawn objects from WebGPU EP. "
          "Make sure WebGPU EP is initialized first by calling SetWebGPUProvider.");
    }

    // Initialize Dawn proc table if available
    if (dawn_proc_table_ptr) {
      const DawnProcTable* dawn_procs = reinterpret_cast<const DawnProcTable*>(dawn_proc_table_ptr);
      dawnProcSetProcs(dawn_procs);
      std::cout << "Initialized Dawn proc table from WebGPU EP" << std::endl;
    }

    // Wrap the C handles in C++ objects
    // Note: We use Acquire() to wrap existing handles without taking ownership
    device_ = wgpu::Device::Acquire(static_cast<WGPUDevice>(device_ptr));
    instance_ = wgpu::Instance::Acquire(static_cast<WGPUInstance>(instance_ptr));
    queue_ = device_.GetQueue();

    // Set up the static pointers for WebGPUMemory and other classes
    webgpu_device_ = &device_;
    webgpu_queue_ = &queue_;
    webgpu_instance_ = &instance_;

    // Initialize kernels using the WebGPU EP's Dawn
    webgpu_cast_kernel_.Initialize(device_, queue_);

    dawn_initialized_from_ep_ = true;
    
    std::cout << "Successfully initialized from WebGPU EP context " << webgpu_ep_context_id_ << std::endl;
    std::cout << "  Device: " << device_ptr << std::endl;
    std::cout << "  Instance: " << instance_ptr << std::endl;
    std::cout << "  Graph capture and WebGPU operations are now enabled" << std::endl;
  }
#endif

  // WebGPU kernels (available in both modes)
  WebGPUUpdateMaskKernel<int32_t> webgpu_update_mask_kernel_int32_;
  WebGPUUpdateMaskKernel<int64_t> webgpu_update_mask_kernel_int64_;
  WebGPUUpdatePositionIdsKernel<int32_t> webgpu_update_position_ids_kernel_int32_;
  WebGPUUpdatePositionIdsKernel<int64_t> webgpu_update_position_ids_kernel_int64_;
  CastKernel webgpu_cast_kernel_;

#ifdef USE_WEBGPU
  std::vector<wgpu::FeatureName> GetAvailableRequiredFeatures(const wgpu::Adapter& adapter) const {
    std::vector<wgpu::FeatureName> required_features;
    constexpr wgpu::FeatureName features[]{
        wgpu::FeatureName::ChromiumExperimentalTimestampQueryInsidePasses,
        wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix,
        wgpu::FeatureName::TimestampQuery,
        wgpu::FeatureName::ShaderF16,
        wgpu::FeatureName::Subgroups,
        wgpu::FeatureName::BufferMapExtendedUsages,
    };
    for (auto feature : features) {
      if (adapter.HasFeature(feature)) {
        required_features.push_back(feature);
      }
    }
    return required_features;
  }

  wgpu::Limits GetRequiredLimits(const wgpu::Adapter& adapter) const {
    wgpu::Limits required_limits{};
    wgpu::Limits adapter_limits;
    adapter.GetLimits(&adapter_limits);

    required_limits.maxBindGroups = adapter_limits.maxBindGroups;
    required_limits.maxComputeWorkgroupStorageSize = adapter_limits.maxComputeWorkgroupStorageSize;
    required_limits.maxComputeWorkgroupsPerDimension = adapter_limits.maxComputeWorkgroupsPerDimension;
    required_limits.maxStorageBufferBindingSize = adapter_limits.maxStorageBufferBindingSize;
    required_limits.maxBufferSize = adapter_limits.maxBufferSize;
    required_limits.maxComputeInvocationsPerWorkgroup = adapter_limits.maxComputeInvocationsPerWorkgroup;
    required_limits.maxComputeWorkgroupSizeX = adapter_limits.maxComputeWorkgroupSizeX;
    required_limits.maxComputeWorkgroupSizeY = adapter_limits.maxComputeWorkgroupSizeY;
    required_limits.maxComputeWorkgroupSizeZ = adapter_limits.maxComputeWorkgroupSizeZ;

    return required_limits;
  }

  bool InitializeDawn() {
    // Set up Dawn instance
    dawnProcSetProcs(&dawn::native::GetProcs());

    // Create Dawn instance
    wgpu::InstanceFeatureName required_instance_features[] = {wgpu::InstanceFeatureName::TimedWaitAny};
    wgpu::InstanceDescriptor instance_desc{};
    instance_desc.requiredFeatures = required_instance_features;
    instance_desc.requiredFeatureCount = sizeof(required_instance_features) / sizeof(required_instance_features[0]);
    instance_ = wgpu::CreateInstance(&instance_desc);

    wgpu::RequestAdapterOptions adapter_options = {};
    adapter_options.backendType = wgpu::BackendType::D3D12;
    adapter_options.powerPreference = wgpu::PowerPreference::HighPerformance;

    std::vector<const char*> enableToggleNames;
    //  enableToggleNames.push_back("dump_shaders");
    enableToggleNames.push_back("use_dxc");
    enableToggleNames.push_back("allow_unsafe_apis");
    //   enableToggleNames.push_back("disable_symbol_renaming");
    // enableToggleNames.push_back("disable_workgroup_init");
    // enableToggleNames.push_back("disable_robustness");
    //  enableToggleNames.push_back("emit_hlsl_debug_symbols");
    wgpu::DawnTogglesDescriptor toggles = {};
    toggles.enabledToggles = enableToggleNames.data();
    toggles.enabledToggleCount = enableToggleNames.size();
    adapter_options.nextInChain = &toggles;

    // Synchronously create the adapter
    wgpu::Adapter w_adapter;
    instance_.WaitAny(
        instance_.RequestAdapter(
            &adapter_options, wgpu::CallbackMode::WaitAnyOnly,
            [&](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, const char* message) {
              if (status != wgpu::RequestAdapterStatus::Success) {
                return false;
              }
              w_adapter = std::move(adapter);
              return true;
            }),
        UINT64_MAX);
    if (w_adapter == nullptr) {
      return false;
    }

    wgpu::AdapterInfo info;
    w_adapter.GetInfo(&info);

    // Create device from adapter
    wgpu::DeviceDescriptor device_desc = {};
    device_desc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](const wgpu::Device&, wgpu::DeviceLostReason reason, wgpu::StringView message) {
          const char* reasonName = "";
          switch (reason) {
            case wgpu::DeviceLostReason::Unknown:
              reasonName = "Unknown";
              break;
            case wgpu::DeviceLostReason::Destroyed:
              reasonName = "Destroyed";
              break;
            case wgpu::DeviceLostReason::FailedCreation:
              reasonName = "FailedCreation";
              break;
            default:
              //  DAWN_UNREACHABLE();
              break;
          }
          std::cerr << "Device lost because of " << reasonName << ": " << message.data;
        });
    device_desc.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType type, wgpu::StringView message) {
          const char* errorTypeName = "";
          switch (type) {
            case wgpu::ErrorType::Validation:
              errorTypeName = "Validation";
              break;
            case wgpu::ErrorType::OutOfMemory:
              errorTypeName = "Out of memory";
              break;
            case wgpu::ErrorType::Unknown:
              errorTypeName = "Unknown";
              break;
            default:
              //   DAWN_UNREACHABLE();
              break;
          }
          std::cerr << errorTypeName << " error: " << message.data;
        });

    // Configure device toggles (add this after error callbacks, before setting features)
    std::vector<const char*> enabledDeviceToggles = {
        "skip_validation",  // only use "skip_validation" when ValidationMode is set to "Disabled"
        "disable_robustness",
        "d3d_disable_ieee_strictness"};

    std::vector<const char*> disabledDeviceToggles = {
        "lazy_clear_resource_on_first_use",
        "timestamp_quantization"};

    wgpu::DawnTogglesDescriptor deviceToggles = {};
    deviceToggles.enabledToggles = enabledDeviceToggles.data();
    deviceToggles.enabledToggleCount = enabledDeviceToggles.size();
    deviceToggles.disabledToggles = disabledDeviceToggles.data();
    deviceToggles.disabledToggleCount = disabledDeviceToggles.size();
    device_desc.nextInChain = &deviceToggles;

    std::vector<wgpu::FeatureName> required_features = GetAvailableRequiredFeatures(w_adapter);
    if (required_features.size() > 0) {
      device_desc.requiredFeatures = required_features.data();
      device_desc.requiredFeatureCount = required_features.size();
    }

    wgpu::Limits required_limits = GetRequiredLimits(w_adapter);
    device_desc.requiredLimits = &required_limits;

    // Synchronously create the device
    instance_.WaitAny(
        w_adapter.RequestDevice(
            &device_desc, wgpu::CallbackMode::WaitAnyOnly,
            [&](wgpu::RequestDeviceStatus status, wgpu::Device device, const char* message) {
              if (status != wgpu::RequestDeviceStatus::Success) {
                return false;
              }
              device_ = std::move(device);
              queue_ = device_.GetQueue();
              return true;
            }),
        UINT64_MAX);
    if (device_ == nullptr) {
      return false;
    }

    // Set up device lost callback
    device_.SetLoggingCallback([](wgpu::LoggingType type, struct wgpu::StringView message) {
      std::cerr << message.data;
    });

    // Initialize kernels
    webgpu_cast_kernel_.Initialize(device_, queue_);

    std::cout << "WebGPU device initialized successfully!" << std::endl;
    return true;
  }
#endif
};

}  // namespace WebGPU

std::unique_ptr<WebGPU::InterfaceImpl> g_webgpu_device;

void InitWebGPUInterface() {
  if (!g_webgpu_device)
    g_webgpu_device = std::make_unique<WebGPU::InterfaceImpl>();
}

void CloseWebGPUInterface() {
  g_webgpu_device.reset();
}

void SetWebGPUProvider(OrtSessionOptions& session_options, const std::unordered_map<std::string, std::string>& provider_options) {
  std::vector<const char*> keys, values;
  std::vector<std::string> key_strings, value_strings;

  // Add all existing provider options
  for (const auto& option : provider_options) {
    key_strings.push_back(option.first);
    value_strings.push_back(option.second);
  }

#ifdef USE_WEBGPU
  // USE_WEBGPU=ON: Pass our own Dawn objects to WebGPU EP
  auto* webgpu_interface = static_cast<WebGPU::InterfaceImpl*>(g_webgpu_device.get());
  if (!webgpu_interface) {
    throw std::runtime_error("WebGPU interface not initialized");
  }

  key_strings.push_back("dawnProcTable");
  value_strings.push_back(std::to_string(reinterpret_cast<size_t>(&dawn::native::GetProcs())));

  key_strings.push_back("webgpuInstance");
  value_strings.push_back(std::to_string(reinterpret_cast<size_t>(webgpu_interface->GetInstance().Get())));

  key_strings.push_back("webgpuDevice");
  value_strings.push_back(std::to_string(reinterpret_cast<size_t>(webgpu_interface->GetDevice().Get())));

  key_strings.push_back("deviceId");
  value_strings.push_back("1");
#else
  // USE_WEBGPU=OFF: Let WebGPU EP initialize its own Dawn
  // We'll retrieve it later in InitOrt() via the exposed APIs
  std::cout << "Using WebGPU EP's internal Dawn (shared mode)" << std::endl;
#endif

  // Convert to C-style arrays
  for (const auto& key : key_strings) {
    keys.push_back(key.c_str());
  }
  for (const auto& value : value_strings) {
    values.push_back(value.c_str());
  }

  // Append the WebGPU provider
  Ort::ThrowOnError(Ort::api->SessionOptionsAppendExecutionProvider(
      &session_options, "WebGPU", keys.data(), values.data(), keys.size()));
}

DeviceInterface* GetWebGPUInterface() {
  if (!g_webgpu_device) {
    InitWebGPUInterface();
  }
  return g_webgpu_device.get();
}

}  // namespace Generators
