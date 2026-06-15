// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "smartptrs.h"
#include "config.h"      // for Config and Config::ProviderOptions
#include "generators.h"  // for GetOrtGlobals

namespace Generators {

DeviceInterface::~DeviceInterface() {};

GlobalAllocatorSession& DeviceInterface::GetGlobalAllocatorSession(const Config& /* config */) const {
  return GetOrtGlobals()->device_allocators_[static_cast<int>(GetType())];
}

std::unique_ptr<OrtMemoryInfo> DeviceInterface::GetMemoryInfo() const {
  // Names for the device memory types used by 'OrtMemoryInfo::Create'
  static const char* device_memory_type_names[] = {"CPU (Not used, see above)", "Cuda", "DML", "WebGPU_Buf", "QnnHtpShared", "OpenVINO (Not used, see above)", "Cuda", "Cpu"};
  static_assert(std::size(device_memory_type_names) == static_cast<size_t>(DeviceType::MAX));

  // Get the allocator from the OrtSession for the DeviceType (it's called 'AllocatorCreate' but it's really 'AllocatorGet')
  auto name = device_memory_type_names[static_cast<int>(GetType())];
  return OrtMemoryInfo::Create(name, OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
}

Config::ProviderOptions DeviceInterface::GetProviderOptionsForAllocatorSession(const Config& /* config */) const {
  // Names for the device types used by 'SetProviderSessionOptions'
  static const char* device_type_names[] = {"CPU (Not used, see above)", "cuda", "DML", "WebGPU", "QNN", "OpenVINO (Not used, see above)", "NvTensorRtRtx", "RyzenAI"};
  static_assert(std::size(device_type_names) == static_cast<size_t>(DeviceType::MAX));

  return Config::ProviderOptions{device_type_names[static_cast<int>(GetType())], {}};
}

void* DeviceInterface::GetCudaStream() {
  assert(false);
  return nullptr;
}  // Temporary until we fully factor out providers

}  // namespace Generators
