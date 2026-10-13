// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vision_stage_io.h"

#include <cstring>

namespace Generators {

void ApplyVisionRunOption(OrtRunOptions& options, const char* key, const char* value) {
  if (strcmp(key, "terminate_session") == 0) {
    if (strcmp(value, "0") == 0) {
      options.UnsetTerminate();
    } else if (strcmp(value, "1") == 0) {
      options.SetTerminate();
    } else {
      throw std::runtime_error(std::string("terminate_session key value unexpected: ") + value);
    }
  } else if (strcmp(key, "enable_profiling") == 0) {
#if ORT_API_VERSION >= 25
    if (strcmp(value, "0") == 0) {
      options.DisableProfiling();
    } else {
      constexpr const char* default_profile_prefix = "onnxruntime_run_profile";
      const char* prefix = strcmp(value, "1") == 0 ? default_profile_prefix : value;
      options.EnableProfiling(fs::path(prefix).c_str());
    }
#else
    throw std::runtime_error("enable_profiling requires ONNX Runtime 1.25 or later");
#endif
  } else {
    options.AddConfigEntry(key, value);
  }
}

std::unique_ptr<OrtRunOptions> CreateVisionStageRunOptions(const Config::Model::Vision& vision, size_t stage_index) {
  auto options = OrtRunOptions::Create();
  const auto& configured = vision.pipeline[stage_index].run_options.has_value()
                               ? vision.pipeline[stage_index].run_options
                               : vision.run_options;
  if (configured) {
    for (const auto& [key, value] : *configured) {
      ApplyVisionRunOption(*options, key.c_str(), value.c_str());
    }
  }
  return options;
}

std::unique_ptr<OrtValue> StageVisionEncoderOutput(OrtValue& encoded, DeviceInterface& encoder_device,
                                                   DeviceInterface& projector_device) {
  if (encoded.GetTensorMemoryInfo().GetDeviceType() != OrtMemoryInfoDeviceType_CPU &&
      encoder_device.GetType() == DeviceType::CPU) {
    throw std::runtime_error("Vision encoder returned device memory without a matching device interface");
  }
  auto& source_device = DeviceForTensor(encoded, encoder_device);
  if (source_device.GetType() != DeviceType::CPU) {
    const auto& memory_info = encoded.GetTensorMemoryInfo();
    auto expected_info = source_device.GetMemoryInfo();
    if (memory_info.GetAllocatorName() != expected_info->GetAllocatorName() ||
        memory_info.GetDeviceId() != expected_info->GetDeviceId()) {
      throw std::runtime_error("Vision encoder output memory does not match the configured encoder device");
    }
  }
  if (SessionCanAccess(projector_device, source_device)) return nullptr;

  auto info = encoded.GetTensorTypeAndShapeInfo();
  auto& cpu = *GetDeviceInterface(DeviceType::CPU);
  auto staged = OrtValue::CreateTensor(cpu.GetAllocator(), info->GetShape(), info->GetElementType());
  if (info->GetElementCount() != 0) {
    ByteWrapTensor(cpu, *staged).CopyFrom(ByteWrapTensor(source_device, encoded));
  }
  return staged;
}

}  // namespace Generators
