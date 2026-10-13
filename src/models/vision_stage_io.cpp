// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vision_stage_io.h"

namespace Generators {

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
