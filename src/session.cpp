// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "session.h"

#pragma once

namespace Generators {

Ort::Allocator& Session::CreateAllocator(DeviceType type) {
  // should not be called for CPU
  assert(type != DeviceType::CPU);

  size_t idx = static_cast<int>(type);
  if (!device_allocator_[idx]) {
    auto name = Generators::GetDeviceName(type);
    auto memory_info = OrtMemoryInfo::Create(name.c_str(), OrtAllocatorType::OrtDeviceAllocator, 0,
                                             OrtMemType::OrtMemTypeDefault);
    device_allocator_[idx] = std::move(Ort::Allocator::Create(*ort_session_, *memory_info));
    if (!device_allocator_[idx])
      throw std::runtime_error("Unexpected failure creating device memory allocator for " + std::string(name));
  }

  return *device_allocator_[idx];
}

}  // namespace Generators
