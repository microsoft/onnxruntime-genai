// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "session.h"
#include "device.h"

namespace Generators {

namespace {
// Since Python/Others can and will hold onto a generator object past the model object's lifetime we need to ensure
// the allocator used is not destroyed until last. This keeps the allocator around until exit, after all other memory
// has been destroyed. Without this, we will crash in the Onnxruntime BFCArena code when deleting tensors due to the
// arena already being destroyed.
void EnsureDeviceOrtInit(Session& session, DeviceType type) {
  // CPU uses a default allocator that is not per-session
  if (type == DeviceType::CPU)
    return;

  Ort::Allocator& allocator = session.CreateAllocator(type);

  // Necessary for any shared library providers so they can access Ort::api
  GetDeviceInterface(type)->InitOrt(*Ort::api, allocator);
}
}  // namespace

Ort::Allocator& Session::CreateAllocator(DeviceType type) {
  // should not be called for CPU as we know from the ORT implementation details that it's a static instance
  // across all sessions
  assert(type != DeviceType::CPU);

  size_t idx = static_cast<int>(type);
  if (!device_allocator_[idx]) {
    auto name = DeviceTypeToName(type);
    auto memory_info = OrtMemoryInfo::Create(name, OrtAllocatorType::OrtDeviceAllocator, 0,
                                             OrtMemType::OrtMemTypeDefault);
    device_allocator_[idx] = std::move(Ort::Allocator::Create(*ort_session_, *memory_info));
    if (!device_allocator_[idx])
      throw std::runtime_error("Unexpected failure creating device memory allocator for " + std::string(name));
  }

  return *device_allocator_[idx];
}

void Session::InitDeviceAllocator(Session& session) {
  EnsureDeviceOrtInit(session, p_device_->GetType());

  // Only CUDA and DML does every input on the device
  if (p_device_->GetType() == DeviceType::CUDA || p_device_->GetType() == DeviceType::DML)
    p_device_inputs_ = p_device_;
  else
    p_device_inputs_ = GetDeviceInterface(DeviceType::CPU);

  // The kvcache is always allocated in device memory
  p_device_kvcache_ = p_device_;

  session_info_ = std::make_unique<SessionInfo>(session);
}

}  // namespace Generators
