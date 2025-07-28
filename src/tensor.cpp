// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <stdexcept>
#include "generators.h"
#include "models/utils.h"

namespace Generators {

Tensor::Tensor(DeviceInterface* device, ONNXTensorElementDataType type) : p_device_{device}, type_{type} {}

Tensor::Tensor(std::unique_ptr<OrtValue> ort_tensor) : ort_tensor_{std::move(ort_tensor)} {
  p_device_ = GetDeviceInterface(DeviceType::CPU);
  type_ = ort_tensor_->GetTensorTypeAndShapeInfo()->GetElementType();
}

Tensor::~Tensor() {
  if (buffer_ != nullptr) {
    p_device_->GetAllocator().Free(buffer_);
    buffer_ = nullptr;
  }
}

void Tensor::CreateTensor(std::span<const int64_t> shape, bool make_static) {
  if (make_static) {
    size_t new_bytes = Ort::SizeOf(type_) * ElementCountFromShape(shape);
    if (buffer_ == nullptr) {
      bytes_ = new_bytes;
      buffer_ = p_device_->GetAllocator().Alloc(bytes_);
    } else if (new_bytes > bytes_) {
      throw std::runtime_error("Tensor: Static buffer new_bytes > bytes_");
    }
    ort_tensor_ = OrtValue::CreateTensor(p_device_->GetAllocator().GetInfo(), buffer_, new_bytes, shape, type_);
    is_static_ = true;
  } else {
    ort_tensor_ = OrtValue::CreateTensor(p_device_->GetAllocator(), shape, type_);
    is_static_ = false;
  }
}

void Tensor::MakeStatic() {
  if (ort_tensor_ == nullptr) {
    throw std::runtime_error("Tensor: MakeStatic called before CreateTensor");
  }
  size_t new_bytes = GetElementCount() * Ort::SizeOf(type_);
  if (buffer_ == nullptr) {
    buffer_ = p_device_->GetAllocator().Alloc(new_bytes);
    bytes_ = new_bytes;
  } else if (new_bytes > bytes_) {
    throw std::runtime_error("Tensor: Static buffer new_bytes > bytes_");
  }
  // Copy the data to the static buffer
  auto new_static_tensor = OrtValue::CreateTensor(p_device_->GetAllocator().GetInfo(), buffer_, new_bytes, GetShape(), type_);
  auto new_static_span = ByteWrapTensor(*p_device_, *new_static_tensor);
  auto old_static_span = GetByteSpan();
  new_static_span.CopyFrom(old_static_span);
  ort_tensor_ = std::move(new_static_tensor);
  is_static_ = true;
}

OrtValue* Tensor::GetOrtTensor() {
  if (ort_tensor_ == nullptr) {
    return nullptr;
  }
  return ort_tensor_.get();
}

DeviceSpan<uint8_t> Tensor::GetByteSpan() {
  if (ort_tensor_ == nullptr) {
    throw std::runtime_error("Tensor: GetByteSpan called before CreateTensor");
  }
  return ByteWrapTensor(*p_device_, *ort_tensor_);
}

void* Tensor::GetMutableRawData() {
  if (ort_tensor_ == nullptr) {
    throw std::runtime_error("Tensor: GetMutableRawData called before CreateTensor");
  }
  return ort_tensor_->GetTensorMutableRawData();
}

const void* Tensor::GetRawData() const {
  if (ort_tensor_ == nullptr) {
    throw std::runtime_error("Tensor: GetRawData called before CreateTensor");
  }
  return ort_tensor_->GetTensorRawData();
}

std::vector<int64_t> Tensor::GetShape() const {
  if (ort_tensor_ == nullptr) {
    throw std::runtime_error("Tensor: GetShape called before CreateTensor");
  }
  return ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape();
}

ONNXTensorElementDataType Tensor::GetType() const {
  return type_;
}

size_t Tensor::GetElementCount() const {
  if (ort_tensor_ == nullptr) {
    return 0;
  }
  return ort_tensor_->GetTensorTypeAndShapeInfo()->GetElementCount();
}

}  // namespace Generators