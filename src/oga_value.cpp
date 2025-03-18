// // Copyright (c) Microsoft Corporation. All rights reserved.
// // Licensed under the MIT License.
// #include <stdexcept>
// #include "generators.h"
// #include "oga_value.h"
// #include "models/utils.h"

// namespace Generators {

// OgaValue::OgaValue(DeviceInterface* device, ONNXTensorElementDataType type) : type_{type}, p_device_{device} {}

// OgaValue::~OgaValue() {
//   if (buffer_ != nullptr) {
//     p_device_->GetAllocator().Free(buffer_);
//   }
// }

// void OgaValue::CreateTensor(std::span<const int64_t> shape, bool make_static) {
//   if (make_static) {
//     size_t new_bytes = SizeOf(type_) * ElementCountFromShape(shape);
//     if (buffer_ == nullptr) {
//       bytes_ = new_bytes;
//       buffer_ = p_device_->GetAllocator().Alloc(bytes_);
//     } else if (new_bytes > bytes_) {
//       throw std::runtime_error("OgaValue: Static buffer new_bytes > bytes_");
//     }
//     ort_value_ = OrtValue::CreateTensor(p_device_->GetAllocator().GetInfo(), buffer_, new_bytes, shape, type_);
//     is_static_ = true;
//   } else {
//     ort_value_ = OrtValue::CreateTensor(p_device_->GetAllocator(), shape, type_);
//     is_static_ = false;
//   }
// }

// void OgaValue::MakeStatic() {
//   if (ort_value_ == nullptr) {
//     throw std::runtime_error("OgaValue: MakeStatic called before CreateTensor");
//   }
//   size_t new_bytes = GetElementCount() * SizeOf(type_);
//   if (buffer_ == nullptr) {
//     buffer_ = p_device_->GetAllocator().Alloc(new_bytes);
//     bytes_ = new_bytes;
//   } else if (new_bytes > bytes_) {
//     throw std::runtime_error("OgaValue: Static buffer new_bytes > bytes_");
//   }
//   // Copy the data to the static buffer
//   auto new_static_tensor = OrtValue::CreateTensor(p_device_->GetAllocator().GetInfo(), buffer_, new_bytes, GetShape(), type_);
//   auto new_static_span = ByteWrapTensor(*p_device_, *new_static_tensor);
//   auto old_static_span = GetByteSpan();
//   new_static_span.CopyFrom(old_static_span);
//   ort_value_ = std::move(new_static_tensor);
//   is_static_ = true;
// }

// OrtValue* OgaValue::GetOrtValue() {
//   if (ort_value_ == nullptr) {
//     return nullptr;
//   }
//   return ort_value_.get();
// }

// DeviceSpan<uint8_t> OgaValue::GetByteSpan() {
//   if (ort_value_ == nullptr) {
//     throw std::runtime_error("OgaValue: GetByteSpan called before CreateTensor");
//   }
//   return ByteWrapTensor(*p_device_, *ort_value_);
// }

// void* OgaValue::GetMutableRawData() {
//   if (ort_value_ == nullptr) {
//     throw std::runtime_error("OgaValue: GetMutableRawData called before CreateTensor");
//   }
//   return ort_value_->GetTensorMutableRawData();
// }

// const void* OgaValue::GetRawData() const {
//   if (ort_value_ == nullptr) {
//     throw std::runtime_error("OgaValue: GetRawData called before CreateTensor");
//   }
//   return ort_value_->GetTensorRawData();
// }

// std::vector<int64_t> OgaValue::GetShape() const {
//   if (ort_value_ == nullptr) {
//     throw std::runtime_error("OgaValue: GetShape called before CreateTensor");
//   }
//   return ort_value_->GetTensorTypeAndShapeInfo()->GetShape();
// }

// ONNXTensorElementDataType OgaValue::GetType() const {
//   return type_;
// }

// size_t OgaValue::GetElementCount() const {
//   if (ort_value_ == nullptr) {
//     return 0;
//   }
//   return ort_value_->GetTensorTypeAndShapeInfo()->GetElementCount();
// }

// }  // namespace Generators