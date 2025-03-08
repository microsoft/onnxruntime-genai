// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "models/onnxruntime_api.h"
#include "smartptrs.h"

namespace Generators {

struct OgaValue {
  OgaValue(DeviceInterface* device, ONNXTensorElementDataType type);
  ~OgaValue();

  // A static tensor is allocated once on a buffer which is reused
  // A non-static tensor is allocated as a new OrtValue every time CreateTensor is called
  void CreateTensor(std::span<const int64_t> shape, bool make_static = false);

  void MakeStatic(); // Make the tensor static, if it is not already

  OrtValue* GetOrtValue();

  template <typename T>
  T* GetMutableData() {
    if (ort_value_ == nullptr)
      throw std::runtime_error("OgaValue: GetMutableData called before CreateTensor");
    return ort_value_->GetTensorMutableData<T>();
  }

  template <typename T>
  const T* GetData() const {
    if (ort_value_ == nullptr)
      throw std::runtime_error("OgaValue: GetData called before CreateTensor");
    return ort_value_->GetTensorData<T>();
  }

  void* GetMutableRawData();
  const void* GetRawData() const;

  std::vector<int64_t> GetShape() const;

  ONNXTensorElementDataType GetType() const;
  
  size_t GetElementCount() const;
  
  ONNXTensorElementDataType type_;
  bool is_static_{};
  mutable DeviceInterface* p_device_{};
  std::unique_ptr<OrtValue> ort_value_;
  // For static tensors, allocated once
  void* buffer_{};
  size_t bytes_{};
};

}  // namespace Generators