// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "oga_object.h"

namespace OgaPy {

struct OgaModel : OgaObject {
  explicit OgaModel(::OgaModel* p) : ptr_(p) {}
  ~OgaModel() override { if (ptr_) OgaDestroyModel(ptr_); }
  ::OgaModel* get() const { return ptr_; }
  
  // Get the model type
  const char* GetType() const {
    const char* out = nullptr;
    OgaCheckResult(OgaModelGetType(ptr_, &out));
    return out;
  }
  
  // Get the device type
  const char* GetDeviceType() const {
    const char* out = nullptr;
    OgaCheckResult(OgaModelGetDeviceType(ptr_, &out));
    return out;
  }
  
private:
  ::OgaModel* ptr_;
};

} // namespace OgaPy
