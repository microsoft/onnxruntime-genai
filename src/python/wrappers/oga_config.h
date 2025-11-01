// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "oga_object.h"

namespace OgaPy {

struct OgaConfig : OgaObject {
  explicit OgaConfig(::OgaConfig* p) : ptr_(p) {}
  ~OgaConfig() override { if (ptr_) OgaDestroyConfig(ptr_); }
  ::OgaConfig* get() const { return ptr_; }
private:
  ::OgaConfig* ptr_;
};

} // namespace OgaPy
