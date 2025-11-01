// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "oga_object.h"

namespace OgaPy {

struct OgaAdapters : OgaObject {
  explicit OgaAdapters(::OgaAdapters* p) : ptr_(p) {}
  ~OgaAdapters() override { if (ptr_) OgaDestroyAdapters(ptr_); }
  ::OgaAdapters* get() const { return ptr_; }
private:
  ::OgaAdapters* ptr_;
};

} // namespace OgaPy
