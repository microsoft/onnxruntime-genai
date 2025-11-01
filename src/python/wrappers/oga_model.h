// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "oga_object.h"

namespace OgaPy {

struct OgaModel : OgaObject {
  explicit OgaModel(::OgaModel* p) : ptr_(p) {}
  ~OgaModel() override { if (ptr_) OgaDestroyModel(ptr_); }
  ::OgaModel* get() const { return ptr_; }
private:
  ::OgaModel* ptr_;
};

} // namespace OgaPy
