// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "oga_object.h"

namespace OgaPy {

struct OgaRequest : OgaObject {
  explicit OgaRequest(::OgaRequest* p) : ptr_(p) {}
  ~OgaRequest() override { if (ptr_) OgaDestroyRequest(ptr_); }
  ::OgaRequest* get() const { return ptr_; }
private:
  ::OgaRequest* ptr_;
};

struct OgaEngine : OgaObject {
  explicit OgaEngine(::OgaEngine* p) : ptr_(p) {}
  ~OgaEngine() override { if (ptr_) OgaDestroyEngine(ptr_); }
  ::OgaEngine* get() const { return ptr_; }
private:
  ::OgaEngine* ptr_;
};

} // namespace OgaPy
