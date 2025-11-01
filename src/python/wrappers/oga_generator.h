// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "oga_object.h"

namespace OgaPy {

struct OgaGeneratorParams : OgaObject {
  explicit OgaGeneratorParams(::OgaGeneratorParams* p) : ptr_(p) {}
  ~OgaGeneratorParams() override { if (ptr_) OgaDestroyGeneratorParams(ptr_); }
  ::OgaGeneratorParams* get() const { return ptr_; }
private:
  ::OgaGeneratorParams* ptr_;
};

struct OgaGenerator : OgaObject {
  explicit OgaGenerator(::OgaGenerator* p) : ptr_(p) {}
  ~OgaGenerator() override { if (ptr_) OgaDestroyGenerator(ptr_); }
  ::OgaGenerator* get() const { return ptr_; }
private:
  ::OgaGenerator* ptr_;
};

} // namespace OgaPy
