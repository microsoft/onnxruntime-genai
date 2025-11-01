// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "oga_object.h"

namespace OgaPy {

struct OgaTensor : OgaObject {
  explicit OgaTensor(::OgaTensor* p) : ptr_(p) {}
  ~OgaTensor() override { if (ptr_) OgaDestroyTensor(ptr_); }
  ::OgaTensor* get() const { return ptr_; }
private:
  ::OgaTensor* ptr_;
};

struct OgaNamedTensors : OgaObject {
  explicit OgaNamedTensors(::OgaNamedTensors* p) : ptr_(p) {}
  ~OgaNamedTensors() override { if (ptr_) OgaDestroyNamedTensors(ptr_); }
  ::OgaNamedTensors* get() const { return ptr_; }
private:
  ::OgaNamedTensors* ptr_;
};

} // namespace OgaPy
