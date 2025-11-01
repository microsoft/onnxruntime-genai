// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "oga_object.h"

namespace OgaPy {

struct OgaImages : OgaObject {
  explicit OgaImages(::OgaImages* p) : ptr_(p) {}
  ~OgaImages() override { if (ptr_) OgaDestroyImages(ptr_); }
  ::OgaImages* get() const { return ptr_; }
private:
  ::OgaImages* ptr_;
};

struct OgaAudios : OgaObject {
  explicit OgaAudios(::OgaAudios* p) : ptr_(p) {}
  ~OgaAudios() override { if (ptr_) OgaDestroyAudios(ptr_); }
  ::OgaAudios* get() const { return ptr_; }
private:
  ::OgaAudios* ptr_;
};

struct OgaMultiModalProcessor : OgaObject {
  explicit OgaMultiModalProcessor(::OgaMultiModalProcessor* p) : ptr_(p) {}
  ~OgaMultiModalProcessor() override { if (ptr_) OgaDestroyMultiModalProcessor(ptr_); }
  ::OgaMultiModalProcessor* get() const { return ptr_; }
private:
  ::OgaMultiModalProcessor* ptr_;
};

} // namespace OgaPy
