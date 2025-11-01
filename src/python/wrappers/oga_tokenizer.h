// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "oga_object.h"

namespace OgaPy {

struct OgaTokenizer : OgaObject {
  explicit OgaTokenizer(::OgaTokenizer* p) : ptr_(p) {}
  ~OgaTokenizer() override { if (ptr_) OgaDestroyTokenizer(ptr_); }
  ::OgaTokenizer* get() const { return ptr_; }
private:
  ::OgaTokenizer* ptr_;
};

struct OgaTokenizerStream : OgaObject {
  explicit OgaTokenizerStream(::OgaTokenizerStream* p) : ptr_(p) {}
  ~OgaTokenizerStream() override { if (ptr_) OgaDestroyTokenizerStream(ptr_); }
  ::OgaTokenizerStream* get() const { return ptr_; }
private:
  ::OgaTokenizerStream* ptr_;
};

} // namespace OgaPy
