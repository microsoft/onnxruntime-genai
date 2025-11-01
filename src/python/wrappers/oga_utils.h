// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "../../ort_genai_c.h"

namespace OgaPy {

// Simple RAII wrappers for temporary objects (not exposed to Python)
struct OgaResult {
  explicit OgaResult(::OgaResult* p) : ptr_(p) {}
  ~OgaResult() { if (ptr_) OgaDestroyResult(ptr_); }
  ::OgaResult* get() const { return ptr_; }
private:
  ::OgaResult* ptr_;
};

struct OgaStringArray {
  explicit OgaStringArray(::OgaStringArray* p) : ptr_(p) {}
  ~OgaStringArray() { if (ptr_) OgaDestroyStringArray(ptr_); }
  ::OgaStringArray* get() const { return ptr_; }
private:
  ::OgaStringArray* ptr_;
};

struct OgaSequences {
  explicit OgaSequences(::OgaSequences* p) : ptr_(p) {}
  ~OgaSequences() { if (ptr_) OgaDestroySequences(ptr_); }
  ::OgaSequences* get() const { return ptr_; }
private:
  ::OgaSequences* ptr_;
};

// RAII wrapper for strings created by the C API
struct OgaString {
  OgaString(const char* p) : p_{p} {}
  ~OgaString() { OgaDestroyString(p_); }
  operator const char*() const { return p_; }
  const char* p_;
};

// This is used to turn OgaResult return values from the C API into std::runtime_error exceptions
inline void OgaCheckResult(::OgaResult* result) {
  if (result) {
    OgaResult p_result{result};  // Take ownership so it's destroyed properly
    const char* error = OgaResultGetError(p_result.get());
    throw std::runtime_error(error);
  }
}

} // namespace OgaPy
