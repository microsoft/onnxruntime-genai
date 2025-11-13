// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "oga_object.h"
#include "../../ort_genai_c.h"

namespace OgaPy {

// This is used to turn OgaResult return values from the C API into std::runtime_error exceptions
inline void OgaCheckResult(::OgaResult* result) {
  if (result) {
    std::string error_message = OgaResultGetError(result);
    OgaDestroyResult(result);                            
    throw std::runtime_error(error_message);
  }
}

// Simple RAII wrappers for temporary objects (not exposed to Python)
struct OgaResult {
  explicit OgaResult(::OgaResult* p) : ptr_(p) {}
  ~OgaResult() { if (ptr_) OgaDestroyResult(ptr_); }
  ::OgaResult* get() const { return ptr_; }

 private:
  ::OgaResult* ptr_;
};

struct OgaStringArray : OgaObject {
  explicit OgaStringArray(::OgaStringArray* p) : ptr_(p) {}
  ~OgaStringArray() override { if (ptr_) OgaDestroyStringArray(ptr_); }
  ::OgaStringArray* get() const { return ptr_; }

  // Add a string to the array
  void AddString(const char* str) {
    OgaCheckResult(OgaStringArrayAddString(ptr_, str));
  }
  
  // Get the number of strings in the array
  size_t GetCount() const {
    size_t out = 0;
    OgaCheckResult(OgaStringArrayGetCount(ptr_, &out));
    return out;
  }
  
  // Get a string at a specific index
  const char* GetString(size_t index) const {
    const char* out = nullptr;
    OgaCheckResult(OgaStringArrayGetString(ptr_, index, &out));
    return out;
  }

 private:
  ::OgaStringArray* ptr_;
};

struct OgaSequences : OgaObject {
  explicit OgaSequences(::OgaSequences* p) : ptr_(p) {}
  ~OgaSequences() override { if (ptr_) OgaDestroySequences(ptr_); }
  ::OgaSequences* get() const { return ptr_; }

  // Get the number of sequences
  size_t Count() const {
    return OgaSequencesCount(ptr_);
  }
  
  // Append a token sequence
  void AppendTokenSequence(const int32_t* token_ptr, size_t token_cnt) {
    OgaCheckResult(OgaAppendTokenSequence(token_ptr, token_cnt, ptr_));
  }
  
  // Append a token to a specific sequence
  void AppendTokenToSequence(int32_t token, size_t sequence_index) {
    OgaCheckResult(OgaAppendTokenToSequence(token, ptr_, sequence_index));
  }
  
  // Get the number of tokens in a specific sequence
  size_t GetSequenceCount(size_t sequence_index) const {
    return OgaSequencesGetSequenceCount(ptr_, sequence_index);
  }

  // Get sequence data as a borrowed view (automatically handles reference counting)
  SequenceDataView* GetSequenceData(size_t index) const {
    const int32_t* data = OgaSequencesGetSequenceData(ptr_, index);
    size_t count = OgaSequencesGetSequenceCount(ptr_, index);
    return new SequenceDataView(const_cast<OgaSequences*>(this), data, count);
  }

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

}  // namespace OgaPy
