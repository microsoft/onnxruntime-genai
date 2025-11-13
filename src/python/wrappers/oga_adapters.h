// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "oga_object.h"

namespace OgaPy {

struct OgaAdapters : OgaObject {
  explicit OgaAdapters(::OgaAdapters* p) : ptr_(p) {}
  ~OgaAdapters() override { if (ptr_) OgaDestroyAdapters(ptr_); }
  ::OgaAdapters* get() const { return ptr_; }
  
  // Load an adapter
  void LoadAdapter(const char* adapter_file_path, const char* adapter_name) {
    OgaCheckResult(OgaLoadAdapter(ptr_, adapter_file_path, adapter_name));
  }
  
  // Unload an adapter
  void UnloadAdapter(const char* adapter_name) {
    OgaCheckResult(OgaUnloadAdapter(ptr_, adapter_name));
  }
  
private:
  ::OgaAdapters* ptr_;
};

// Forward declaration from oga_generator.h
struct OgaGenerator;

// Inline implementation of OgaGenerator::SetActiveAdapter (defined here after OgaAdapters is complete)
inline void OgaGenerator::SetActiveAdapter(OgaAdapters* adapters, const char* adapter_name) {
  OgaCheckResult(OgaSetActiveAdapter(get(), adapters->get(), adapter_name));
}

} // namespace OgaPy
