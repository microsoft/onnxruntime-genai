// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "oga_object.h"
#include "oga_utils.h"

namespace OgaPy {

// Forward declaration
struct OgaStringArray;

struct OgaTensor : OgaObject {
  explicit OgaTensor(::OgaTensor* p) : ptr_(p) {}
  ~OgaTensor() override { if (ptr_) OgaDestroyTensor(ptr_); }
  ::OgaTensor* get() const { return ptr_; }
  
  // Get the element type of the tensor
  OgaElementType GetType() const {
    OgaElementType out;
    OgaCheckResult(OgaTensorGetType(ptr_, &out));
    return out;
  }
  
  // Get the number of dimensions (rank) of the tensor
  size_t GetShapeRank() const {
    size_t out;
    OgaCheckResult(OgaTensorGetShapeRank(ptr_, &out));
    return out;
  }
  
  // Get the shape of the tensor
  void GetShape(int64_t* shape_dims, size_t shape_dims_count) const {
    OgaCheckResult(OgaTensorGetShape(ptr_, shape_dims, shape_dims_count));
  }
  
  // Get a pointer to the tensor data
  void* GetData() const {
    void* out = nullptr;
    OgaCheckResult(OgaTensorGetData(ptr_, &out));
    return out;
  }
  
private:
  ::OgaTensor* ptr_;
};

struct OgaNamedTensors : OgaObject {
  explicit OgaNamedTensors(::OgaNamedTensors* p) : ptr_(p) {}
  ~OgaNamedTensors() override { if (ptr_) OgaDestroyNamedTensors(ptr_); }
  ::OgaNamedTensors* get() const { return ptr_; }
  
  // Get a tensor by name
  OgaTensor* Get(const char* name) const {
    ::OgaTensor* out = nullptr;
    OgaCheckResult(OgaNamedTensorsGet(ptr_, name, &out));
    return new OgaTensor(out);
  }
  
  // Get all tensor names
  OgaStringArray* GetNames() const {
    ::OgaStringArray* out = nullptr;
    OgaCheckResult(OgaNamedTensorsGetNames(ptr_, &out));
    return new OgaStringArray(out);
  }
  
private:
  ::OgaNamedTensors* ptr_;
};

} // namespace OgaPy
