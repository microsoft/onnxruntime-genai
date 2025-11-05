// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "py_utils.h"
#include "py_numpy.h"
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace Generators {

// Simple wrapper class for Tensor - stores numpy array
struct PyTensor {
  nb::ndarray<> array;  // Store the original numpy array
  std::vector<int64_t> shape;
  std::string dtype;
  
  PyTensor(nb::ndarray<> arr) : array(arr) {
    // Get shape from ndarray
    for (size_t i = 0; i < arr.ndim(); ++i) {
      shape.push_back(arr.shape(i));
    }
    
    // Get dtype
    auto dtype_info = arr.dtype();
    if (dtype_info.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Float) && dtype_info.bits == 32) {
      dtype = "float32";
    } else if (dtype_info.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Int) && dtype_info.bits == 32) {
      dtype = "int32";
    } else if (dtype_info.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Int) && dtype_info.bits == 64) {
      dtype = "int64";
    } else {
      dtype = "unknown";
    }
  }
  
  // Copy constructor
  PyTensor(const PyTensor& other) : array(other.array), shape(other.shape), dtype(other.dtype) {}
  
  nb::ndarray<> AsNumpy() const {
    return array;
  }
};

void BindTensor(nb::module_& m) {
  nb::class_<PyTensor>(m, "Tensor")
    .def("__init__", [](PyTensor* t, nb::ndarray<> array) {
      new (t) PyTensor(array);
    })
    .def("shape", [](PyTensor& self) -> std::vector<int64_t> {
      return self.shape;
    })
    .def("type", [](PyTensor& self) -> std::string {
      return self.dtype;
    })
    .def("as_numpy", &PyTensor::AsNumpy);
}

} // namespace Generators
