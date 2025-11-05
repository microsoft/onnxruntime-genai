// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "py_utils.h"
#include "../generators.h"
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;

namespace Generators {

// Forward declare PyTensor
struct PyTensor {
  nb::ndarray<> array;
  std::vector<int64_t> shape;
  std::string dtype;
  
  PyTensor(nb::ndarray<> arr);
  nb::ndarray<> AsNumpy() const;
};

// Wrapper for NamedTensors - stores tensors as PyTensor objects
struct PyNamedTensors {
  std::unordered_map<std::string, std::shared_ptr<PyTensor>> tensors;
  
  void Set(const std::string& name, nb::object value);
  std::shared_ptr<PyTensor> Get(const std::string& name);
  void Delete(const std::string& name);
  bool Contains(const std::string& name) const;
  std::vector<std::string> Keys() const;
  size_t Size() const;
};

// Helper functions
nb::ndarray<> ConvertTensorToNumpy(const std::shared_ptr<Tensor>& tensor);
std::unique_ptr<PyNamedTensors> ConvertNamedTensors(std::unique_ptr<NamedTensors> cpp_tensors);

} // namespace Generators
