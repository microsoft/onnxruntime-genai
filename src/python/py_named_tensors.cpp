// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "py_named_tensors.h"
#include "py_utils.h"
#include "../generators.h"
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace Generators {

// PyTensor implementation
PyTensor::PyTensor(nb::ndarray<> arr) : array(arr) {
  for (size_t i = 0; i < arr.ndim(); ++i) {
    shape.push_back(arr.shape(i));
  }
  // Store dtype info
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

nb::ndarray<> PyTensor::AsNumpy() const {
  return array;
}

// PyNamedTensors implementation
void PyNamedTensors::Set(const std::string& name, nb::object value) {
  // If it's a plain numpy array, wrap it in a Tensor
  if (nb::isinstance<nb::ndarray<>>(value)) {
    auto arr = nb::cast<nb::ndarray<>>(value);
    tensors[name] = std::make_shared<PyTensor>(arr);
  } else if (nb::isinstance<PyTensor>(value)) {
    // It's already a PyTensor
    tensors[name] = std::make_shared<PyTensor>(nb::cast<PyTensor&>(value));
  } else {
    throw std::runtime_error("NamedTensors only accepts numpy arrays or og.Tensor objects");
  }
}

std::shared_ptr<PyTensor> PyNamedTensors::Get(const std::string& name) {
  auto it = tensors.find(name);
  if (it == tensors.end()) {
    throw std::runtime_error("Tensor not found: " + name);
  }
  return it->second;
}

void PyNamedTensors::Delete(const std::string& name) {
  tensors.erase(name);
}

bool PyNamedTensors::Contains(const std::string& name) const {
  return tensors.find(name) != tensors.end();
}

std::vector<std::string> PyNamedTensors::Keys() const {
  std::vector<std::string> keys;
  for (const auto& pair : tensors) {
    keys.push_back(pair.first);
  }
  return keys;
}

size_t PyNamedTensors::Size() const {
  return tensors.size();
}

// Implementation of ConvertTensorToNumpy
nb::ndarray<> ConvertTensorToNumpy(const std::shared_ptr<Tensor>& tensor) {
  OrtValue* ort_value = tensor->GetOrtTensor();
  if (!ort_value) {
    throw std::runtime_error("Tensor has no underlying OrtValue");
  }
  
  // Cast to C API type (binary compatible)
  ::OrtValue* c_api_value = reinterpret_cast<::OrtValue*>(ort_value);
  
  // Get tensor type and shape info
  ::OrtTensorTypeAndShapeInfo* type_info_raw;
  Ort::ThrowOnError(Ort::api->GetTensorTypeAndShape(c_api_value, &type_info_raw));
  
  auto deleter = [](::OrtTensorTypeAndShapeInfo* p) { Ort::api->ReleaseTensorTypeAndShapeInfo(p); };
  std::unique_ptr<::OrtTensorTypeAndShapeInfo, decltype(deleter)> type_info(type_info_raw, deleter);
  
  // Get shape
  size_t num_dims;
  Ort::ThrowOnError(Ort::api->GetDimensionsCount(type_info.get(), &num_dims));
  std::vector<int64_t> shape_vector(num_dims);
  Ort::ThrowOnError(Ort::api->GetDimensions(type_info.get(), shape_vector.data(), num_dims));
  
  // Get element count
  size_t element_count;
  Ort::ThrowOnError(Ort::api->GetTensorShapeElementCount(type_info.get(), &element_count));
  
  // Convert to size_t for nanobind
  std::vector<size_t> shape(shape_vector.begin(), shape_vector.end());
  
  // Get element type
  ONNXTensorElementDataType element_type;
  Ort::ThrowOnError(Ort::api->GetTensorElementType(type_info.get(), &element_type));
  
  // Get data pointer
  void* data_raw;
  Ort::ThrowOnError(Ort::api->GetTensorMutableData(c_api_value, &data_raw));
  
  // Clone data to CPU memory that Python owns and return as ndarray<>
  // Support all types that pybind11 version supported
  switch (element_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
      const bool* data = static_cast<const bool*>(data_raw);
      bool* buffer = new bool[element_count];
      std::memcpy(buffer, data, element_count * sizeof(bool));
      nb::capsule owner(buffer, [](void* p) noexcept { delete[] static_cast<bool*>(p); });
      nb::ndarray<nb::numpy, bool> typed_array(buffer, shape.size(), shape.data(), owner);
      return nb::ndarray<>(typed_array);
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
      const int8_t* data = static_cast<const int8_t*>(data_raw);
      int8_t* buffer = new int8_t[element_count];
      std::memcpy(buffer, data, element_count * sizeof(int8_t));
      nb::capsule owner(buffer, [](void* p) noexcept { delete[] static_cast<int8_t*>(p); });
      nb::ndarray<nb::numpy, int8_t> typed_array(buffer, shape.size(), shape.data(), owner);
      return nb::ndarray<>(typed_array);
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
      const uint8_t* data = static_cast<const uint8_t*>(data_raw);
      uint8_t* buffer = new uint8_t[element_count];
      std::memcpy(buffer, data, element_count * sizeof(uint8_t));
      nb::capsule owner(buffer, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
      nb::ndarray<nb::numpy, uint8_t> typed_array(buffer, shape.size(), shape.data(), owner);
      return nb::ndarray<>(typed_array);
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
      const int16_t* data = static_cast<const int16_t*>(data_raw);
      int16_t* buffer = new int16_t[element_count];
      std::memcpy(buffer, data, element_count * sizeof(int16_t));
      nb::capsule owner(buffer, [](void* p) noexcept { delete[] static_cast<int16_t*>(p); });
      nb::ndarray<nb::numpy, int16_t> typed_array(buffer, shape.size(), shape.data(), owner);
      return nb::ndarray<>(typed_array);
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
      const uint16_t* data = static_cast<const uint16_t*>(data_raw);
      uint16_t* buffer = new uint16_t[element_count];
      std::memcpy(buffer, data, element_count * sizeof(uint16_t));
      nb::capsule owner(buffer, [](void* p) noexcept { delete[] static_cast<uint16_t*>(p); });
      nb::ndarray<nb::numpy, uint16_t> typed_array(buffer, shape.size(), shape.data(), owner);
      return nb::ndarray<>(typed_array);
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
      const int32_t* data = static_cast<const int32_t*>(data_raw);
      int32_t* buffer = new int32_t[element_count];
      std::memcpy(buffer, data, element_count * sizeof(int32_t));
      nb::capsule owner(buffer, [](void* p) noexcept { delete[] static_cast<int32_t*>(p); });
      nb::ndarray<nb::numpy, int32_t> typed_array(buffer, shape.size(), shape.data(), owner);
      return nb::ndarray<>(typed_array);
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: {
      const uint32_t* data = static_cast<const uint32_t*>(data_raw);
      uint32_t* buffer = new uint32_t[element_count];
      std::memcpy(buffer, data, element_count * sizeof(uint32_t));
      nb::capsule owner(buffer, [](void* p) noexcept { delete[] static_cast<uint32_t*>(p); });
      nb::ndarray<nb::numpy, uint32_t> typed_array(buffer, shape.size(), shape.data(), owner);
      return nb::ndarray<>(typed_array);
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
      const int64_t* data = static_cast<const int64_t*>(data_raw);
      int64_t* buffer = new int64_t[element_count];
      std::memcpy(buffer, data, element_count * sizeof(int64_t));
      nb::capsule owner(buffer, [](void* p) noexcept { delete[] static_cast<int64_t*>(p); });
      nb::ndarray<nb::numpy, int64_t> typed_array(buffer, shape.size(), shape.data(), owner);
      return nb::ndarray<>(typed_array);
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: {
      const uint64_t* data = static_cast<const uint64_t*>(data_raw);
      uint64_t* buffer = new uint64_t[element_count];
      std::memcpy(buffer, data, element_count * sizeof(uint64_t));
      nb::capsule owner(buffer, [](void* p) noexcept { delete[] static_cast<uint64_t*>(p); });
      nb::ndarray<nb::numpy, uint64_t> typed_array(buffer, shape.size(), shape.data(), owner);
      return nb::ndarray<>(typed_array);
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
      // For fp16, treat as uint16 buffer but with numpy float16 dtype
      const uint16_t* data = reinterpret_cast<const uint16_t*>(data_raw);
      uint16_t* buffer = new uint16_t[element_count];
      std::memcpy(buffer, data, element_count * sizeof(uint16_t));
      nb::capsule owner(buffer, [](void* p) noexcept { delete[] static_cast<uint16_t*>(p); });
      
      // Create dtype for float16 (NumPy type code 23)
      nb::dlpack::dtype fp16_dtype;
      fp16_dtype.code = (uint8_t)nb::dlpack::dtype_code::Float;
      fp16_dtype.bits = 16;
      fp16_dtype.lanes = 1;
      
      nb::ndarray<nb::numpy> typed_array(buffer, shape.size(), shape.data(), owner, nullptr, fp16_dtype);
      return nb::ndarray<>(typed_array);
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      const float* data = static_cast<const float*>(data_raw);
      float* buffer = new float[element_count];
      std::memcpy(buffer, data, element_count * sizeof(float));
      nb::capsule owner(buffer, [](void* p) noexcept { delete[] static_cast<float*>(p); });
      nb::ndarray<nb::numpy, float> typed_array(buffer, shape.size(), shape.data(), owner);
      return nb::ndarray<>(typed_array);
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
      const double* data = static_cast<const double*>(data_raw);
      double* buffer = new double[element_count];
      std::memcpy(buffer, data, element_count * sizeof(double));
      nb::capsule owner(buffer, [](void* p) noexcept { delete[] static_cast<double*>(p); });
      nb::ndarray<nb::numpy, double> typed_array(buffer, shape.size(), shape.data(), owner);
      return nb::ndarray<>(typed_array);
    }
    default:
      throw std::runtime_error("Unsupported tensor element type: " + std::to_string(element_type));
  }
}

// Implementation of ConvertNamedTensors
std::unique_ptr<PyNamedTensors> ConvertNamedTensors(std::unique_ptr<NamedTensors> cpp_tensors) {
  auto py_tensors = std::make_unique<PyNamedTensors>();
  
  for (const auto& [name, tensor] : *cpp_tensors) {
    // Convert each C++ Tensor to numpy array
    auto numpy_array = ConvertTensorToNumpy(tensor);
    // Create PyTensor from numpy array
    py_tensors->tensors[name] = std::make_shared<PyTensor>(numpy_array);
  }
  
  return py_tensors;
}

void BindNamedTensors(nb::module_& m) {
  nb::class_<PyNamedTensors>(m, "NamedTensors")
    .def(nb::init<>())
    .def("__setitem__", &PyNamedTensors::Set, "name"_a, "value"_a)
    .def("__getitem__", &PyNamedTensors::Get, "name"_a)
    .def("__delitem__", &PyNamedTensors::Delete, "name"_a)
    .def("__contains__", &PyNamedTensors::Contains, "name"_a)
    .def("__len__", &PyNamedTensors::Size)
    .def("keys", &PyNamedTensors::Keys);
}

} // namespace Generators
