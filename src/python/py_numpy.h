// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "../models/onnxruntime_api.h"
#include <span>

namespace nb = nanobind;

namespace Generators {

// NumPy dtype mapping for float16
// NumPy uses dtype.num = 23 for float16
constexpr int NPY_FLOAT16 = 23;

// Convert numpy dtype to ONNX tensor type
inline ONNXTensorElementDataType NumpyToOnnxType(const nb::dlpack::dtype& dtype) {
  switch (static_cast<nb::dlpack::dtype_code>(dtype.code)) {
    case nb::dlpack::dtype_code::Int:
      switch (dtype.bits) {
        case 8: return Ort::TypeToTensorType<int8_t>;
        case 16: return Ort::TypeToTensorType<int16_t>;
        case 32: return Ort::TypeToTensorType<int32_t>;
        case 64: return Ort::TypeToTensorType<int64_t>;
      }
      break;
    case nb::dlpack::dtype_code::UInt:
      switch (dtype.bits) {
        case 8: return Ort::TypeToTensorType<uint8_t>;
        case 16: return Ort::TypeToTensorType<uint16_t>;
        case 32: return Ort::TypeToTensorType<uint32_t>;
        case 64: return Ort::TypeToTensorType<uint64_t>;
      }
      break;
    case nb::dlpack::dtype_code::Float:
      switch (dtype.bits) {
        case 16: return Ort::TypeToTensorType<Ort::Float16_t>;
        case 32: return Ort::TypeToTensorType<float>;
        case 64: return Ort::TypeToTensorType<double>;
      }
      break;
    case nb::dlpack::dtype_code::Bool:
      return Ort::TypeToTensorType<bool>;
  }
  throw std::runtime_error("Unsupported numpy dtype");
}

// Convert ONNX tensor type to numpy dtype
inline nb::dlpack::dtype OnnxToNumpyType(ONNXTensorElementDataType type) {
  switch (type) {
    case Ort::TypeToTensorType<bool>:
      return nb::dlpack::dtype{static_cast<uint8_t>(nb::dlpack::dtype_code::Bool), 8, 1};
    case Ort::TypeToTensorType<int8_t>:
      return nb::dlpack::dtype{static_cast<uint8_t>(nb::dlpack::dtype_code::Int), 8, 1};
    case Ort::TypeToTensorType<uint8_t>:
      return nb::dlpack::dtype{static_cast<uint8_t>(nb::dlpack::dtype_code::UInt), 8, 1};
    case Ort::TypeToTensorType<int16_t>:
      return nb::dlpack::dtype{static_cast<uint8_t>(nb::dlpack::dtype_code::Int), 16, 1};
    case Ort::TypeToTensorType<uint16_t>:
      return nb::dlpack::dtype{static_cast<uint8_t>(nb::dlpack::dtype_code::UInt), 16, 1};
    case Ort::TypeToTensorType<int32_t>:
      return nb::dlpack::dtype{static_cast<uint8_t>(nb::dlpack::dtype_code::Int), 32, 1};
    case Ort::TypeToTensorType<uint32_t>:
      return nb::dlpack::dtype{static_cast<uint8_t>(nb::dlpack::dtype_code::UInt), 32, 1};
    case Ort::TypeToTensorType<int64_t>:
      return nb::dlpack::dtype{static_cast<uint8_t>(nb::dlpack::dtype_code::Int), 64, 1};
    case Ort::TypeToTensorType<uint64_t>:
      return nb::dlpack::dtype{static_cast<uint8_t>(nb::dlpack::dtype_code::UInt), 64, 1};
    case Ort::TypeToTensorType<Ort::Float16_t>:
      return nb::dlpack::dtype{static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 16, 1};
    case Ort::TypeToTensorType<float>:
      return nb::dlpack::dtype{static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 32, 1};
    case Ort::TypeToTensorType<double>:
      return nb::dlpack::dtype{static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 64, 1};
    default:
      throw std::runtime_error("Unsupported onnx type");
  }
}

// Helper to convert nb::ndarray to std::span
template <typename T>
std::span<T> ToSpan(nb::ndarray<T, nb::ndim<1>, nb::c_contig> array) {
  return std::span<T>(array.data(), array.shape(0));
}

template <typename T>
std::span<const T> ToSpan(nb::ndarray<const T, nb::ndim<1>, nb::c_contig> array) {
  return std::span<const T>(array.data(), array.shape(0));
}

// Helper to convert std::vector to numpy array with given shape
template <typename T>
inline nb::ndarray<nb::numpy, T> ToNumpy(const std::vector<T>& data, const std::vector<int64_t>& shape) {
  // Create shape array for nanobind
  std::vector<size_t> nb_shape;
  for (auto dim : shape) {
    nb_shape.push_back(static_cast<size_t>(dim));
  }
  
  // Allocate Python-owned memory and copy data
  size_t total_size = data.size();
  T* buffer = new T[total_size];
  std::copy(data.begin(), data.end(), buffer);
  
  // Create capsule for memory management
  nb::capsule owner(buffer, [](void* p) noexcept {
    delete[] static_cast<T*>(p);
  });
  
  // Return ndarray with proper ownership
  return nb::ndarray<nb::numpy, T>(buffer, nb_shape.size(), nb_shape.data(), owner);
}

} // namespace Generators
