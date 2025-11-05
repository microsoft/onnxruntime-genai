// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "py_utils.h"
#include "py_numpy.h"  // For ToNumpy helper
#include "../generators.h"
#include "../search.h"  // Needed for Generator::search_
#include "../models/model.h"  // Needed for State
#include "../models/adapters.h"  // Needed for Adapters
#include "../constrained_logits_processor.h"  // Needed for ConstrainedLogitsProcessor
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>

#if USE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace nb = nanobind;

namespace Generators {

// Forward declare wrapper classes - defined in their respective .cpp files
struct PyModel {
  std::shared_ptr<Model> model;
  std::shared_ptr<Model> GetModel() { return model; }
};

struct PyGeneratorParams {
  std::shared_ptr<GeneratorParams> params;
  std::shared_ptr<GeneratorParams> GetParams() { return params; }
};

struct PyAdapters {
  std::shared_ptr<Adapters> adapters;
  std::shared_ptr<Adapters> GetAdapters() { return adapters; }
};

// Wrapper for Generator
struct PyGenerator {
  std::unique_ptr<Generator> generator;
  
  PyGenerator(PyModel& model, PyGeneratorParams& params) {
    generator = CreateGenerator(*model.GetModel(), *params.GetParams());
  }
  
  ~PyGenerator() {
    // Explicitly reset to ensure destruction
    generator.reset();
  }
  
  bool IsDone() const {
    return generator->IsDone();
  }
  
  void AppendTokens(nb::ndarray<nb::numpy, int32_t> tokens_array) {
    // Convert numpy array to span - handles any-dimensional arrays by flattening
    // This matches the old pybind11 behavior
    int32_t* data = tokens_array.data();  // Non-const to accept writable arrays
    size_t total_size = tokens_array.size();  // Total elements across all dimensions
    
    // Create a span and pass to generator (it will copy the data to device)
    cpu_span<const int32_t> token_span(data, total_size);
    generator->AppendTokens(token_span);
  }
  
  void AppendTokens(const std::vector<int32_t>& tokens) {
    generator->AppendTokens(cpu_span<const int32_t>(tokens.data(), tokens.size()));
  }
  
  void GenerateNextToken() {
    generator->GenerateNextToken();
  }
  
  std::vector<int32_t> GetNextTokens() {
    // Get the next tokens from search and copy from device to CPU
    auto tokens = generator->search_->GetNextTokens().CopyDeviceToCpu();
    return std::vector<int32_t>(tokens.begin(), tokens.end());
  }
  
  std::vector<int32_t> GetSequence(size_t index) {
    auto sequence = generator->GetSequence(index);
    // DeviceSpan - need to copy to CPU
    auto cpu_sequence = sequence.CopyDeviceToCpu();
    return std::vector<int32_t>(cpu_sequence.begin(), cpu_sequence.end());
  }
  
  nb::ndarray<nb::numpy, float> GetLogits() {
    // Get logits from generator
    auto logits_span = generator->GetLogits();
    // Copy to CPU
    auto cpu_logits = logits_span.CopyDeviceToCpu();
    
    // Create numpy array
    size_t size = cpu_logits.size();
    float* data = new float[size];
    std::copy(cpu_logits.begin(), cpu_logits.end(), data);
    
    // Create cleanup capsule for the buffer
    nb::capsule owner(data, [](void* p) noexcept {
      delete[] static_cast<float*>(p);
    });
    
    // Create numpy array with ownership transfer
    size_t shape[1] = {size};
    return nb::ndarray<nb::numpy, float>(data, 1, shape, owner, nullptr, 
                                         nb::dtype<float>());
  }
  
  void SetLogits(nb::ndarray<float> logits_array) {
    // Get the current logits buffer from the generator
    auto current_logits = generator->GetLogits();
    
    // Verify size matches
    if (static_cast<size_t>(logits_array.size()) != current_logits.size()) {
      throw std::runtime_error("Generator::SetLogits passed an array of size " + 
                              std::to_string(logits_array.size()) + 
                              " but should be size " + 
                              std::to_string(current_logits.size()));
    }
    
    // Get raw pointer from numpy array
    const float* data = logits_array.data();
    
    // Create a span from the numpy array data (CPU memory)
    auto cpu_span = std::span<const float>(data, logits_array.size());
    
    // Copy from CPU span to the device span
    // CpuSpan() gets the CPU-side buffer that will be copied to device
    copy(cpu_span, current_logits.CpuSpan());
    current_logits.CopyCpuToDevice();
    
    // Set the logits (this will mark computed_logits_ = true)
    generator->SetLogits(current_logits);
  }
  
  void RewindTo(size_t new_length) {
    generator->RewindToLength(new_length);
  }
  
  void SetActiveAdapter(PyAdapters& adapters, std::string_view adapter_name) {
    if (!generator || !generator->state_) {
      throw std::runtime_error("Generator or generator state is null");
    }
    std::string name_str(adapter_name);
    generator->state_->SetActiveAdapter(adapters.GetAdapters().get(), name_str);
  }
  
  nb::ndarray<nb::numpy, float> GetOutput(std::string_view name) {
    // Check that generator and state are valid
    if (!generator) {
      throw std::runtime_error("Generator is null");
    }
    
    if (!generator->state_) {
      throw std::runtime_error("Generator state is null");
    }
    
    // For all outputs, get from state (including logits)
    // Note: In Generators namespace, OrtValue refers to the C++ wrapper (same memory layout as C API)
    std::string name_str(name);
    OrtValue* ort_value = generator->state_->GetOutput(name_str.c_str());
    
    // Check if the output exists
    if (!ort_value) {
      throw std::runtime_error(std::string("Output '") + name_str + "' not found. Available outputs might be empty or the name is incorrect.");
    }
    
    // Cast to C API type (they are binary compatible - same struct, different namespace)
    ::OrtValue* c_api_value = reinterpret_cast<::OrtValue*>(ort_value);
    
    // Use C API to check if it's a tensor
    int is_tensor = 0;
    OrtStatus* status = Ort::api->IsTensor(c_api_value, &is_tensor);
    Ort::ThrowOnError(status);
    
    if (!is_tensor) {
      throw std::runtime_error(std::string("Output '") + name_str + "' is not a tensor");
    }
    
    // Use C API to get tensor information - ::OrtTensorTypeAndShapeInfo is a C API opaque pointer
    ::OrtTensorTypeAndShapeInfo* type_info_raw;
    Ort::ThrowOnError(Ort::api->GetTensorTypeAndShape(c_api_value, &type_info_raw));
    
    // Wrap in unique_ptr with custom deleter for RAII
    auto deleter = [](::OrtTensorTypeAndShapeInfo* p) { Ort::api->ReleaseTensorTypeAndShapeInfo(p); };
    std::unique_ptr<::OrtTensorTypeAndShapeInfo, decltype(deleter)> type_info(type_info_raw, deleter);
    
    // Get element type
    ONNXTensorElementDataType element_type;
    Ort::ThrowOnError(Ort::api->GetTensorElementType(type_info.get(), &element_type));
    
    // Get shape
    size_t num_dims;
    Ort::ThrowOnError(Ort::api->GetDimensionsCount(type_info.get(), &num_dims));
    std::vector<int64_t> shape_vector(num_dims);
    Ort::ThrowOnError(Ort::api->GetDimensions(type_info.get(), shape_vector.data(), num_dims));
    
    // Get element count
    size_t element_count;
    Ort::ThrowOnError(Ort::api->GetTensorShapeElementCount(type_info.get(), &element_count));
    
    // Convert shape to size_t for nanobind
    std::vector<size_t> shape(shape_vector.begin(), shape_vector.end());
    
    // Check memory location
    const ::OrtMemoryInfo* mem_info;
    Ort::ThrowOnError(Ort::api->GetTensorMemoryInfo(c_api_value, &mem_info));
    
    OrtMemoryInfoDeviceType device_type;
    Ort::api->MemoryInfoGetDeviceType(mem_info, &device_type);
    
    // Handle different data types
    if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      void* data_raw;
      Ort::ThrowOnError(Ort::api->GetTensorMutableData(c_api_value, &data_raw));
      const float* data = static_cast<const float*>(data_raw);
      
      if (!data) {
        throw std::runtime_error(std::string("Failed to get tensor data for output '") + name_str + "'");
      }
      
      float* buffer = new float[element_count];
      
      if (device_type == OrtMemoryInfoDeviceType_GPU) {
#if USE_CUDA
        auto err = cudaMemcpy(buffer, data, element_count * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
          delete[] buffer;
          throw std::runtime_error(std::string("CUDA memcpy failed: ") + cudaGetErrorString(err));
        }
#else
        delete[] buffer;
        throw std::runtime_error("GPU memory detected but CUDA support is not enabled");
#endif
      } else {
        std::copy(data, data + element_count, buffer);
      }
      
      nb::capsule owner(buffer, [](void* p) noexcept {
        delete[] static_cast<float*>(p);
      });
      
      return nb::ndarray<nb::numpy, float>(buffer, shape.size(), shape.data(), owner, nullptr, nb::dtype<float>());
    } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      // For fp16, we convert to float32 for numpy compatibility
      void* data_raw;
      Ort::ThrowOnError(Ort::api->GetTensorMutableData(c_api_value, &data_raw));
      const uint16_t* data = static_cast<const uint16_t*>(data_raw);
      
      if (!data) {
        throw std::runtime_error(std::string("Failed to get tensor data for output '") + name_str + "'");
      }
      
      // Allocate host buffer for fp16 data
      std::vector<uint16_t> fp16_buffer(element_count);
      
      if (device_type == OrtMemoryInfoDeviceType_GPU) {
#if USE_CUDA
        auto err = cudaMemcpy(fp16_buffer.data(), data, element_count * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
          throw std::runtime_error(std::string("CUDA memcpy failed: ") + cudaGetErrorString(err));
        }
#else
        throw std::runtime_error("GPU memory detected but CUDA support is not enabled");
#endif
      } else {
        std::copy(data, data + element_count, fp16_buffer.data());
      }
      
      // Convert fp16 to fp32
      float* buffer = new float[element_count];
      for (size_t i = 0; i < element_count; ++i) {
        // Simple fp16 to fp32 conversion
        uint16_t fp16 = fp16_buffer[i];
        uint32_t sign = (fp16 & 0x8000) << 16;
        uint32_t exp = (fp16 & 0x7C00) >> 10;
        uint32_t mantissa = (fp16 & 0x03FF);
        
        uint32_t fp32;
        if (exp == 0) {
          if (mantissa == 0) {
            fp32 = sign;  // Zero
          } else {
            // Denormalized number - normalize it
            exp = 1;
            while ((mantissa & 0x400) == 0) {
              mantissa <<= 1;
              exp--;
            }
            mantissa &= 0x3FF;
            fp32 = sign | ((exp + 112) << 23) | (mantissa << 13);
          }
        } else if (exp == 0x1F) {
          fp32 = sign | 0x7F800000 | (mantissa << 13);  // Inf or NaN
        } else {
          fp32 = sign | ((exp + 112) << 23) | (mantissa << 13);
        }
        
        std::memcpy(&buffer[i], &fp32, sizeof(float));
      }
      
      nb::capsule owner(buffer, [](void* p) noexcept {
        delete[] static_cast<float*>(p);
      });
      
      return nb::ndarray<nb::numpy, float>(buffer, shape.size(), shape.data(), owner, nullptr, nb::dtype<float>());
    }
    
    throw std::runtime_error(std::string("Unsupported tensor element type for output '") + name_str + "'. Only float32 and float16 are currently supported.");
  }
};

void BindGenerator(nb::module_& m) {
  nb::class_<PyGenerator>(m, "Generator")
    .def("__init__", [](PyGenerator* self, PyModel& model, PyGeneratorParams& params) {
      new (self) PyGenerator(model, params);
    })
    .def("is_done", &PyGenerator::IsDone)
    .def("append_tokens", [](PyGenerator& self, nb::ndarray<nb::numpy, int32_t> tokens) {
      self.AppendTokens(tokens);
    }, "tokens"_a)
    .def("append_tokens", nb::overload_cast<const std::vector<int32_t>&>(&PyGenerator::AppendTokens), "tokens"_a)
    .def("generate_next_token", &PyGenerator::GenerateNextToken)
    .def("get_next_tokens", &PyGenerator::GetNextTokens)
    .def("get_sequence", &PyGenerator::GetSequence, "index"_a)
    .def("get_logits", &PyGenerator::GetLogits)
    .def("set_logits", &PyGenerator::SetLogits, "logits"_a)
    .def("get_output", &PyGenerator::GetOutput, "name"_a, "Get a model output by name")
    .def("rewind_to", &PyGenerator::RewindTo, "new_length"_a)
    .def("set_active_adapter", &PyGenerator::SetActiveAdapter, "adapters"_a, "adapter_name"_a);
}

} // namespace Generators
