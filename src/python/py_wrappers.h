// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace Generators {

// Forward declarations for C++ types
class Model;
class GeneratorParams;
class Adapters;
class Config;
class MultiModalProcessor;

// Forward declarations of Python wrapper classes
// These are shared across all Python binding files to ensure consistent declarations

struct PyMultiModalProcessor;

struct PyModel {
  std::shared_ptr<Model> model;
  
  // Constructors - implemented in py_model.cpp
  PyModel(const std::string& config_path);
  PyModel(Config& config);
  
  // Methods - implemented in py_model.cpp
  std::string GetType() const;
  std::string GetDeviceType() const;
  PyMultiModalProcessor CreateMultiModalProcessor();
  
  // Inline getter
  std::shared_ptr<Model> GetModel() { return model; }
};

struct PyGeneratorParams {
  std::shared_ptr<GeneratorParams> params;
  
  // Constructor - implemented in py_generator_params.cpp
  PyGeneratorParams(PyModel& model);
  
  // Methods - implemented in py_generator_params.cpp
  void SetSearchOptions(const nb::kwargs& kwargs);
  void SetInputs(nb::ndarray<int32_t> input_ids);
  
  // Inline getter
  std::shared_ptr<GeneratorParams> GetParams() { return params; }
};

struct PyAdapters {
  std::shared_ptr<Adapters> adapters;
  
  // Constructor - implemented in py_adapters.cpp
  PyAdapters(PyModel& model);
  
  // Methods - implemented in py_adapters.cpp
  void LoadAdapter(std::string_view adapter_file_path, std::string_view adapter_name);
  void UnloadAdapter(std::string_view adapter_name);
  
  // Inline getter
  std::shared_ptr<Adapters> GetAdapters() { return adapters; }
};

}  // namespace Generators
