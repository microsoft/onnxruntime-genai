// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "py_utils.h"
#include "../generators.h"
#include "../models/adapters.h"
#include "../models/model.h"
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

namespace Generators {

// Forward declare PyModel
struct PyModel {
  std::shared_ptr<Model> model;
  std::shared_ptr<Model> GetModel() { return model; }
};

// Wrapper for Adapters
struct PyAdapters {
  std::shared_ptr<Adapters> adapters;
  
  PyAdapters(PyModel& model) {
    adapters = std::make_shared<Adapters>(model.GetModel().get());
  }
  
  void LoadAdapter(std::string_view adapter_file_path, std::string_view adapter_name) {
    std::string path_str(adapter_file_path);
    std::string name_str(adapter_name);
    adapters->LoadAdapter(path_str.c_str(), name_str);
  }
  
  void UnloadAdapter(std::string_view adapter_name) {
    std::string name_str(adapter_name);
    adapters->UnloadAdapter(name_str);
  }
  
  std::shared_ptr<Adapters> GetAdapters() { return adapters; }
};

void BindAdapters(nb::module_& m) {
  nb::class_<PyAdapters>(m, "Adapters")
    .def(nb::init<PyModel&>())
    .def("load", &PyAdapters::LoadAdapter,
         nb::arg("adapter_file_path"),
         nb::arg("adapter_name"))
    .def("unload", &PyAdapters::UnloadAdapter,
         nb::arg("adapter_name"));
}

} // namespace Generators
