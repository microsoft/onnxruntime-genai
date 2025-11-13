// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "py_utils.h"
#include "py_wrappers.h"
#include "../generators.h"
#include "../config.h"
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace Generators {

// Forward declarations of config helper functions
void SetSearchNumber(Config::Search& search, std::string_view name, double value);
void SetSearchBool(Config::Search& search, std::string_view name, bool value);

// Implementation of PyGeneratorParams methods
PyGeneratorParams::PyGeneratorParams(PyModel& model) {
  params = CreateGeneratorParams(*model.GetModel());
}
  
void PyGeneratorParams::SetSearchOptions(const nb::kwargs& kwargs) {
  for (const auto& [key, value] : kwargs) {
    std::string name = nb::cast<std::string>(key);
    
    // Check if it's a bool
    if (nb::isinstance<nb::bool_>(value)) {
      bool val = nb::cast<bool>(value);
      SetSearchBool(params->search, name, val);
    }
    // Check if it's a float
    else if (nb::isinstance<nb::float_>(value)) {
      double val = nb::cast<double>(value);
      SetSearchNumber(params->search, name, val);
    }
    // Check if it's an int
    else if (nb::isinstance<nb::int_>(value)) {
      double val = static_cast<double>(nb::cast<int>(value));
      SetSearchNumber(params->search, name, val);
    }
    else {
      throw std::runtime_error("Unsupported search option type for: " + name);
    }
  }
}

void PyGeneratorParams::SetInputs(nb::ndarray<int32_t> input_ids) {
  // For future: handle input_ids tensor
  // For now, we'll use append_tokens in Generator instead
}

void BindGeneratorParams(nb::module_& m) {
  nb::class_<PyGeneratorParams>(m, "GeneratorParams")
    .def(nb::init<PyModel&>(), "model"_a)
    .def("set_search_options", &PyGeneratorParams::SetSearchOptions)
    .def("set_inputs", &PyGeneratorParams::SetInputs, "input_ids"_a);
}

} // namespace Generators
