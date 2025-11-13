// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "py_utils.h"
#include "py_wrappers.h"
#include "../generators.h"  // Must include first for proper dependencies
#include "../config.h"
#include "../models/model.h"  // Now we can include this
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

namespace Generators {

// Forward declare PyMultiModalProcessor
struct PyMultiModalProcessor {
  std::shared_ptr<MultiModalProcessor> processor;
};

// Implementation of PyModel methods
PyModel::PyModel(const std::string& config_path) {
  model = CreateModel(GetOrtEnv(), config_path.c_str());
}

PyModel::PyModel(Config& config) {
  auto config_copy = std::make_unique<Config>(config);
  model = CreateModel(GetOrtEnv(), std::move(config_copy));
}

std::string PyModel::GetType() const {
  return model->config_->model.type;
}

std::string PyModel::GetDeviceType() const {
  return to_string(model->p_device_->GetType());
}

PyMultiModalProcessor PyModel::CreateMultiModalProcessor() {
  auto processor = model->CreateMultiModalProcessor();
  return PyMultiModalProcessor{std::move(processor)};
}

void BindModel(nb::module_& m) {
  nb::class_<PyModel>(m, "Model")
    .def(nb::init<const std::string&>(), "config_path"_a)
    .def(nb::init<Config&>(), "config"_a)
    .def_prop_ro("type", &PyModel::GetType)
    .def_prop_ro("device_type", &PyModel::GetDeviceType)
    .def("create_multimodal_processor", &PyModel::CreateMultiModalProcessor);
}

} // namespace Generators
