// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string_view.h>

namespace nb = nanobind;
using namespace nb::literals;

// Forward declarations - actual types are defined in their respective binding files
namespace Generators {

// Forward declare wrapper classes
struct PyModel;
struct PyTokenizer;
struct PyTokenizerStream;
struct PyGeneratorParams;
struct PyGenerator;
struct PyTensor;

// Forward declarations for binding functions
void BindGeneratorParams(nb::module_& m);
void BindTokenizer(nb::module_& m);
void BindTokenizerStream(nb::module_& m);
void BindNamedTensors(nb::module_& m);
void BindTensor(nb::module_& m);
void BindConfig(nb::module_& m);
void BindModel(nb::module_& m);
void BindGenerator(nb::module_& m);
void BindMultiModal(nb::module_& m);
void BindAdapters(nb::module_& m);
void BindEngine(nb::module_& m);

} // namespace Generators
