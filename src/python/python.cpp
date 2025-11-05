// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "py_utils.h"
#include <nanobind/nanobind.h>
#include "../models/onnxruntime_api.h"

namespace nb = nanobind;

// Module initialization
NB_MODULE(onnxruntime_genai, m) {
  m.doc() = "ONNX Runtime Generate AI";

  // Bind classes
  Generators::BindTensor(m);
  Generators::BindConfig(m);
  Generators::BindModel(m);
  Generators::BindTokenizer(m);
  Generators::BindGeneratorParams(m);
  Generators::BindGenerator(m);
  Generators::BindNamedTensors(m);
  Generators::BindMultiModal(m);
  Generators::BindAdapters(m);
  Generators::BindEngine(m);

  // Device availability checks
  m.def("is_cuda_available", []() { return USE_CUDA != 0; });
  m.def("is_dml_available", []() { return USE_DML != 0; });
  m.def("is_rocm_available", []() { return USE_ROCM != 0; });
  m.def("is_webgpu_available", []() { return true; });
  m.def("is_qnn_available", []() { return true; });
  m.def("is_openvino_available", []() { return true; });

  // GPU device management
  m.def("set_current_gpu_device_id", [](int device_id) { 
    Ort::SetCurrentGpuDeviceId(device_id); 
  });
  m.def("get_current_gpu_device_id", []() { 
    return Ort::GetCurrentGpuDeviceId(); 
  });
  
  // Logging functions
  m.def("set_log_options", [](const nb::kwargs& kwargs) {
    for (const auto& [key, value] : kwargs) {
      auto name = nb::cast<std::string>(key);
      
      // Check if it's a bool
      if (nb::isinstance<nb::bool_>(value)) {
        Generators::SetLogBool(name, nb::cast<bool>(value));
      }
      // Check if it's a string
      else if (nb::isinstance<nb::str>(value)) {
        Generators::SetLogString(name, nb::cast<std::string>(value));
      }
      else {
        throw std::runtime_error("Unknown log option type for '" + name + "', must be bool or string");
      }
    }
  }, "Configure logging options. Pass keyword arguments with bool or string values.");
  
  m.def("set_log_callback", [](nb::object callback) {
    static nb::object log_callback;  // Keep callback alive
    
    if (callback.is_none()) {
      // Clear callback
      Generators::SetLogCallback(nullptr);
      log_callback = nb::none();
    } else {
      // Set callback
      log_callback = callback;
      Generators::SetLogCallback([](const char* message, size_t length) {
        // Acquire GIL before calling Python
        nb::gil_scoped_acquire gil;
        log_callback(nb::str(message, length));
      });
    }
  }, nb::arg("callback").none(), "Set a callback function for log messages. Pass None to clear.");
  
  // Register cleanup function to run before Python shutdown
  // This ensures our objects are destroyed before LeakChecked counts are checked
  auto cleanup = []() {
    // Force Python GC to run before C++ static destructors
    PyGC_Collect();
  };
  
  // Use Python's atexit to register cleanup
  auto atexit = nb::module_::import_("atexit");
  atexit.attr("register")(nb::cpp_function(cleanup));
}
