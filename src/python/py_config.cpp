// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "py_utils.h"
#include "../generators.h"  // Must include first for proper dependencies
#include "../config.h"
#include "../filesystem.h"
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <span>

namespace nb = nanobind;

namespace Generators {

// Helper functions declared in config.cpp
void ClearProviders(Config& config);
void SetProviderOption(Config& config, std::string_view provider_name, std::string_view option_name, std::string_view option_value);
void SetDecoderProviderOptionsHardwareDeviceType(Config& config, std::string_view provider_name, std::string_view hardware_device_type);
void SetDecoderProviderOptionsHardwareDeviceId(Config& config, std::string_view provider_name, uint32_t hardware_device_id);
void SetDecoderProviderOptionsHardwareVendorId(Config& config, std::string_view provider_name, uint32_t hardware_vendor_id);
void ClearDecoderProviderOptionsHardwareDeviceType(Config& config, std::string_view provider_name);
void ClearDecoderProviderOptionsHardwareDeviceId(Config& config, std::string_view provider_name);
void ClearDecoderProviderOptionsHardwareVendorId(Config& config, std::string_view provider_name);

void BindConfig(nb::module_& m) {
  nb::class_<Config>(m, "Config")
    .def("__init__", [](Config* config, const std::string& config_path) {
      new (config) Config(fs::path(config_path), std::string_view{});
    })
    .def("clear_providers", [](Config& config) {
      ClearProviders(config);
    })
    .def("append_provider", [](Config& config, const std::string& provider) {
      // AppendProvider just adds the provider without options
      SetProviderOption(config, provider, {}, {});
    })
    .def("set_provider_option", [](Config& config, const std::string& provider, 
                                    const std::string& key, const std::string& value) {
      SetProviderOption(config, provider, key, value);
    })
    .def("set_decoder_provider_options_hardware_device_type", 
         [](Config& config, const std::string& provider, const std::string& device_type) {
      SetDecoderProviderOptionsHardwareDeviceType(config, provider, device_type);
    })
    .def("set_decoder_provider_options_hardware_device_id",
         [](Config& config, const std::string& provider, uint32_t device_id) {
      SetDecoderProviderOptionsHardwareDeviceId(config, provider, device_id);
    })
    .def("set_decoder_provider_options_hardware_vendor_id",
         [](Config& config, const std::string& provider, uint32_t vendor_id) {
      SetDecoderProviderOptionsHardwareVendorId(config, provider, vendor_id);
    })
    .def("clear_decoder_provider_options_hardware_device_type",
         [](Config& config, const std::string& provider) {
      ClearDecoderProviderOptionsHardwareDeviceType(config, provider);
    })
    .def("clear_decoder_provider_options_hardware_device_id",
         [](Config& config, const std::string& provider) {
      ClearDecoderProviderOptionsHardwareDeviceId(config, provider);
    })
    .def("clear_decoder_provider_options_hardware_vendor_id",
         [](Config& config, const std::string& provider) {
      ClearDecoderProviderOptionsHardwareVendorId(config, provider);
    })
    .def("add_model_data", [](Config& config, const std::string& model_filename, nb::object model_data_obj) {
      // Support bytes, bytearray, memoryview - anything with buffer protocol
      if (PyObject_CheckBuffer(model_data_obj.ptr())) {
        Py_buffer view;
        if (PyObject_GetBuffer(model_data_obj.ptr(), &view, PyBUF_SIMPLE) == -1) {
          throw std::runtime_error("Failed to get buffer from model_data");
        }
        
        const auto* byte_ptr = reinterpret_cast<const std::byte*>(view.buf);
        size_t data_size = view.len;
        
        // Release the buffer immediately after copying the pointer
        PyBuffer_Release(&view);
        
        if (byte_ptr == nullptr || data_size == 0) {
          throw std::runtime_error("Expected a valid model data pointer and length. Received nullptr or zero length.");
        }
        
        const auto emplaced = config.model_data_spans_.emplace(
          model_filename, 
          std::span<const std::byte>(byte_ptr, data_size)
        );
        
        if (!emplaced.second) {
          throw std::runtime_error("Model data for '" + model_filename +
                                   "' was already added previously. "
                                   "If you want to replace it, please remove it first.");
        }
      } else {
        throw std::runtime_error("model_data must be a bytes-like object (bytes, bytearray, or memoryview)");
      }
    })
    .def("remove_model_data", [](Config& config, const std::string& model_filename) {
      auto it = config.model_data_spans_.find(model_filename);
      if (it == config.model_data_spans_.end()) {
        throw std::runtime_error("Model data for '" + model_filename + "' was not found.");
      }
      config.model_data_spans_.erase(it);
    });
}

} // namespace Generators
