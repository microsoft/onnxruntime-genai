// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "oga_object.h"

namespace OgaPy {

struct OgaConfig : OgaObject {
  explicit OgaConfig(::OgaConfig* p) : ptr_(p) {}
  ~OgaConfig() override { if (ptr_) OgaDestroyConfig(ptr_); }
  ::OgaConfig* get() const { return ptr_; }
  
  // Clear all providers
  void ClearProviders() {
    OgaCheckResult(OgaConfigClearProviders(ptr_));
  }
  
  // Append a provider
  void AppendProvider(const char* provider) {
    OgaCheckResult(OgaConfigAppendProvider(ptr_, provider));
  }
  
  // Set a provider option
  void SetProviderOption(const char* provider, const char* key, const char* value) {
    OgaCheckResult(OgaConfigSetProviderOption(ptr_, provider, key, value));
  }
  
  // Add model data to load the model from memory
  void AddModelData(const char* model_filename, const void* model_data, size_t model_data_length) {
    OgaCheckResult(OgaConfigAddModelData(ptr_, model_filename, model_data, model_data_length));
  }
  
  // Remove model data
  void RemoveModelData(const char* model_filename) {
    OgaCheckResult(OgaConfigRemoveModelData(ptr_, model_filename));
  }
  
  // Set decoder provider hardware device type
  void SetDecoderProviderOptionsHardwareDeviceType(const char* provider, const char* hardware_device_type) {
    OgaCheckResult(OgaConfigSetDecoderProviderOptionsHardwareDeviceType(ptr_, provider, hardware_device_type));
  }
  
  // Set decoder provider hardware device ID
  void SetDecoderProviderOptionsHardwareDeviceId(const char* provider, uint32_t hardware_device_id) {
    OgaCheckResult(OgaConfigSetDecoderProviderOptionsHardwareDeviceId(ptr_, provider, hardware_device_id));
  }
  
  // Set decoder provider hardware vendor ID
  void SetDecoderProviderOptionsHardwareVendorId(const char* provider, uint32_t hardware_vendor_id) {
    OgaCheckResult(OgaConfigSetDecoderProviderOptionsHardwareVendorId(ptr_, provider, hardware_vendor_id));
  }
  
  // Clear decoder provider hardware device type
  void ClearDecoderProviderOptionsHardwareDeviceType(const char* provider) {
    OgaCheckResult(OgaConfigClearDecoderProviderOptionsHardwareDeviceType(ptr_, provider));
  }
  
  // Clear decoder provider hardware device ID
  void ClearDecoderProviderOptionsHardwareDeviceId(const char* provider) {
    OgaCheckResult(OgaConfigClearDecoderProviderOptionsHardwareDeviceId(ptr_, provider));
  }
  
  // Clear decoder provider hardware vendor ID
  void ClearDecoderProviderOptionsHardwareVendorId(const char* provider) {
    OgaCheckResult(OgaConfigClearDecoderProviderOptionsHardwareVendorId(ptr_, provider));
  }
  
  // Overlay JSON on top of config
  void Overlay(const char* json) {
    OgaCheckResult(OgaConfigOverlay(ptr_, json));
  }
  
private:
  ::OgaConfig* ptr_;
};

} // namespace OgaPy
