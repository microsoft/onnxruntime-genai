// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <functional>
#include "onnxruntime_api.h"

namespace Generators {

// Forward declarations
struct OneOpModelConfig;

// Input/Output tensor specification for 1-op model execution
struct OneOpTensorSpec {
  void* data;
  ONNXTensorElementDataType elem_type;
  std::vector<int64_t> shape;
  size_t size_in_bytes;

  OneOpTensorSpec(void* d, ONNXTensorElementDataType t, const std::vector<int64_t>& s, size_t sz)
      : data(d), elem_type(t), shape(s), size_in_bytes(sz) {}
};

// Parameters for executing a 1-op model
struct OneOpExecutionParams {
  std::string execution_provider_name;  // e.g., "WebGPU", "CUDA", "DML"
  const OrtMemoryInfo* memory_info;     // Memory info for creating tensors
  std::vector<OneOpTensorSpec> inputs;
  std::vector<OneOpTensorSpec> outputs;

  // Optional: session configuration entries
  std::vector<const char*> session_config_keys;
  std::vector<const char*> session_config_values;

  OneOpExecutionParams(const std::string& ep_name, const OrtMemoryInfo* mem_info)
      : execution_provider_name(ep_name), memory_info(mem_info) {}
};

// Manages creation, caching, and execution of 1-op ONNX models
// This is a common utility that all EPs can use
class OneOpModelExecutor {
 public:
  // Execute a 1-op model with the given configuration and parameters
  // The session is automatically cached based on a hash of the model configuration
  static void Execute(
      const OneOpModelConfig& model_config,
      const OneOpExecutionParams& exec_params);

  // Clear all cached sessions (useful for cleanup)
  static void ClearCache();

 private:
  // Generate a cache key from the model configuration and EP name
  static uint64_t GenerateCacheKey(
      const OneOpModelConfig& config,
      const std::string& ep_name,
      const std::vector<const char*>& session_config_keys,
      const std::vector<const char*>& session_config_values);

  // Create a new session for the given model and EP
  static std::unique_ptr<OrtSession> CreateSession(
      OrtModel* model,
      const std::string& ep_name,
      const std::vector<const char*>& session_config_keys,
      const std::vector<const char*>& session_config_values);

  // Get or create a cached session
  static OrtSession* GetOrCreateSession(
      const OneOpModelConfig& config,
      const std::string& ep_name,
      const std::vector<const char*>& session_config_keys,
      const std::vector<const char*>& session_config_values);
};

// Helper functions for common 1-op model operations

// Execute a Cast operation using a 1-op model
// This is a convenience wrapper around OneOpModelExecutor::Execute
void ExecuteCastOp(
    void* input_data,
    void* output_data,
    ONNXTensorElementDataType input_type,
    ONNXTensorElementDataType output_type,
    size_t element_count,
    const std::string& execution_provider_name,
    const OrtMemoryInfo* memory_info,
    const std::vector<const char*>& session_config_keys = {},
    const std::vector<const char*>& session_config_values = {});

}  // namespace Generators
