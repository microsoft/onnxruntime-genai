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
struct ModelConfig;

// Input/Output tensor specification for graph model execution
struct TensorSpec {
  void* data;
  ONNXTensorElementDataType elem_type;
  std::vector<int64_t> shape;
  size_t size_in_bytes;

  TensorSpec(void* d, ONNXTensorElementDataType t, const std::vector<int64_t>& s, size_t sz)
      : data(d), elem_type(t), shape(s), size_in_bytes(sz) {}
};

// Parameters for executing a graph model
struct ExecutionParams {
  std::string execution_provider_name;  // e.g., "WebGPU", "CUDA", "DML"
  const OrtMemoryInfo* memory_info;     // Memory info for creating tensors
  std::vector<TensorSpec> inputs;
  std::vector<TensorSpec> outputs;

  // Optional: session configuration entries
  std::vector<const char*> session_config_keys;
  std::vector<const char*> session_config_values;

  ExecutionParams(const std::string& ep_name, const OrtMemoryInfo* mem_info)
      : execution_provider_name(ep_name), memory_info(mem_info) {}
};

// Namespace for graph execution utilities
// This is a common utility that all EPs can use
namespace GraphExecutor {

// Execute a graph model with the given configuration and parameters
// The session is automatically cached based on a hash of the model configuration
void Execute(
    const ModelConfig& model_config,
    const ExecutionParams& exec_params);

}  // namespace GraphExecutor

// Helper functions for common graph model operations

// Execute a Cast operation using a graph model
// This is a convenience wrapper around GraphExecutor::Execute
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
