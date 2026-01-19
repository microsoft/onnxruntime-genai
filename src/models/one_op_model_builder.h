// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>
#include "onnxruntime_api.h"

namespace Generators {

// Attribute types for ONNX operators
// These values correspond to the AttributeProto.type field in the ONNX protobuf specification
// Reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
enum class AttributeType {
  INT = 2,     // Single integer attribute
  FLOAT = 1,   // Single float attribute
  STRING = 3,  // String attribute
  INTS = 7,    // Integer array attribute
  FLOATS = 6,  // Float array attribute
  STRINGS = 8  // String array attribute
};

// Attribute value for a single operator parameter
struct AttributeValue {
  std::string name;
  AttributeType type;

  // Union-like storage for different types
  int64_t int_value{0};
  float float_value{0.0f};
  std::string string_value;
  std::vector<int64_t> ints_value;
  std::vector<float> floats_value;
  std::vector<std::string> strings_value;

  static AttributeValue Int(const std::string& name, int64_t value);
  static AttributeValue Float(const std::string& name, float value);
  static AttributeValue String(const std::string& name, const std::string& value);
  static AttributeValue Ints(const std::string& name, const std::vector<int64_t>& value);
  static AttributeValue Floats(const std::string& name, const std::vector<float>& value);
  static AttributeValue Strings(const std::string& name, const std::vector<std::string>& value);
};

// Configuration for a single input/output tensor
struct TensorConfig {
  std::string name;
  ONNXTensorElementDataType elem_type;
  std::vector<int64_t> shape;  // Use -1 for dynamic dimensions

  TensorConfig(const std::string& n, ONNXTensorElementDataType t, const std::vector<int64_t>& s)
      : name(n), elem_type(t), shape(s) {}
};

// Configuration for building a 1-op ONNX model
struct OneOpModelConfig {
  std::string op_type;  // e.g., "Cast", "TopK", "Argmax"
  std::vector<TensorConfig> inputs;
  std::vector<TensorConfig> outputs;
  std::vector<AttributeValue> attributes;

  // ONNX opset version for the generated model.
  // Default is 21 which is widely supported and matches the opset used elsewhere in the codebase.
  // Can be overridden if a specific opset is required, but ensure the ONNX Runtime build
  // supports it and the operator exists in that opset version.
  int opset_version{21};

  OneOpModelConfig(const std::string& op) : op_type(op) {}
};

// Builder class for creating 1-op ONNX models using the Model Editor API
class OneOpModelBuilder {
 public:
  // Build a complete ONNX model using the Model Editor API
  // Returns an OrtModel that can be used to create sessions
  // Caller is responsible for calling ReleaseOrtModel when done
  static OrtModel* Build(const OneOpModelConfig& config);

  // Convenience helper to create a Cast model
  static OrtModel* CreateCastModel(
      ONNXTensorElementDataType input_type,
      ONNXTensorElementDataType output_type);
};

}  // namespace Generators
