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

// Configuration for building an ONNX model
struct ModelConfig {
  std::string op_type;  // e.g., "Cast", "TopK", "Argmax"
  std::vector<TensorConfig> inputs;
  std::vector<TensorConfig> outputs;
  std::vector<AttributeValue> attributes;

  // ONNX opset version for the generated model. Default to 21.
  const int opset_version{21};

  ModelConfig(const std::string& op) : op_type(op) {}
};

// Namespace for graph building utilities using the Model Editor API
namespace GraphBuilder {

// Build a complete ONNX model using the Model Editor API
// Returns an OrtModel that can be used to create sessions
std::unique_ptr<OrtModel> Build(const ModelConfig& config);

}  // namespace GraphBuilder

}  // namespace Generators
