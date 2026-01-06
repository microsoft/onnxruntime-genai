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
enum class AttributeType {
  INT = 2,
  FLOAT = 1,
  STRING = 3,
  INTS = 7,
  FLOATS = 6,
  STRINGS = 8
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
  int opset_version{17};  // Default to opset 17

  OneOpModelConfig(const std::string& op) : op_type(op) {}
};

// Builder class for creating 1-op ONNX models as protobuf bytes
class OneOpModelBuilder {
 public:
  // Build a complete ONNX model protobuf from the configuration
  static std::vector<uint8_t> Build(const OneOpModelConfig& config);

  // Helper to create a Cast model (backward compatibility)
  static std::vector<uint8_t> CreateCastModel(
      ONNXTensorElementDataType input_type,
      ONNXTensorElementDataType output_type);

 private:
  // Protobuf encoding helpers
  static void EncodeVarint(std::vector<uint8_t>& buffer, uint64_t value);
  static void EncodeKey(std::vector<uint8_t>& buffer, uint32_t field_number, uint32_t wire_type);
  static void EncodeString(std::vector<uint8_t>& buffer, uint32_t field_number, const std::string& value);
  static void EncodeInt64(std::vector<uint8_t>& buffer, uint32_t field_number, int64_t value);
  static void EncodeFloat(std::vector<uint8_t>& buffer, uint32_t field_number, float value);
  static void EncodeMessage(std::vector<uint8_t>& buffer, uint32_t field_number, const std::vector<uint8_t>& message);

  // Component builders
  static std::vector<uint8_t> BuildAttributeProto(const AttributeValue& attr);
  static std::vector<uint8_t> BuildNodeProto(const OneOpModelConfig& config);
  static std::vector<uint8_t> BuildTensorShapeProto(const std::vector<int64_t>& shape);
  static std::vector<uint8_t> BuildValueInfoProto(const TensorConfig& tensor);
  static std::vector<uint8_t> BuildOpsetImport(int version);
  static std::vector<uint8_t> BuildGraphProto(const OneOpModelConfig& config);
};

}  // namespace Generators
