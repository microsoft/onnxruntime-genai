// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "one_op_model_builder.h"
#include <cstring>

namespace Generators {

// AttributeValue factory methods
AttributeValue AttributeValue::Int(const std::string& name, int64_t value) {
  AttributeValue attr;
  attr.name = name;
  attr.type = AttributeType::INT;
  attr.int_value = value;
  return attr;
}

AttributeValue AttributeValue::Float(const std::string& name, float value) {
  AttributeValue attr;
  attr.name = name;
  attr.type = AttributeType::FLOAT;
  attr.float_value = value;
  return attr;
}

AttributeValue AttributeValue::String(const std::string& name, const std::string& value) {
  AttributeValue attr;
  attr.name = name;
  attr.type = AttributeType::STRING;
  attr.string_value = value;
  return attr;
}

AttributeValue AttributeValue::Ints(const std::string& name, const std::vector<int64_t>& value) {
  AttributeValue attr;
  attr.name = name;
  attr.type = AttributeType::INTS;
  attr.ints_value = value;
  return attr;
}

AttributeValue AttributeValue::Floats(const std::string& name, const std::vector<float>& value) {
  AttributeValue attr;
  attr.name = name;
  attr.type = AttributeType::FLOATS;
  attr.floats_value = value;
  return attr;
}

AttributeValue AttributeValue::Strings(const std::string& name, const std::vector<std::string>& value) {
  AttributeValue attr;
  attr.name = name;
  attr.type = AttributeType::STRINGS;
  attr.strings_value = value;
  return attr;
}

// Protobuf varint encoding
void OneOpModelBuilder::EncodeVarint(std::vector<uint8_t>& buffer, uint64_t value) {
  while (value >= 0x80) {
    buffer.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  buffer.push_back(static_cast<uint8_t>(value & 0x7F));
}

// Encode field key (field_number << 3 | wire_type)
void OneOpModelBuilder::EncodeKey(std::vector<uint8_t>& buffer, uint32_t field_number, uint32_t wire_type) {
  EncodeVarint(buffer, (field_number << 3) | wire_type);
}

// Encode length-delimited field (wire_type = 2)
void OneOpModelBuilder::EncodeString(std::vector<uint8_t>& buffer, uint32_t field_number, const std::string& value) {
  EncodeKey(buffer, field_number, 2);
  EncodeVarint(buffer, value.size());
  buffer.insert(buffer.end(), value.begin(), value.end());
}

// Encode varint field (wire_type = 0)
void OneOpModelBuilder::EncodeInt64(std::vector<uint8_t>& buffer, uint32_t field_number, int64_t value) {
  EncodeKey(buffer, field_number, 0);
  EncodeVarint(buffer, static_cast<uint64_t>(value));
}

// Encode float field (wire_type = 5, 32-bit fixed)
void OneOpModelBuilder::EncodeFloat(std::vector<uint8_t>& buffer, uint32_t field_number, float value) {
  EncodeKey(buffer, field_number, 5);
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(float));
  buffer.push_back(static_cast<uint8_t>(bits & 0xFF));
  buffer.push_back(static_cast<uint8_t>((bits >> 8) & 0xFF));
  buffer.push_back(static_cast<uint8_t>((bits >> 16) & 0xFF));
  buffer.push_back(static_cast<uint8_t>((bits >> 24) & 0xFF));
}

// Encode embedded message (wire_type = 2)
void OneOpModelBuilder::EncodeMessage(std::vector<uint8_t>& buffer, uint32_t field_number, const std::vector<uint8_t>& message) {
  EncodeKey(buffer, field_number, 2);
  EncodeVarint(buffer, message.size());
  buffer.insert(buffer.end(), message.begin(), message.end());
}

// Build AttributeProto
std::vector<uint8_t> OneOpModelBuilder::BuildAttributeProto(const AttributeValue& attr) {
  std::vector<uint8_t> result;

  // Field 1: name (string)
  EncodeString(result, 1, attr.name);

  // Field 20: type (AttributeType enum)
  EncodeInt64(result, 20, static_cast<int64_t>(attr.type));

  // Value fields based on type
  switch (attr.type) {
    case AttributeType::INT:
      EncodeInt64(result, 3, attr.int_value);
      break;
    case AttributeType::FLOAT:
      EncodeFloat(result, 2, attr.float_value);
      break;
    case AttributeType::STRING:
      EncodeString(result, 4, attr.string_value);
      break;
    case AttributeType::INTS:
      for (auto val : attr.ints_value) {
        EncodeInt64(result, 7, val);
      }
      break;
    case AttributeType::FLOATS:
      for (auto val : attr.floats_value) {
        EncodeFloat(result, 6, val);
      }
      break;
    case AttributeType::STRINGS:
      for (const auto& val : attr.strings_value) {
        EncodeString(result, 8, val);
      }
      break;
  }

  return result;
}

// Build NodeProto
std::vector<uint8_t> OneOpModelBuilder::BuildNodeProto(const OneOpModelConfig& config) {
  std::vector<uint8_t> result;

  // Field 1: input (repeated string)
  for (const auto& input : config.inputs) {
    EncodeString(result, 1, input.name);
  }

  // Field 2: output (repeated string)
  for (const auto& output : config.outputs) {
    EncodeString(result, 2, output.name);
  }

  // Field 4: op_type (string)
  EncodeString(result, 4, config.op_type);

  // Field 5: attribute (repeated AttributeProto)
  for (const auto& attr : config.attributes) {
    auto attr_proto = BuildAttributeProto(attr);
    EncodeMessage(result, 5, attr_proto);
  }

  return result;
}

// Build TensorShapeProto
std::vector<uint8_t> OneOpModelBuilder::BuildTensorShapeProto(const std::vector<int64_t>& shape) {
  std::vector<uint8_t> result;

  for (int64_t dim : shape) {
    // TensorShapeProto::Dimension
    std::vector<uint8_t> dim_proto;
    EncodeInt64(dim_proto, 1, dim);  // dim_value

    // Add dimension to shape
    EncodeMessage(result, 1, dim_proto);  // Field 1: dim (repeated)
  }

  return result;
}

// Build ValueInfoProto
std::vector<uint8_t> OneOpModelBuilder::BuildValueInfoProto(const TensorConfig& tensor) {
  std::vector<uint8_t> result;

  // Build TensorShapeProto
  auto shape_proto = BuildTensorShapeProto(tensor.shape);

  // Build TypeProto::Tensor
  std::vector<uint8_t> tensor_type_proto;
  EncodeInt64(tensor_type_proto, 1, static_cast<int64_t>(tensor.elem_type));  // elem_type
  EncodeMessage(tensor_type_proto, 2, shape_proto);                           // shape

  // Build TypeProto
  std::vector<uint8_t> type_proto;
  EncodeMessage(type_proto, 1, tensor_type_proto);  // tensor_type

  // Build ValueInfoProto
  EncodeString(result, 1, tensor.name);  // name
  EncodeMessage(result, 2, type_proto);  // type

  return result;
}

// Build OperatorSetIdProto
std::vector<uint8_t> OneOpModelBuilder::BuildOpsetImport(int version) {
  std::vector<uint8_t> result;
  EncodeString(result, 1, "");      // domain (empty = default domain)
  EncodeInt64(result, 2, version);  // version
  return result;
}

// Build GraphProto
std::vector<uint8_t> OneOpModelBuilder::BuildGraphProto(const OneOpModelConfig& config) {
  std::vector<uint8_t> result;

  // Field 1: node (repeated NodeProto)
  auto node_proto = BuildNodeProto(config);
  EncodeMessage(result, 1, node_proto);

  // Field 2: name (string)
  EncodeString(result, 2, config.op_type + "_model");

  // Field 11: input (repeated ValueInfoProto)
  for (const auto& input : config.inputs) {
    auto value_info = BuildValueInfoProto(input);
    EncodeMessage(result, 11, value_info);
  }

  // Field 12: output (repeated ValueInfoProto)
  for (const auto& output : config.outputs) {
    auto value_info = BuildValueInfoProto(output);
    EncodeMessage(result, 12, value_info);
  }

  return result;
}

// Build complete ModelProto
std::vector<uint8_t> OneOpModelBuilder::Build(const OneOpModelConfig& config) {
  std::vector<uint8_t> result;

  // Build components
  auto opset_import = BuildOpsetImport(config.opset_version);
  auto graph_proto = BuildGraphProto(config);

  // ModelProto
  EncodeInt64(result, 1, 8);               // ir_version = 8
  EncodeMessage(result, 7, graph_proto);   // graph
  EncodeMessage(result, 8, opset_import);  // opset_import

  return result;
}

// Helper to create a Cast model (backward compatibility with existing code)
std::vector<uint8_t> OneOpModelBuilder::CreateCastModel(
    ONNXTensorElementDataType input_type,
    ONNXTensorElementDataType output_type) {
  OneOpModelConfig config("Cast");

  // Add input tensor with dynamic shape
  config.inputs.push_back(TensorConfig("input", input_type, {-1}));

  // Add output tensor with dynamic shape
  config.outputs.push_back(TensorConfig("output", output_type, {-1}));

  // Add "to" attribute
  config.attributes.push_back(AttributeValue::Int("to", static_cast<int64_t>(output_type)));

  return Build(config);
}

}  // namespace Generators
