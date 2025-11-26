// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cast_model_builder.h"
#include <string>
#include <cstring>

namespace Generators {
namespace WebGPU {

// Protobuf varint encoding
static void EncodeVarint(std::vector<uint8_t>& buffer, uint64_t value) {
  while (value >= 0x80) {
    buffer.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  buffer.push_back(static_cast<uint8_t>(value & 0x7F));
}

// Encode field key (field_number << 3 | wire_type)
static void EncodeKey(std::vector<uint8_t>& buffer, uint32_t field_number, uint32_t wire_type) {
  EncodeVarint(buffer, (field_number << 3) | wire_type);
}

// Encode length-delimited field (wire_type = 2)
static void EncodeString(std::vector<uint8_t>& buffer, uint32_t field_number, const std::string& value) {
  EncodeKey(buffer, field_number, 2);
  EncodeVarint(buffer, value.size());
  buffer.insert(buffer.end(), value.begin(), value.end());
}

// Encode varint field (wire_type = 0)
static void EncodeInt64(std::vector<uint8_t>& buffer, uint32_t field_number, int64_t value) {
  EncodeKey(buffer, field_number, 0);
  EncodeVarint(buffer, value);
}

// Encode embedded message (wire_type = 2)
static void EncodeMessage(std::vector<uint8_t>& buffer, uint32_t field_number, const std::vector<uint8_t>& message) {
  EncodeKey(buffer, field_number, 2);
  EncodeVarint(buffer, message.size());
  buffer.insert(buffer.end(), message.begin(), message.end());
}

std::vector<uint8_t> CreateCastModelBytes(ONNXTensorElementDataType input_type, ONNXTensorElementDataType output_type) {
  // Build ONNX ModelProto structure manually
  // Reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto

  // AttributeProto for "to" attribute
  // Field numbers: name=1, i=3, type=20
  std::vector<uint8_t> attr_proto;
  EncodeString(attr_proto, 1, "to");                              // name
  EncodeInt64(attr_proto, 3, static_cast<int64_t>(output_type));  // i (integer value) - field 3, not 2!
  EncodeInt64(attr_proto, 20, 2);                                 // type = INT (2)

  // NodeProto for Cast operator
  std::vector<uint8_t> node_proto;
  EncodeString(node_proto, 1, "input");      // input
  EncodeString(node_proto, 2, "output");     // output
  EncodeString(node_proto, 4, "Cast");       // op_type
  EncodeMessage(node_proto, 5, attr_proto);  // attribute

  // TensorShapeProto::Dimension (dynamic dimension)
  std::vector<uint8_t> dim_proto;
  EncodeInt64(dim_proto, 1, -1);  // dim_value = -1 (dynamic)

  // TensorShapeProto
  std::vector<uint8_t> shape_proto;
  EncodeMessage(shape_proto, 1, dim_proto);  // dim

  // TypeProto::Tensor for input
  std::vector<uint8_t> input_tensor_type;
  EncodeInt64(input_tensor_type, 1, static_cast<int64_t>(input_type));  // elem_type
  EncodeMessage(input_tensor_type, 2, shape_proto);                     // shape

  // TypeProto for input
  std::vector<uint8_t> input_type_proto;
  EncodeMessage(input_type_proto, 1, input_tensor_type);  // tensor_type

  // ValueInfoProto for input
  std::vector<uint8_t> input_value_info;
  EncodeString(input_value_info, 1, "input");            // name
  EncodeMessage(input_value_info, 2, input_type_proto);  // type

  // TypeProto::Tensor for output
  std::vector<uint8_t> output_tensor_type;
  EncodeInt64(output_tensor_type, 1, static_cast<int64_t>(output_type));  // elem_type
  EncodeMessage(output_tensor_type, 2, shape_proto);                      // shape

  // TypeProto for output
  std::vector<uint8_t> output_type_proto;
  EncodeMessage(output_type_proto, 1, output_tensor_type);  // tensor_type

  // ValueInfoProto for output
  std::vector<uint8_t> output_value_info;
  EncodeString(output_value_info, 1, "output");            // name
  EncodeMessage(output_value_info, 2, output_type_proto);  // type

  // OperatorSetIdProto
  std::vector<uint8_t> opset_import;
  EncodeString(opset_import, 1, "");  // domain (empty = default domain)
  EncodeInt64(opset_import, 2, 17);   // version = 17

  // GraphProto
  std::vector<uint8_t> graph_proto;
  EncodeMessage(graph_proto, 1, node_proto);          // node
  EncodeString(graph_proto, 2, "cast_model");         // name
  EncodeMessage(graph_proto, 11, input_value_info);   // input
  EncodeMessage(graph_proto, 12, output_value_info);  // output

  // ModelProto
  std::vector<uint8_t> model_proto;
  EncodeInt64(model_proto, 1, 8);               // ir_version = 8
  EncodeMessage(model_proto, 7, graph_proto);   // graph
  EncodeMessage(model_proto, 8, opset_import);  // opset_import

  return model_proto;
}

}  // namespace WebGPU
}  // namespace Generators
