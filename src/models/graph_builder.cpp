// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "graph_builder.h"
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

namespace {

// Helper to create OrtOpAttr from AttributeValue using Model Editor API
OrtOpAttr* CreateOpAttr(const AttributeValue& attr) {
  OrtOpAttr* op_attr = nullptr;

  switch (attr.type) {
    case AttributeType::INT:
      Ort::ThrowOnError(Ort::api->CreateOpAttr(attr.name.c_str(), &attr.int_value, 1,
                                               OrtOpAttrType::ORT_OP_ATTR_INT, &op_attr));
      break;
    case AttributeType::FLOAT:
      Ort::ThrowOnError(Ort::api->CreateOpAttr(attr.name.c_str(), &attr.float_value, 1,
                                               OrtOpAttrType::ORT_OP_ATTR_FLOAT, &op_attr));
      break;
    case AttributeType::STRING:
      Ort::ThrowOnError(Ort::api->CreateOpAttr(attr.name.c_str(), attr.string_value.c_str(),
                                               static_cast<int>(attr.string_value.size()),
                                               OrtOpAttrType::ORT_OP_ATTR_STRING, &op_attr));
      break;
    case AttributeType::INTS:
      Ort::ThrowOnError(Ort::api->CreateOpAttr(attr.name.c_str(), attr.ints_value.data(),
                                               static_cast<int>(attr.ints_value.size()),
                                               OrtOpAttrType::ORT_OP_ATTR_INTS, &op_attr));
      break;
    case AttributeType::FLOATS:
      Ort::ThrowOnError(Ort::api->CreateOpAttr(attr.name.c_str(), attr.floats_value.data(),
                                               static_cast<int>(attr.floats_value.size()),
                                               OrtOpAttrType::ORT_OP_ATTR_FLOATS, &op_attr));
      break;
    case AttributeType::STRINGS: {
      std::vector<const char*> string_ptrs;
      string_ptrs.reserve(attr.strings_value.size());
      for (const auto& str : attr.strings_value) {
        string_ptrs.push_back(str.c_str());
      }
      Ort::ThrowOnError(Ort::api->CreateOpAttr(attr.name.c_str(), string_ptrs.data(),
                                               static_cast<int>(string_ptrs.size()),
                                               OrtOpAttrType::ORT_OP_ATTR_STRINGS, &op_attr));
      break;
    }
  }

  return op_attr;
}

}  // anonymous namespace

namespace GraphBuilder {

// Build complete ONNX model using the Model Editor API
OrtModel* Build(const ModelConfig& config) {
  const auto& model_editor_api = Ort::GetModelEditorApi();

  OrtGraph* graph = nullptr;
  OrtModel* model = nullptr;
  std::vector<OrtOpAttr*> node_attributes;

  try {
    // Create graph
    Ort::ThrowOnError(model_editor_api.CreateGraph(&graph));

    // Create input ValueInfos
    std::vector<OrtValueInfo*> graph_inputs;
    for (const auto& input : config.inputs) {
      OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
      Ort::ThrowOnError(Ort::api->CreateTensorTypeAndShapeInfo(&tensor_info));
      Ort::ThrowOnError(Ort::api->SetTensorElementType(tensor_info, input.elem_type));
      Ort::ThrowOnError(Ort::api->SetDimensions(tensor_info, input.shape.data(), input.shape.size()));

      OrtTypeInfo* type_info = nullptr;
      Ort::ThrowOnError(model_editor_api.CreateTensorTypeInfo(tensor_info, &type_info));
      Ort::api->ReleaseTensorTypeAndShapeInfo(tensor_info);

      OrtValueInfo* value_info = nullptr;
      Ort::ThrowOnError(model_editor_api.CreateValueInfo(input.name.c_str(), type_info, &value_info));
      Ort::api->ReleaseTypeInfo(type_info);

      graph_inputs.push_back(value_info);
    }

    // Create output ValueInfos
    std::vector<OrtValueInfo*> graph_outputs;
    for (const auto& output : config.outputs) {
      OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
      Ort::ThrowOnError(Ort::api->CreateTensorTypeAndShapeInfo(&tensor_info));
      Ort::ThrowOnError(Ort::api->SetTensorElementType(tensor_info, output.elem_type));
      Ort::ThrowOnError(Ort::api->SetDimensions(tensor_info, output.shape.data(), output.shape.size()));

      OrtTypeInfo* type_info = nullptr;
      Ort::ThrowOnError(model_editor_api.CreateTensorTypeInfo(tensor_info, &type_info));
      Ort::api->ReleaseTensorTypeAndShapeInfo(tensor_info);

      OrtValueInfo* value_info = nullptr;
      Ort::ThrowOnError(model_editor_api.CreateValueInfo(output.name.c_str(), type_info, &value_info));
      Ort::api->ReleaseTypeInfo(type_info);

      graph_outputs.push_back(value_info);
    }

    // Set graph inputs and outputs (graph takes ownership of ValueInfos)
    Ort::ThrowOnError(model_editor_api.SetGraphInputs(graph, graph_inputs.data(), graph_inputs.size()));
    Ort::ThrowOnError(model_editor_api.SetGraphOutputs(graph, graph_outputs.data(), graph_outputs.size()));

    // Create node attributes
    for (const auto& attr : config.attributes) {
      node_attributes.push_back(CreateOpAttr(attr));
    }

    // Create input/output name vectors
    std::vector<const char*> input_names;
    for (const auto& input : config.inputs) {
      input_names.push_back(input.name.c_str());
    }

    std::vector<const char*> output_names;
    for (const auto& output : config.outputs) {
      output_names.push_back(output.name.c_str());
    }

    // Create node
    OrtNode* node = nullptr;
    Ort::ThrowOnError(model_editor_api.CreateNode(
        config.op_type.c_str(),
        "",  // empty domain = ONNX domain
        (config.op_type + "_node").c_str(),
        input_names.data(),
        input_names.size(),
        output_names.data(),
        output_names.size(),
        node_attributes.empty() ? nullptr : node_attributes.data(),
        node_attributes.size(),
        &node));

    // Add node to graph (graph takes ownership of node)
    Ort::ThrowOnError(model_editor_api.AddNodeToGraph(graph, node));
    // Release node attributes - CreateNode made its own copy
    for (auto* attr : node_attributes) {
      Ort::api->ReleaseOpAttr(attr);
    }
    node_attributes.clear();
    // Create model with opset
    const char* domain_name = "";
    Ort::ThrowOnError(model_editor_api.CreateModel(&domain_name, &config.opset_version, 1, &model));

    // Add graph to model (model takes ownership of graph)
    Ort::ThrowOnError(model_editor_api.AddGraphToModel(model, graph));
    graph = nullptr;  // model now owns graph

    return model;

  } catch (...) {
    // Clean up on error
    for (auto* attr : node_attributes) {
      Ort::api->ReleaseOpAttr(attr);
    }
    if (graph != nullptr) {
      Ort::api->ReleaseGraph(graph);
    }
    if (model != nullptr) {
      Ort::api->ReleaseModel(model);
    }
    throw;
  }
}

// Helper to create a Cast model
OrtModel* CreateCastModel(
    ONNXTensorElementDataType input_type,
    ONNXTensorElementDataType output_type) {
  ModelConfig config("Cast");

  // Add input tensor with dynamic shape
  config.inputs.push_back(TensorConfig("input", input_type, {-1}));

  // Add output tensor with dynamic shape
  config.outputs.push_back(TensorConfig("output", output_type, {-1}));

  // Add "to" attribute
  config.attributes.push_back(AttributeValue::Int("to", static_cast<int64_t>(output_type)));

  return Build(config);
}

}  // namespace GraphBuilder

}  // namespace Generators
