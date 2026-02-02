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
  switch (attr.type) {
    case AttributeType::INT:
      return OrtOpAttr::Create(attr.name.c_str(), &attr.int_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT).release();
    case AttributeType::FLOAT:
      return OrtOpAttr::Create(attr.name.c_str(), &attr.float_value, 1, OrtOpAttrType::ORT_OP_ATTR_FLOAT).release();
    case AttributeType::STRING:
      return OrtOpAttr::Create(attr.name.c_str(), attr.string_value.c_str(),
                               static_cast<int>(attr.string_value.size()),
                               OrtOpAttrType::ORT_OP_ATTR_STRING)
          .release();
    case AttributeType::INTS:
      return OrtOpAttr::Create(attr.name.c_str(), attr.ints_value.data(),
                               static_cast<int>(attr.ints_value.size()),
                               OrtOpAttrType::ORT_OP_ATTR_INTS)
          .release();
    case AttributeType::FLOATS:
      return OrtOpAttr::Create(attr.name.c_str(), attr.floats_value.data(),
                               static_cast<int>(attr.floats_value.size()),
                               OrtOpAttrType::ORT_OP_ATTR_FLOATS)
          .release();
    case AttributeType::STRINGS: {
      std::vector<const char*> string_ptrs;
      string_ptrs.reserve(attr.strings_value.size());
      for (const auto& str : attr.strings_value) {
        string_ptrs.push_back(str.c_str());
      }
      return OrtOpAttr::Create(attr.name.c_str(), string_ptrs.data(),
                               static_cast<int>(string_ptrs.size()),
                               OrtOpAttrType::ORT_OP_ATTR_STRINGS)
          .release();
    }
  }
  return nullptr;  // Should never reach here
}

}  // anonymous namespace

namespace GraphBuilder {

// Build complete ONNX model using the Model Editor API
OrtModel* Build(const ModelConfig& config) {
  const auto& model_editor_api = Ort::GetModelEditorApi();

  // Create graph using RAII wrapper
  auto graph = OrtGraph::Create();

  // Create input ValueInfos
  std::vector<std::unique_ptr<OrtValueInfo>> graph_inputs;
  for (const auto& input : config.inputs) {
    auto tensor_info = OrtTensorTypeAndShapeInfo::Create();
    tensor_info->SetElementType(input.elem_type);
    tensor_info->SetDimensions(input.shape.data(), input.shape.size());

    auto value_info = OrtValueInfo::Create(input.name.c_str(), tensor_info.get());
    graph_inputs.push_back(std::move(value_info));
  }

  // Create output ValueInfos
  std::vector<std::unique_ptr<OrtValueInfo>> graph_outputs;
  for (const auto& output : config.outputs) {
    auto tensor_info = OrtTensorTypeAndShapeInfo::Create();
    tensor_info->SetElementType(output.elem_type);
    tensor_info->SetDimensions(output.shape.data(), output.shape.size());

    auto value_info = OrtValueInfo::Create(output.name.c_str(), tensor_info.get());
    graph_outputs.push_back(std::move(value_info));
  }

  // Set graph inputs and outputs (graph takes ownership of ValueInfos)
  std::vector<OrtValueInfo*> input_ptrs;
  input_ptrs.reserve(graph_inputs.size());
  for (auto& vi : graph_inputs) {
    input_ptrs.push_back(vi.get());
  }

  std::vector<OrtValueInfo*> output_ptrs;
  output_ptrs.reserve(graph_outputs.size());
  for (auto& vi : graph_outputs) {
    output_ptrs.push_back(vi.get());
  }

  Ort::ThrowOnError(model_editor_api.SetGraphInputs(graph.get(), input_ptrs.data(), input_ptrs.size()));
  Ort::ThrowOnError(model_editor_api.SetGraphOutputs(graph.get(), output_ptrs.data(), output_ptrs.size()));

  // Release ownership since graph took it
  for (auto& vi : graph_inputs) vi.release();
  for (auto& vi : graph_outputs) vi.release();

  // Create node attributes
  std::vector<OrtOpAttr*> node_attributes;
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

  // Create node using RAII wrapper
  auto node = OrtNode::Create(
      config.op_type.c_str(),
      "",  // empty domain = ONNX domain
      (config.op_type + "_node").c_str(),
      input_names.data(),
      input_names.size(),
      output_names.data(),
      output_names.size(),
      node_attributes.empty() ? nullptr : node_attributes.data(),
      node_attributes.size());

  // Add node to graph (graph takes ownership of node)
  Ort::ThrowOnError(model_editor_api.AddNodeToGraph(graph.get(), node.get()));
  node.release();  // graph now owns node

  // Create model with opset using RAII wrapper
  const char* domain_name = "";
  int opset = config.opset_version;
  auto model = OrtModel::Create(&domain_name, &opset, 1);

  // Add graph to model (model takes ownership of graph)
  Ort::ThrowOnError(model_editor_api.AddGraphToModel(model.get(), graph.get()));
  graph.release();  // model now owns graph

  // IMPORTANT: Release node attributes AFTER model is built.
  // CreateNode stores references to the attributes (doesn't copy them), so they must
  // stay alive until after AddGraphToModel is called. Now we explicitly release them.
  for (auto* attr : node_attributes) {
    delete attr;  // Uses OrtOpAttr::operator delete which calls ReleaseOpAttr
  }

  return model.release();  // Return ownership to caller
}

}  // namespace GraphBuilder

}  // namespace Generators
