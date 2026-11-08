// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnx_kv_cache.h"
#include "onnx_proto_reader.h"

#include <fstream>
#include <unordered_map>
#include <unordered_set>

namespace Generators {
namespace {

struct Node {
  std::string op, domain;
  std::vector<std::string> inputs, outputs;
};

bool ReadNode(std::istream& stream, const ProtoFieldReader::Field& field, Node& node) {
  ProtoFieldReader reader(stream, field.payload_begin, field.payload_begin + field.payload_size);
  ProtoFieldReader::Field entry;
  while (reader.Next(entry)) {
    if (entry.number == 1 || entry.number == 2 || entry.number == 4 || entry.number == 7) {
      std::string value;
      if (!entry.length_delimited || entry.payload_size > (uint64_t{1} << 20) || !reader.ReadPayload(entry, value)) return false;
      if (entry.number == 1) node.inputs.push_back(std::move(value));
      if (entry.number == 2) node.outputs.push_back(std::move(value));
      if (entry.number == 4) node.op = std::move(value);
      if (entry.number == 7) node.domain = std::move(value);
    } else if (entry.number == 5) {  // NodeProto.attribute
      if (!entry.length_delimited) return false;
      ProtoFieldReader attribute(stream, entry.payload_begin, entry.payload_begin + entry.payload_size);
      ProtoFieldReader::Field value;
      while (attribute.Next(value)) {
        // Subgraphs can capture outer-scope KV without listing it as a node input.
        if (value.number == 6 || value.number == 11) return false;
      }
      if (attribute.Failed()) return false;
    }
  }
  return !reader.Failed() && !node.op.empty();
}

}  // namespace

SharedKvBindings ReadOnnxSharedKvBindings(std::istream& stream, uint64_t size) {
  ProtoFieldReader model(stream, 0, size);
  ProtoFieldReader::Field field;
  SharedKvBindings bindings;
  std::unordered_map<std::string, size_t> uses;
  std::unordered_set<std::string> pasts, presents;
  std::unordered_set<std::string> inputs, outputs;
  bool has_graph = false;
  while (model.Next(field)) {
    if (field.number == 25) return {};  // Model-local functions may replace operator semantics.
    if (field.number != 7) continue;    // ModelProto.graph
    if (!field.length_delimited || has_graph) return {};
    has_graph = true;
    ProtoFieldReader graph(stream, field.payload_begin, field.payload_begin + field.payload_size);
    ProtoFieldReader::Field entry;
    while (graph.Next(entry)) {
      if (entry.number == 11 || entry.number == 12) {  // GraphProto input/output ValueInfoProto
        if (!entry.length_delimited) return {};
        ProtoFieldReader value_info(stream, entry.payload_begin, entry.payload_begin + entry.payload_size);
        ProtoFieldReader::Field value;
        std::string name;
        while (value_info.Next(value)) {
          if (value.number == 1 && (!value.length_delimited || value.payload_size > (uint64_t{1} << 20) ||
                                    !value_info.ReadPayload(value, name))) return {};
        }
        if (value_info.Failed() || name.empty()) return {};
        (entry.number == 11 ? inputs : outputs).insert(std::move(name));
        continue;
      }
      if (entry.number != 1) continue;  // GraphProto.node; skip weights/value info
      if (!entry.length_delimited) return {};
      Node node;
      if (!ReadNode(stream, entry, node)) return {};
      for (const auto& input : node.inputs) {
        if (!input.empty()) ++uses[input];
      }
      if (node.domain != "com.microsoft" || node.op != "GroupQueryAttention") continue;
      if (node.inputs.size() < 7 || node.outputs.size() < 3 ||
          node.inputs[5].empty() || node.inputs[6].empty()) return {};
      for (size_t i = 0; i < 2; ++i) {
        const auto& past = node.inputs[3 + i];
        const auto& present = node.outputs[1 + i];
        if (past.empty() || present.empty() || past == present ||
            !pasts.insert(past).second || !presents.insert(present).second) return {};
        bindings.emplace_back(past, present);
      }
    }
    if (graph.Failed()) return {};
  }
  if (model.Failed() || !has_graph) return {};
  for (const auto& [past, present] : bindings) {
    // Extra Shape/Slice/Concat consumers can depend on the logical tensor extent.
    if (uses[past] != 1 || uses[present] != 0 || !inputs.contains(past) || !outputs.contains(present)) return {};
  }
  return bindings;
}

SharedKvBindings ReadOnnxSharedKvBindings(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) return {};
  const auto size = file.tellg();
  if (size <= 0) return {};
  return ReadOnnxSharedKvBindings(file, static_cast<uint64_t>(size));
}

}  // namespace Generators
