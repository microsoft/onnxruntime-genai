// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <sstream>
#include "models/onnx_kv_cache.h"

namespace Generators {
namespace {

std::string Varint(uint64_t value) {
  std::string result;
  do {
    result.push_back(static_cast<char>((value & 127) | (value > 127 ? 128 : 0)));
    value >>= 7;
  } while (value);
  return result;
}

std::string Field(uint32_t number, const std::string& value) {
  return Varint((uint64_t{number} << 3) | 2) + Varint(value.size()) + value;
}

std::string Node(const std::string& op, const std::vector<std::string>& inputs,
                 const std::vector<std::string>& outputs, const std::string& domain = {}) {
  std::string result;
  for (const auto& input : inputs) result += Field(1, input);
  for (const auto& output : outputs) result += Field(2, output);
  return result + Field(4, op) + Field(7, domain);
}

std::string Gqa(const std::string& lengths = "seqlens_k") {
  return Field(1, Node("GroupQueryAttention", {"q", "k", "v", "past.key", "past.value", lengths, "total"},
                       {"output", "present.key", "present.value"}, "com.microsoft"));
}

std::string Interface() {
  return Field(11, Field(1, "past.key")) + Field(11, Field(1, "past.value")) +
         Field(12, Field(1, "present.key")) + Field(12, Field(1, "present.value"));
}

SharedKvBindings Read(const std::string& model) {
  std::istringstream stream{model};
  return ReadOnnxSharedKvBindings(stream, model.size());
}

TEST(OnnxSharedKv, RecognizesDirectGqaAndSkipsInitializerPayloads) {
  const auto model = Field(7, Gqa() + Field(5, std::string(4096, '\xff')) + Interface());
  const SharedKvBindings expected{{"past.key", "present.key"}, {"past.value", "present.value"}};
  EXPECT_EQ(Read(model), expected);
}

TEST(OnnxSharedKv, RejectsExtentDependentConsumersAndNonGqaCaches) {
  for (const char* op : {"Shape", "Slice", "Concat"}) {
    EXPECT_TRUE(Read(Field(7, Gqa() + Field(1, Node(op, {"past.key"}, {"extra"})) + Interface())).empty());
    EXPECT_TRUE(Read(Field(7, Gqa() + Field(1, Node(op, {"present.value"}, {"extra"})) + Interface())).empty());
  }
  EXPECT_TRUE(Read(Field(7, Field(1, Node("Concat", {"past.key", "k"}, {"present.key"})) + Interface())).empty());
}

TEST(OnnxSharedKv, RequiresLengthPortsGraphBindingsAndUniqueCaches) {
  EXPECT_TRUE(Read(Field(7, Gqa("") + Interface())).empty());
  EXPECT_TRUE(Read(Field(7, Gqa())).empty());
  EXPECT_TRUE(Read(Field(7, Gqa() + Gqa() + Interface())).empty());
  EXPECT_TRUE(Read(Field(7, Gqa() + Interface()) + Field(25, "function")).empty());
}

TEST(OnnxSharedKv, RejectsHiddenSubgraphConsumersAndMalformedMetadata) {
  const auto subgraph = Field(5, Field(6, Field(1, Node("Shape", {"past.key"}, {"shape"}))));
  EXPECT_TRUE(Read(Field(7, Gqa() + Field(1, Node("If", {"cond"}, {"extra"}) + subgraph) + Interface())).empty());
  const auto model = Field(7, Gqa() + Interface());
  for (size_t size = 0; size < model.size(); ++size) EXPECT_TRUE(Read(model.substr(0, size)).empty());
  EXPECT_TRUE(Read(model + std::string(1, '\0')).empty());
}

}  // namespace
}  // namespace Generators
