// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>
#include <iostream>

#include "../src/flatbuffers.h"
#include "gtest/gtest.h"
#include "../src/span.h"
#include "../src/flatbuffers/lora_format_version.h"
#include "../src/flatbuffers/flatbuffers_utils.h"
#include "../src/flatbuffers/schema/genai_lora.fbs.h"


namespace Generators {
namespace lora_parameters {
namespace utils {

void LoadLoraParameter(const Tensor& tensor, std::unique_ptr<OrtValue>& tensor) {
  std::string name, doc_string;
  LoadStringFromLoraFormat(name, tensor.name());
  LoadStringFromLoraFormat(doc_string, tensor.doc_string());

  std::vector<int64_t> dims;
  dims.reserve(tensor.dims()->size());
  for (auto d : *tensor.dims()) {
    dims.push_back(d);
  }

  std::vector<uint8_t> data;
  data.reserve(tensor.raw_data()->size());
  // We may need to do a copy
  for (auto d : *tensor.raw_data()) {
    data.push_back(d);
  }
}

}  // namespace utils


namespace test {
TEST(LoraParameters, FlatbuffersTest) {

  // Create a flatbuffer
  flatbuffers::FlatBufferBuilder builder;

  const std::array<int64_t, 2> lora_param_shape = {4, 2};
  const std::array<float, 8> lora_param = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::span<const float> float_span = lora_param;
  auto byte_span = std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(float_span.data()), float_span.size_bytes());


  // We serialize 100 tensors
  constexpr size_t const tensor_count = 100;
  std::vector<flatbuffers::Offset<Tensor>> tensors;
  tensors.reserve(tensor_count);
  for (size_t i = 0; i < tensor_count; ++i) {
    std::string numeric_name = "lora_param_" + std::to_string(i);
    flatbuffers::Offset<Tensor> tensor;
    utils::SaveLoraParameter(builder, numeric_name, numeric_name,
                             TensorDataType::FLOAT, lora_param_shape, byte_span, tensor);
    tensors.push_back(tensor);
  }

  auto parameters = CreateParameters(builder, kLoraFormatVersion, builder.CreateVector(tensors));
  builder.Finish(parameters, ParametersIdentifier());

  std::string serialized;
  {
    std::stringstream stream;
    stream.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
    ASSERT_TRUE(stream);
    serialized = stream.str();
  }

  std::span<const uint8_t> serialized_span(reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size());
  utils::IsGenAiLoraFormatModelBytes(serialized_span.data(), serialized_span.size());

  flatbuffers::Verifier verifier(serialized_span.data(), serialized_span.size());
  ASSERT_TRUE(VerifyParametersBuffer(verifier));

  const auto* fbs_parameters = GetParameters(serialized_span.data());
  ASSERT_NE(nullptr, fbs_parameters) << "Parameters are null";

  ASSERT_TRUE(IsLoraFormatVersionSupported(fbs_parameters->version())) << "Format version mismatch";

}

}   // namespace test
}  // namespace lora_parameters
}  // namespace Generators
