// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <fstream>
#include <iostream>

#include "../src/flatbuffers.h"
#include "gtest/gtest.h"
#include "../src/span.h"
#include "../src/flatbuffers/lora_format_version.h"
#include "../src/flatbuffers/flatbuffers_utils.h"
#include "../src/flatbuffers/schema/genai_lora.fbs.h"

#include "../src/models/onnxruntime_api.h"

namespace Generators {
namespace lora_parameters {
namespace test {
TEST(LoraParameters, FlatbuffersTest) {
  // Create a flatbuffer
  flatbuffers::FlatBufferBuilder builder;

  const std::array<int64_t, 2> lora_param_shape = {4, 2};
  const std::array<float, 8> lora_param = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::span<const float> float_span = lora_param;
  auto byte_span =
      std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(float_span.data()), float_span.size_bytes());

  // We serialize 100 tensors
  constexpr size_t const tensor_count = 100;
  std::vector<flatbuffers::Offset<Param>> params;
  params.reserve(tensor_count);
  for (size_t i = 0; i < tensor_count; ++i) {
    std::string numeric_name = "lora_param_" + std::to_string(i);
    flatbuffers::Offset<Param> fbs_tensor;
    utils::SaveLoraParameter(builder, numeric_name, TensorDataType::FLOAT, lora_param_shape, byte_span, fbs_tensor);
    params.push_back(fbs_tensor);
  }

  auto parameters = CreateParameters(builder, kLoraFormatVersion, builder.CreateVector(params));
  builder.Finish(parameters, ParametersIdentifier());

  std::string serialized;
  {
    std::stringstream stream;
    stream.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
    ASSERT_TRUE(stream);
    serialized = stream.str();
  }

  // Verify the buffer first
  std::span<const uint8_t> serialized_span(reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size());
  utils::IsGenAiLoraFormatModelBytes(serialized_span.data(), serialized_span.size());

  flatbuffers::Verifier verifier(serialized_span.data(), serialized_span.size());
  ASSERT_TRUE(VerifyParametersBuffer(verifier));

  const auto* fbs_parameters = GetParameters(serialized_span.data());
  ASSERT_NE(nullptr, fbs_parameters) << "Parameters are null";

  ASSERT_TRUE(IsLoraFormatVersionSupported(fbs_parameters->version())) << "Format version mismatch";

  // Run some checks
  for (const auto* fbs_tensor : *fbs_parameters->parameters()) {
    ASSERT_NE(nullptr, fbs_tensor) << "Tensor is null";

    std::string name;
    utils::LoadStringFromLoraFormat(name, fbs_tensor->name());
    ASSERT_FALSE(name.empty()) << "Name is empty";

    const auto data_type = fbs_tensor->data_type();
    ASSERT_EQ(TensorDataType::FLOAT, data_type) << "Data type mismatch";

    std::vector<int64_t> dims;
    dims.reserve(fbs_tensor->dims()->size());
    for (auto d : *fbs_tensor->dims()) {
      dims.push_back(d);
    }

    ASSERT_EQ(lora_param_shape.size(), dims.size()) << "Shape size mismatch";
    for (size_t i = 0; i < lora_param_shape.size(); ++i) {
      ASSERT_EQ(lora_param_shape[i], dims[i]) << "Shape mismatch";
    }

    const auto* raw_data = fbs_tensor->raw_data();
    ASSERT_NE(nullptr, raw_data) << "Raw data is null";
    ASSERT_EQ(byte_span.size_bytes(), raw_data->size()) << "Raw data size mismatch";
    for (size_t i = 0; i < byte_span.size_bytes(); ++i) {
      ASSERT_EQ(byte_span[i], raw_data->data()[i]) << "Raw data mismatch";
    }
  }

  // Now invoke the utils
  for (const auto* fbs_tensor : *fbs_parameters->parameters()) {
    const auto& [name, ort_value] = utils::CreateOrtValueOverFlatBufferLoraParameter(*fbs_tensor);
    ASSERT_FALSE(name.empty()) << "Name is empty";
    ASSERT_NE(nullptr, ort_value) << "OrtValue is null";

    const auto type_and_shape = ort_value->GetTensorTypeAndShapeInfo();
    auto shape = type_and_shape->GetShape();
    ASSERT_EQ(lora_param_shape.size(), shape.size()) << "Shape size mismatch";
    for (size_t i = 0; i < lora_param_shape.size(); ++i) {
      ASSERT_EQ(lora_param_shape[i], shape[i]) << "Shape mismatch";
    }

    ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_and_shape->GetElementType());
    ASSERT_EQ(lora_param.size(), type_and_shape->GetElementCount());
    const auto* data = ort_value->GetTensorMutableData<float>();
    for (size_t i = 0, lim = lora_param.size(); i < lim; ++i) {
      ASSERT_EQ(lora_param[i], data[i]) << "Data mismatch";
    }
  }
}

TEST(LoraParameters, LoadPythonGeneratedFile) {
  const std::string file_path = MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32-lora/two_lora_params.fb";
  std::ifstream file(file_path, std::ios::binary);
  ASSERT_TRUE(file) << "Failed to open file: " << file_path;

  std::string serialized;
  {
    std::stringstream stream;
    stream << file.rdbuf();
    ASSERT_TRUE(stream);
    serialized = stream.str();
  }

  file.close();

  std::span<const uint8_t> serialized_span(reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size());
  utils::IsGenAiLoraFormatModelBytes(serialized_span.data(), serialized_span.size());

  flatbuffers::Verifier verifier(serialized_span.data(), serialized_span.size());
  ASSERT_TRUE(VerifyParametersBuffer(verifier));

  const auto* fbs_parameters = GetParameters(serialized_span.data());
  ASSERT_NE(nullptr, fbs_parameters) << "Parameters are null";

  ASSERT_TRUE(IsLoraFormatVersionSupported(fbs_parameters->version())) << "Format version mismatch";
  ASSERT_EQ(2U, fbs_parameters->parameters()->size());
  for (const auto* fbs_param : *fbs_parameters->parameters()) {
    ASSERT_EQ(TensorDataType::FLOAT, fbs_param->data_type());
    std::span<const int64_t> shape_span(fbs_param->dims()->data(), fbs_param->dims()->size());
    ASSERT_EQ(2, shape_span.size());

    std::span<const float> data_span(reinterpret_cast<const float*>(fbs_param->raw_data()->data()),
                                     fbs_param->raw_data()->size() / sizeof(float));

    for (size_t i = 0; i < data_span.size(); ++i) {
      ASSERT_TRUE(std::isfinite(data_span[i]));
    }
  }
}

}  // namespace test
}  // namespace lora_parameters
}  // namespace Generators
