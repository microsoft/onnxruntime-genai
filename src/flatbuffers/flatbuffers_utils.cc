// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "flatbuffers_utils.h"
#include "schema/genai_lora.fbs.h"
#include "../../src/models/onnxruntime_api.h"

#include "../models/onnxruntime_api.h"

namespace Generators {
namespace lora_parameters {
namespace utils {

bool IsGenAiLoraFormatModelBytes(const void* bytes, size_t num_bytes) {
  return num_bytes > 8 &&  // check buffer is large enough to contain identifier so we don't read random memory
         ParametersBufferHasIdentifier(bytes);
}

flatbuffers::Offset<flatbuffers::String> SaveStringToLoraFormat(flatbuffers::FlatBufferBuilder& builder,
                                                                bool has_string, const std::string& src) {
  if (has_string) return builder.CreateString(src);

  // If the string does not exist, return 0 (the string does not exist in flatbuffer)
  return 0;
}

void LoadStringFromLoraFormat(std::string& dst, const flatbuffers::String* fbs_string) {
  if (fbs_string) {
    dst = fbs_string->str();
  }
}

void SaveLoraParameter(flatbuffers::FlatBufferBuilder& flat_builder, std::string_view name,
                       Generators::lora_parameters::TensorDataType data_type, std::span<const int64_t> shape,
                       std::span<const uint8_t> data,
                       flatbuffers::Offset<Generators::lora_parameters::Tensor>& fbs_tensor) {
  auto name_str = (name.empty()) ? 0 : flat_builder.CreateString(name.data(), name.size());
  auto shape_vec = flat_builder.CreateVector(shape.data(), shape.size());
  auto data_vec = flat_builder.CreateVector(data.data(), data.size());

  fbs_tensor = CreateTensor(flat_builder, name_str, shape_vec, data_type, data_vec);
}

std::pair<std::string, std::unique_ptr<OrtValue>> CreateOrtValueOverFlatBufferLoraParameter(
    const Generators::lora_parameters::Tensor& tensor) {
  std::string name;
  LoadStringFromLoraFormat(name, tensor.name());

  const auto data_type = tensor.data_type();

  std::vector<int64_t> dims;
  dims.reserve(tensor.dims()->size());
  for (auto d : *tensor.dims()) {
    dims.push_back(d);
  }

  auto mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  auto ort_value =
      OrtValue::CreateTensor(*mem_info, const_cast<uint8_t*>(tensor.raw_data()->data()),
                             static_cast<size_t>(tensor.raw_data()->size()), dims,
                             static_cast<ONNXTensorElementDataType>(data_type));
  return std::make_pair(std::move(name), std::move(ort_value));
}

}  // namespace utils
}  // namespace lora_parameters
}  // namespace Generators
