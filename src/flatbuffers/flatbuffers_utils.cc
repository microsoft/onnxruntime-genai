// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "flatbuffers_utils.h"
#include "schema/genai_lora.fbs.h"


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

void SaveLoraParameter(flatbuffers::FlatBufferBuilder& flat_builder, std::string_view name, std::string_view doc,
                       TensorDataType data_type, std::span<const int64_t> shape, std::span<const uint8_t> data,
                       flatbuffers::Offset<Tensor>& fbs_tensor) {
  auto name_str = (name.empty()) ? 0 : flat_builder.CreateString(name.data(), name.size());
  auto doc_str = (doc.empty()) ? 0 : flat_builder.CreateString(doc.data(), doc.size());
  auto shape_vec = flat_builder.CreateVector(shape.data(), shape.size());
  auto data_vec = flat_builder.CreateVector(data.data(), data.size());

  fbs_tensor = CreateTensor(flat_builder, name_str, doc_str, shape_vec, data_type, data_vec);
}



//static Status SaveTensorShapeOrtFormat(flatbuffers::FlatBufferBuilder& builder,
//                                       const TensorShapeProto& tensor_shape_proto,
//                                       flatbuffers::Offset<fbs::Shape>& fbs_shape) {
//  std::vector<flatbuffers::Offset<fbs::Dimension>> dim;
//  dim.reserve(tensor_shape_proto.dim_size());
//  for (const auto& d : tensor_shape_proto.dim()) {
//    auto fbs_d = SaveTensorDimensionOrtFormat(builder, d);
//    dim.push_back(fbs_d);
//  }
//  fbs_shape = fbs::CreateShapeDirect(builder, &dim);
//  return Status::OK();
//}
//
//static Status LoadTensorShapeOrtFormat(const fbs::Shape& fbs_shape, TensorShapeProto& shape_proto) {
//  auto fbs_dims = fbs_shape.dim();
//  if (fbs_dims) {
//    auto dims = shape_proto.mutable_dim();
//    dims->Reserve(fbs_dims->size());
//    for (const auto fbs_dim : *fbs_dims) {
//      ORT_RETURN_IF(nullptr == fbs_dim, "Null entry in dimensions. Invalid ORT format model.");
//      TensorShapeProto_Dimension dim;
//      ORT_RETURN_IF_ERROR(LoadTensorDimensionOrtFormat(*fbs_dim, *dims->Add()));
//    }
//  }
//  return Status::OK();
//}

}  // namespace utils
}  // namespace lora_parameters
}  // namespace Generators
