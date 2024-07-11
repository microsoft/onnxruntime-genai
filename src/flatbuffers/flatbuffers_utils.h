// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../flatbuffers.h"
#include "../span.h"

#include "schema/genai_lora.fbs.h"

#include <string>
#include <string_view>
#include <unordered_map>

struct OrtValue;

namespace Generators {
namespace lora_parameters {
namespace utils {

constexpr auto kInvalidLoraFormatModelMessage = "Invalid Lora Parameter Format";

// Will only create string in flatbuffers when has_string is true
flatbuffers::Offset<flatbuffers::String> SaveStringToLoraFormat(flatbuffers::FlatBufferBuilder& builder,
                                                                bool has_string, const std::string& src);

void LoadStringFromLoraFormat(std::string& dst, const flatbuffers::String* fbs_string);

/// <summary>
/// Serializes tensor data into flatbuffer
/// </summary>
/// <param name="flat_builder"></param>
/// <param name="name">parameter name</param>
/// <param name="doc">doc, optional</param>
/// <param name="data_type"></param>
/// <param name="shape"></param>
/// <param name="data"></param>
/// <param name="fbs_tensor">output offset</param>
void SaveLoraParameter(flatbuffers::FlatBufferBuilder& flat_builder, std::string_view name,
                       Generators::lora_parameters::TensorDataType data_type,
                       std::span<const int64_t> shape, std::span<const uint8_t> data,
                       flatbuffers::Offset<Generators::lora_parameters::Tensor>& fbs_tensor);

/// <summary>
/// Create an OrtValue on top of the flatbuffer tensor
/// No copying of data is done here. The caller is responsible for managing the lifetime of flatbuffer
/// structures.
///
/// In this scenario, one can memory map the entire flatbuffer tensor data into OrtValue without copying.
/// </summary>
/// <param name="tensor"></param>
/// <returns></returns>
std::pair<std::string, std::unique_ptr<OrtValue>> CreateOrtValueOverFlatBufferLoraParameter(
    const Generators::lora_parameters::Tensor& tensor);

// check if bytes has fileidentifier for lora parameters
bool IsGenAiLoraFormatModelBytes(const void* bytes, size_t num_bytes);

}  // namespace utils
}  // namespace lora_parameters
}  // namespace Generators
