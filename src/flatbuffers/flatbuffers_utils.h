// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../src/flatbuffers.h"
#include "../src/span.h"

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
void SaveLoraParameter(flatbuffers::FlatBufferBuilder& flat_builder, std::string_view name, std::string_view doc,
                       TensorDataType data_type, std::span<const int64_t> shape, std::span<const uint8_t> data,
                       flatbuffers::Offset<Tensor>& fbs_tensor);

/// <summary>
/// Creates an OrtValue on top of the flatbuffer tensor
/// </summary>
/// <param name="tensor"></param>
/// <param name="ort_value"></param>
void LoadLoraParameter(const Tensor& tensor, std::unique_ptr<OrtValue>& ort_value);

// check if bytes has fileidentifier for lora parameters
bool IsGenAiLoraFormatModelBytes(const void* bytes, size_t num_bytes);

}  // namespace utils
}  // namespace lora_parameters
}  // namespace Generators

