// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "adapter_loader.h"

#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "ort_genai.h"

namespace benchmark {
namespace {

struct ParsedSafetensorsTensor {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<int64_t> data_offsets;
};

std::vector<int64_t> ParseInt64Array(const nlohmann::json& value,
                                     const char* field,
                                     const std::string& tensor_name) {
  if (!value.is_array()) {
    throw std::runtime_error("Safetensors field '" + std::string(field) +
                             "' must be an array for tensor: " + tensor_name);
  }

  std::vector<int64_t> values;
  values.reserve(value.size());
  for (const auto& item : value) {
    if (item.is_number_unsigned()) {
      const auto number = item.get<uint64_t>();
      if (number > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("Integer overflow in safetensors field '" +
                                 std::string(field) + "' for tensor: " + tensor_name);
      }
      values.push_back(static_cast<int64_t>(number));
    } else if (item.is_number_integer()) {
      values.push_back(item.get<int64_t>());
    } else {
      throw std::runtime_error("Safetensors field '" + std::string(field) +
                               "' must contain integers for tensor: " + tensor_name);
    }
  }
  return values;
}

std::vector<ParsedSafetensorsTensor> ParseSafetensorsHeader(const std::string& header) {
  nlohmann::json root;
  try {
    root = nlohmann::json::parse(header);
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error("Failed to parse safetensors JSON header: " + std::string(e.what()));
  }

  if (!root.is_object()) {
    throw std::runtime_error("Safetensors header must be a JSON object.");
  }

  std::vector<ParsedSafetensorsTensor> tensors;
  for (auto it = root.begin(); it != root.end(); ++it) {
    const std::string& name = it.key();
    const auto& entry = it.value();
    if (!entry.is_object()) {
      throw std::runtime_error("Malformed safetensors tensor entry.");
    }

    if (name == "__metadata__") {
      continue;
    }

    ParsedSafetensorsTensor tensor{};
    tensor.name = name;
    try {
      tensor.dtype = entry.at("dtype").get<std::string>();
      tensor.shape = ParseInt64Array(entry.at("shape"), "shape", name);
      tensor.data_offsets = ParseInt64Array(entry.at("data_offsets"), "data_offsets", name);
    } catch (const nlohmann::json::exception& e) {
      throw std::runtime_error("Malformed safetensors tensor entry '" + name + "': " + e.what());
    }
    tensors.push_back(std::move(tensor));
  }

  return tensors;
}

constexpr uint64_t kMaxSafetensorsHeaderBytes = 16ull * 1024ull * 1024ull;

size_t ElementCountFromShape(const std::vector<int64_t>& shape, const std::string& name) {
  size_t count = 1;
  for (const int64_t dim : shape) {
    if (dim < 0) {
      throw std::runtime_error("Negative shape dimension in LoRA weight '" + name + "'.");
    }
    const auto dim_size = static_cast<size_t>(dim);
    if (dim_size != 0 && count > (std::numeric_limits<size_t>::max() / dim_size)) {
      throw std::runtime_error("Shape overflow in LoRA weight '" + name + "'.");
    }
    count *= dim_size;
  }
  return count;
}

size_t ElementByteSize(const std::string& dtype, const std::string& name) {
  if (dtype == "F16") {
    return sizeof(uint16_t);
  }
  if (dtype == "I8" || dtype == "U8") {
    return sizeof(uint8_t);
  }
  throw std::runtime_error("Unsupported safetensors dtype for LoRA weight '" + name +
                           "': " + dtype + " (only I8/U8/F16 supported)");
}

}  // namespace

LoadedAdapter LoadSafetensors(const std::string& path) {
  std::ifstream input{path, std::ios::binary};
  if (!input) {
    throw std::runtime_error("Failed to open adapter safetensors file: " + path);
  }

  input.seekg(0, std::ios::end);
  const auto file_size_signed = input.tellg();
  input.seekg(0, std::ios::beg);
  if (!input || file_size_signed < static_cast<std::streamoff>(sizeof(uint64_t))) {
    throw std::runtime_error("Adapter safetensors file is too small: " + path);
  }
  const auto file_size = static_cast<uint64_t>(file_size_signed);

  uint64_t header_size{};
  input.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
  if (!input) {
    throw std::runtime_error("Failed to read safetensors header size: " + path);
  }
  if (header_size == 0 || header_size > kMaxSafetensorsHeaderBytes ||
      header_size > file_size - sizeof(uint64_t)) {
    throw std::runtime_error("Invalid safetensors header size in: " + path);
  }

  std::string header(static_cast<size_t>(header_size), '\0');
  input.read(header.data(), static_cast<std::streamsize>(header_size));
  if (!input) {
    throw std::runtime_error("Failed to read safetensors header: " + path);
  }

  const auto parsed_tensors = ParseSafetensorsHeader(header);
  const auto data_section_offset = static_cast<uint64_t>(sizeof(uint64_t) + header_size);
  const auto data_section_size = file_size - data_section_offset;
  LoadedAdapter adapter{};
  adapter.tensors.reserve(parsed_tensors.size());

  for (const auto& parsed : parsed_tensors) {
    const size_t element_bytes = ElementByteSize(parsed.dtype, parsed.name);
    if (parsed.data_offsets.size() != 2 || parsed.data_offsets[0] < 0 || parsed.data_offsets[1] < 0) {
      throw std::runtime_error("Invalid safetensors data_offsets for tensor: " + parsed.name);
    }

    const auto data_start = static_cast<uint64_t>(parsed.data_offsets[0]);
    const auto data_end = static_cast<uint64_t>(parsed.data_offsets[1]);
    if (data_end < data_start || data_end > data_section_size) {
      throw std::runtime_error("Invalid safetensors data offsets for tensor: " + parsed.name);
    }

    const auto byte_count = static_cast<size_t>(data_end - data_start);
    const size_t element_count = ElementCountFromShape(parsed.shape, parsed.name);
    if (element_bytes != 0 && element_count > (std::numeric_limits<size_t>::max() / element_bytes)) {
      throw std::runtime_error("Payload size overflow in LoRA weight '" + parsed.name + "'.");
    }
    const size_t expected_bytes = element_count * element_bytes;
    if (byte_count != expected_bytes) {
      throw std::runtime_error("LoRA weight '" + parsed.name + "' payload size (" +
                               std::to_string(byte_count) + ") does not match shape (" +
                               std::to_string(expected_bytes) + " bytes).");
    }

    LoadedAdapterTensor tensor{};
    tensor.name = parsed.name;
    tensor.shape = parsed.shape;
    tensor.is_uint8 = parsed.dtype == "U8";
    tensor.is_fp16 = parsed.dtype == "F16";

    input.seekg(static_cast<std::streamoff>(data_section_offset + data_start), std::ios::beg);
    if (tensor.is_fp16) {
      tensor.f16_data.resize(byte_count / sizeof(uint16_t));
      input.read(reinterpret_cast<char*>(tensor.f16_data.data()), static_cast<std::streamsize>(byte_count));
    } else if (tensor.is_uint8) {
      tensor.u8_data.resize(byte_count);
      input.read(reinterpret_cast<char*>(tensor.u8_data.data()), static_cast<std::streamsize>(byte_count));
    } else {
      tensor.data.resize(byte_count);
      input.read(reinterpret_cast<char*>(tensor.data.data()), static_cast<std::streamsize>(byte_count));
    }
    if (!input) {
      throw std::runtime_error("Failed to read safetensors tensor data for: " + parsed.name);
    }

    adapter.tensors.push_back(std::move(tensor));
  }

  if (adapter.tensors.empty()) {
    throw std::runtime_error("No LoRA weight tensors found in safetensors file: " + path);
  }

  return adapter;
}

void BindAdapterToGenerator(OgaGenerator& generator, BoundAdapter& bound_adapter) {
  if (!bound_adapter.loaded) {
    return;
  }

  bound_adapter.ort_tensors.clear();
  bound_adapter.ort_tensors.reserve(bound_adapter.loaded->tensors.size());

  for (const auto& tensor : bound_adapter.loaded->tensors) {
    if (tensor.is_fp16) {
      auto ort_tensor = OgaTensor::Create(
          const_cast<uint16_t*>(tensor.f16_data.data()),
          tensor.shape.data(),
          tensor.shape.size(),
          OgaElementType_float16);
      generator.SetModelInput(tensor.name.c_str(), *ort_tensor);
      bound_adapter.ort_tensors.push_back(std::move(ort_tensor));
      continue;
    }

    if (tensor.is_uint8) {
      auto ort_tensor = OgaTensor::Create(
          const_cast<uint8_t*>(tensor.u8_data.data()),
          tensor.shape.data(),
          tensor.shape.size(),
          OgaElementType_uint8);
      generator.SetModelInput(tensor.name.c_str(), *ort_tensor);
      bound_adapter.ort_tensors.push_back(std::move(ort_tensor));
      continue;
    }

    auto ort_tensor = OgaTensor::Create(
        const_cast<int8_t*>(tensor.data.data()),
        tensor.shape.data(),
        tensor.shape.size(),
        OgaElementType_int8);
    generator.SetModelInput(tensor.name.c_str(), *ort_tensor);
    bound_adapter.ort_tensors.push_back(std::move(ort_tensor));
  }
}

}  // namespace benchmark
