// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "adapter_loader.h"

#include <charconv>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>

#include "ort_genai.h"

namespace benchmark {
namespace {

void SkipWhitespace(std::string_view header, size_t& pos) {
  while (pos < header.size() && std::isspace(static_cast<unsigned char>(header[pos]))) {
    ++pos;
  }
}

std::string ParseJsonString(std::string_view header, size_t& pos) {
  if (pos >= header.size() || header[pos] != '"') {
    throw std::runtime_error("Expected JSON string.");
  }
  ++pos;
  std::string value;
  while (pos < header.size()) {
    const char c = header[pos++];
    if (c == '"') {
      return value;
    }
    if (c == '\\') {
      if (pos >= header.size()) {
        throw std::runtime_error("Invalid escape sequence in JSON string.");
      }
      value.push_back(header[pos++]);
    } else {
      value.push_back(c);
    }
  }
  throw std::runtime_error("Unterminated JSON string.");
}

void SkipJsonValue(std::string_view header, size_t& pos) {
  SkipWhitespace(header, pos);
  if (pos >= header.size()) {
    throw std::runtime_error("Unexpected end of JSON.");
  }

  const char c = header[pos];
  if (c == '"') {
    ParseJsonString(header, pos);
    return;
  }
  if (c == '{') {
    size_t depth = 0;
    do {
      if (header[pos] == '{') {
        ++depth;
      } else if (header[pos] == '}') {
        --depth;
      }
      ++pos;
    } while (pos < header.size() && depth > 0);
    return;
  }
  if (c == '[') {
    size_t depth = 0;
    do {
      if (header[pos] == '[') {
        ++depth;
      } else if (header[pos] == ']') {
        --depth;
      }
      ++pos;
    } while (pos < header.size() && depth > 0);
    return;
  }

  while (pos < header.size() &&
         header[pos] != ',' && header[pos] != '}' && header[pos] != ']' &&
         !std::isspace(static_cast<unsigned char>(header[pos]))) {
    ++pos;
  }
}

std::vector<int64_t> ParseInt64Array(std::string_view header, size_t& pos) {
  if (pos >= header.size() || header[pos] != '[') {
    throw std::runtime_error("Expected JSON array.");
  }
  ++pos;
  std::vector<int64_t> values;
  SkipWhitespace(header, pos);
  if (pos < header.size() && header[pos] == ']') {
    ++pos;
    return values;
  }

  while (pos < header.size()) {
    SkipWhitespace(header, pos);
    size_t number_end = pos;
    while (number_end < header.size() &&
           (std::isdigit(static_cast<unsigned char>(header[number_end])) || header[number_end] == '-')) {
      ++number_end;
    }
    int64_t value{};
    const auto number_view = header.substr(pos, number_end - pos);
    const auto [ptr, ec] = std::from_chars(number_view.data(), number_view.data() + number_view.size(), value);
    if (ec != std::errc{} || ptr != number_view.data() + number_view.size()) {
      throw std::runtime_error("Failed to parse JSON array number.");
    }
    values.push_back(value);
    pos = number_end;
    SkipWhitespace(header, pos);
    if (pos < header.size() && header[pos] == ',') {
      ++pos;
      continue;
    }
    if (pos < header.size() && header[pos] == ']') {
      ++pos;
      return values;
    }
    throw std::runtime_error("Malformed JSON array.");
  }
  throw std::runtime_error("Unterminated JSON array.");
}

struct ParsedSafetensorsTensor {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<int64_t> data_offsets;
};

std::vector<ParsedSafetensorsTensor> ParseSafetensorsHeader(std::string_view header) {
  std::vector<ParsedSafetensorsTensor> tensors;
  size_t pos = 0;
  SkipWhitespace(header, pos);
  if (pos >= header.size() || header[pos] != '{') {
    throw std::runtime_error("Safetensors header must be a JSON object.");
  }
  ++pos;

  while (pos < header.size()) {
    SkipWhitespace(header, pos);
    if (pos < header.size() && header[pos] == '}') {
      break;
    }

    const std::string name = ParseJsonString(header, pos);
    SkipWhitespace(header, pos);
    if (pos >= header.size() || header[pos] != ':') {
      throw std::runtime_error("Malformed safetensors header entry.");
    }
    ++pos;
    SkipWhitespace(header, pos);
    if (pos >= header.size() || header[pos] != '{') {
      throw std::runtime_error("Malformed safetensors tensor entry.");
    }
    ++pos;

    if (name != "__metadata__") {
      tensors.push_back(ParsedSafetensorsTensor{.name = name});
    }

    while (pos < header.size()) {
      SkipWhitespace(header, pos);
      if (pos < header.size() && header[pos] == '}') {
        ++pos;
        break;
      }

      const std::string field = ParseJsonString(header, pos);
      SkipWhitespace(header, pos);
      if (pos >= header.size() || header[pos] != ':') {
        throw std::runtime_error("Malformed safetensors tensor field.");
      }
      ++pos;
      SkipWhitespace(header, pos);

      if (name != "__metadata__") {
        auto& tensor = tensors.back();
        if (field == "dtype") {
          tensor.dtype = ParseJsonString(header, pos);
        } else if (field == "shape") {
          tensor.shape = ParseInt64Array(header, pos);
        } else if (field == "data_offsets") {
          tensor.data_offsets = ParseInt64Array(header, pos);
        } else {
          SkipJsonValue(header, pos);
        }
      } else {
        SkipJsonValue(header, pos);
      }

      SkipWhitespace(header, pos);
      if (pos < header.size() && header[pos] == ',') {
        ++pos;
      }
    }

    SkipWhitespace(header, pos);
    if (pos < header.size() && header[pos] == ',') {
      ++pos;
    }
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
          tensor.shape,
          OgaElementType_float16);
      generator.SetModelInput(tensor.name.c_str(), *ort_tensor);
      bound_adapter.ort_tensors.push_back(std::move(ort_tensor));
      continue;
    }

    if (tensor.is_uint8) {
      auto ort_tensor = OgaTensor::Create(
          const_cast<uint8_t*>(tensor.u8_data.data()),
          tensor.shape,
          OgaElementType_uint8);
      generator.SetModelInput(tensor.name.c_str(), *ort_tensor);
      bound_adapter.ort_tensors.push_back(std::move(ort_tensor));
      continue;
    }

    auto ort_tensor = OgaTensor::Create(
        const_cast<int8_t*>(tensor.data.data()),
        tensor.shape,
        OgaElementType_int8);
    generator.SetModelInput(tensor.name.c_str(), *ort_tensor);
    bound_adapter.ort_tensors.push_back(std::move(ort_tensor));
  }
}

}  // namespace benchmark
