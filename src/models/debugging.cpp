// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "utils.h"
#include <cinttypes>
#include "model.h"

namespace Generators {
static constexpr size_t c_value_count = 10;  // Dump this many values from the start of a tensor

template <typename... Types>
const char* TypeToString(ONNXTensorElementDataType type, Ort::TypeList<Types...>) {
  const char* name = "(please add type to list)";
  (void)((type == Ort::TypeToTensorType<Types> ? name = typeid(Types).name(), true : false) || ...);
  return name;
}

const char* TypeToString(ONNXTensorElementDataType type) {
  return TypeToString(type, Ort::TensorTypes{});
}

std::ostream& operator<<(std::ostream& stream, Ort::Float16_t v) {
  stream << Float16ToFloat32(v);
  return stream;
}

std::ostream& operator<<(std::ostream& stream, Ort::BFloat16_t v) {
  stream << "BF16:" << v.value;  // TODO: implement conversion when useful
  return stream;
}

template <typename T>
void DumpSpan(std::ostream& stream, std::span<const T> values) {
  // If type is uint8_t or int8_t cast to int so it displays as an int vs a char
  using DisplayType = std::conditional_t<std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>, int, T>;

  if (values.size() <= c_value_count) {
    for (auto v : values)
      stream << static_cast<DisplayType>(v) << ' ';
  } else {
    for (size_t i = 0; i < c_value_count / 2; i++)
      stream << static_cast<DisplayType>(values[i]) << ' ';
    stream << "... ";
    for (size_t i = values.size() - c_value_count / 2; i < values.size(); i++)
      stream << static_cast<DisplayType>(values[i]) << ' ';
  }
}

template <typename... Types>
bool DumpSpan(std::ostream& stream, ONNXTensorElementDataType type, const void* p_values_raw, size_t count, Ort::TypeList<Types...>) {
  return ((type == Ort::TypeToTensorType<Types> && (DumpSpan(stream, std::span<const Types>{reinterpret_cast<const Types*>(p_values_raw), count}), true)) || ...);
}

void DumpValues(std::ostream& stream, ONNXTensorElementDataType type, const void* p_values_raw, size_t count) {
  if (count == 0) {
    return;
  }

  stream << SGR::Fg_Green << "Values[ " << SGR::Reset;
  if (!DumpSpan(stream, type, p_values_raw, count, Ort::TensorTypes{}))
    stream << SGR::Fg_Red << "Unhandled data type" << SGR::Reset;

  stream << SGR::Fg_Green << "]" << SGR::Reset << std::endl;
}

void DumpTensor(const Model& model, std::ostream& stream, OrtValue* value, bool dump_value) {
  if (!value) {
    return;
  }
  auto type_info = value->GetTensorTypeAndShapeInfo();
  auto shape = type_info->GetShape();
  stream << SGR::Fg_Green << "Shape[ " << SGR::Reset;
  for (auto dim : shape) {
    stream << static_cast<int>(dim) << ' ';
  }
  stream << SGR::Fg_Green << ']' << SGR::Reset;
  stream << SGR::Fg_Green << " Type: " << SGR::Reset << TypeToString(type_info->GetElementType());

  size_t element_count = type_info->GetElementCount();
  if (!dump_value)
    element_count = 0;

  stream << SGR::Fg_Green << " Location: " << SGR::Reset;

  const auto& memory_info = value->GetTensorMemoryInfo();
  switch (memory_info.GetDeviceType()) {
    case OrtMemoryInfoDeviceType_CPU:
      stream << "CPU\r\n";
      DumpValues(stream, type_info->GetElementType(), value->GetTensorRawData(), element_count);
      break;
    case OrtMemoryInfoDeviceType_GPU: {
      stream << "GPU\r\n";
      auto type = type_info->GetElementType();
      auto tensor_span = std::span<uint8_t>{const_cast<OrtValue*>(value)->GetTensorMutableData<uint8_t>(), SizeOf(type) * element_count};
      auto device_span = model.p_device_->WrapMemory<uint8_t>(tensor_span);
      DumpValues(stream, type, device_span.CopyDeviceToCpu().data(), element_count);
      break;
    }
    default:
      stream << "Unhandled device type: " << static_cast<int>(memory_info.GetDeviceType()) << "\r\n";
      break;
  }
}

void DumpTensors(const Model& model, std::ostream& stream, OrtValue** values, const char** names, size_t count, bool dump_values) {
  for (size_t i = 0; i < count; i++) {
    stream << SGR::Fg_Green << "Name: " << SGR::Reset << names[i] << ' ';
    DumpTensor(model, stream, values[i], dump_values);
  }
}

}  // namespace Generators
