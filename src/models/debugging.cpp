// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "utils.h"
#include <cinttypes>

namespace Generators {
static constexpr size_t c_value_count = 10;  // Dump this many values from the start of a tensor

const char* TypeToString(ONNXTensorElementDataType type) {
  switch (type) {
    case Ort::TypeToTensorType<bool>::type:
      return "bool";
    case Ort::TypeToTensorType<int8_t>::type:
      return "int8";
    case Ort::TypeToTensorType<uint8_t>::type:
      return "uint8";
    case Ort::TypeToTensorType<int16_t>::type:
      return "int16";
    case Ort::TypeToTensorType<uint16_t>::type:
      return "uint16";
    case Ort::TypeToTensorType<int32_t>::type:
      return "int32";
    case Ort::TypeToTensorType<int64_t>::type:
      return "int64";
    case Ort::TypeToTensorType<Ort::Float16_t>::type:
      return "float16";
    case Ort::TypeToTensorType<float>::type:
      return "float32";
    case Ort::TypeToTensorType<double>::type:
      return "float64";
    default:
      assert(false);
      return "(please add type to list)";
  }
}

std::ostream& operator<<(std::ostream& stream, Ort::Float16_t v) {
  stream << Float16ToFloat32(v);
  return stream;
}

template <typename T>
void DumpSpan(std::ostream& stream, std::span<const T> values) {
  if (values.size() <= c_value_count) {
    for (auto v : values)
      stream << v << ' ';
  } else {
    for (size_t i = 0; i < c_value_count / 2; i++)
      stream << values[i] << ' ';
    stream << "... ";
    for (size_t i = values.size() - c_value_count / 2; i < values.size(); i++)
      stream << values[i] << ' ';
  }
}

#if USE_CUDA
template <typename T>
void DumpCudaSpan(std::ostream& stream, std::span<const T> data) {
  auto cpu_copy = std::make_unique<T[]>(data.size());
  CudaCheck() == cudaMemcpy(cpu_copy.get(), data.data(), data.size_bytes(), cudaMemcpyDeviceToHost);

  DumpSpan(stream, std::span<const T>{cpu_copy.get(), data.size()});
}
template void DumpCudaSpan(std::ostream&, std::span<const float>);
template void DumpCudaSpan(std::ostream&, std::span<const int32_t>);
#endif

void DumpValues(std::ostream& stream, ONNXTensorElementDataType type, const void* p_values_raw, size_t count) {
  if (count == 0) {
    return;
  }

  stream << SGR::Fg_Green << "Values[ " << SGR::Reset;

  switch (type) {
    case Ort::TypeToTensorType<bool>::type:
      DumpSpan(stream, std::span<const bool>(reinterpret_cast<const bool*>(p_values_raw), count));
      break;

    case Ort::TypeToTensorType<int8_t>::type:
      DumpSpan(stream, std::span<const int8_t>(reinterpret_cast<const int8_t*>(p_values_raw), count));
      break;

    case Ort::TypeToTensorType<uint8_t>::type:
      DumpSpan(stream, std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(p_values_raw), count));
      break;

    case Ort::TypeToTensorType<int16_t>::type:
      DumpSpan(stream, std::span<const int16_t>(reinterpret_cast<const int16_t*>(p_values_raw), count));
      break;

    case Ort::TypeToTensorType<uint16_t>::type:
      DumpSpan(stream, std::span<const uint16_t>(reinterpret_cast<const uint16_t*>(p_values_raw), count));
      break;

    case Ort::TypeToTensorType<int32_t>::type:
      DumpSpan(stream, std::span<const int32_t>{reinterpret_cast<const int32_t*>(p_values_raw), count});
      break;

    case Ort::TypeToTensorType<uint32_t>::type:
      DumpSpan(stream, std::span<const uint32_t>(reinterpret_cast<const uint32_t*>(p_values_raw), count));
      break;

    case Ort::TypeToTensorType<int64_t>::type:
      DumpSpan(stream, std::span<const int64_t>(reinterpret_cast<const int64_t*>(p_values_raw), count));
      break;

    case Ort::TypeToTensorType<uint64_t>::type:
      DumpSpan(stream, std::span<const uint64_t>{reinterpret_cast<const uint64_t*>(p_values_raw), count});
      break;

    case Ort::TypeToTensorType<Ort::Float16_t>::type:
      DumpSpan(stream, std::span<const Ort::Float16_t>(reinterpret_cast<const Ort::Float16_t*>(p_values_raw), count));
      break;

    case Ort::TypeToTensorType<float>::type:
      DumpSpan(stream, std::span<const float>(reinterpret_cast<const float*>(p_values_raw), count));
      break;

    case Ort::TypeToTensorType<double>::type:
      DumpSpan(stream, std::span<const double>(reinterpret_cast<const double*>(p_values_raw), count));
      break;

    default:
      stream << SGR::Fg_Red << "Unhandled data type" << SGR::Reset;
      break;
  }
  stream << SGR::Fg_Green << "]" << SGR::Reset << std::endl;
}

void DumpTensor(std::ostream& stream, OrtValue* value, bool dump_value) {
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
#if USE_CUDA
      auto type = type_info->GetElementType();
      size_t element_size = SizeOf(type);
      auto cpu_copy = std::make_unique<uint8_t[]>(element_size * element_count);
      CudaCheck() == cudaMemcpy(cpu_copy.get(), value->GetTensorRawData(), element_size * element_count, cudaMemcpyDeviceToHost);
      DumpValues(stream, type, cpu_copy.get(), element_count);
#else
      stream << "Unexpected, using GPU memory but not compiled with CUDA?";
#endif
      break;
    }
    default:
      stream << "Unhandled device type";
      break;
  }
}

void DumpTensors(std::ostream& stream, OrtValue** values, const char** names, size_t count, bool dump_values) {
  for (size_t i = 0; i < count; i++) {
    stream << SGR::Fg_Green << "Name: " << SGR::Reset << names[i] << ' ';
    DumpTensor(stream, values[i], dump_values);
  }
}

}  // namespace Generators
