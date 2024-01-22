#include "../generators.h"
#include "debugging.h"
#include <cinttypes>

namespace Generators {
static constexpr size_t c_value_count = 10;  // Dump this many values from the start of a tensor

const char* TypeToString(ONNXTensorElementDataType type) {
  switch (type) {
    case Ort::TypeToTensorType<int64_t>::type:
      return "int64";
    case Ort::TypeToTensorType<int32_t>::type:
      return "int32";
    case Ort::TypeToTensorType<float>::type:
      return "float32";
    case Ort::TypeToTensorType<Ort::Float16_t>::type:
      return "float16";
    default:
      assert(false);
      return "(please add to list)";
  }
}

void DumpValues(ONNXTensorElementDataType type, const void* p_values_raw, size_t count) {
  if (count == 0) {
    return;
  }

  printf("Values: ");

  switch (type) {
    case Ort::TypeToTensorType<int64_t>::type: {
      const auto* p_values = reinterpret_cast<const int64_t*>(p_values_raw);
      for (size_t i = 0; i < count; i++) {
        printf("%" PRId64 " ", p_values[i]);
      }
      break;
    }

    case Ort::TypeToTensorType<int32_t>::type: {
      const auto* p_values = reinterpret_cast<const int32_t*>(p_values_raw);
      for (size_t i = 0; i < count; i++) {
        printf("%d ", p_values[i]);
      }
      break;
    }

    case Ort::TypeToTensorType<float>::type: {
      const auto* p_values = reinterpret_cast<const float*>(p_values_raw);
      for (size_t i = 0; i < count; i++) {
        printf("%f ", p_values[i]);
      }
      break;
    }

    case Ort::TypeToTensorType<Ort::Float16_t>::type: {
      const auto* p_values = reinterpret_cast<const Ort::Float16_t*>(p_values_raw);
      for (size_t i = 0; i < count; i++) {
        printf("%f ", Float16ToFloat32(p_values[i].value));
      }
      break;
    }
  }
  printf("\r\n");
}

void DumpTensor(OrtValue* value, bool dump_value) {
  auto type_info = value->GetTensorTypeAndShapeInfo();
  auto shape = type_info->GetShape();
  printf("Shape { ");
  for (auto dim : shape) {
    printf("%d ", static_cast<int>(dim));
  }
  printf("}");
  printf(" Type: %s", TypeToString(type_info->GetElementType()));

  size_t element_count = std::min<size_t>(type_info->GetElementCount(), c_value_count);
  if (!dump_value) {
    element_count = 0;
  }

  printf(" Location: ");

  const auto& memory_info = value->GetTensorMemoryInfo();
  switch (memory_info.GetDeviceType()) {
    case OrtMemoryInfoDeviceType_CPU:
      printf("CPU\r\n");
      DumpValues(type_info->GetElementType(), value->GetTensorRawData(), element_count);
      break;
    case OrtMemoryInfoDeviceType_GPU:
      printf("GPU\r\n");
#if USE_CUDA
      auto type = type_info->GetElementType();
      size_t element_size = 1;
      switch (type) {
        case Ort::TypeToTensorType<int64_t>::type:
          element_size = sizeof(int64_t);
          break;
        case Ort::TypeToTensorType<int32_t>::type:
          element_size = sizeof(int32_t);
          break;
        case Ort::TypeToTensorType<float>::type:
          element_size = sizeof(float);
          break;
        case Ort::TypeToTensorType<Ort::Float16_t>::type:
          element_size = sizeof(Ort::Float16_t);
          break;
        default:
          assert(false);
          break;
      }
      auto cpu_copy = std::make_unique<uint8_t[]>(element_size * element_count);
      CudaCheck() == cudaMemcpy(cpu_copy.get(), value->GetTensorRawData(), element_size * element_count, cudaMemcpyDeviceToHost);
      DumpValues(type, cpu_copy.get(), element_count);
#else
      printf("Unexpected, using GPU memory but not compiled with CUDA?");
#endif
      break;
  }
}

void DumpTensors(OrtValue** values, const char** names, size_t count, bool dump_values) {
  for (size_t i = 0; i < count; i++) {
    printf("%s ", names[i]);
    DumpTensor(values[i], dump_values);
  }
}

void DumpMemory(const char* name, std::span<const int32_t> data) {
  printf("%s  ", name);
  for (auto v : data) {
    printf("%d ", v);
  }
  printf("\r\n");
}

void DumpMemory(const char* name, std::span<const float> data) {
  printf("%s  ", name);
  for (auto v : data) {
    printf("%f ", v);
  }
  printf("\r\n");
}

void DumpMemory(const char* name, std::span<const Ort::Float16_t> data) {
  printf("%s  ", name);
  for (auto v : data) {
    printf("%f ", Float16ToFloat32(v.value));
  }
  printf("\r\n");
}

#if USE_CUDA
void DumpCudaMemory(const char* name, std::span<const int32_t> data) {
  printf("%s  ", name);
  auto cpu_copy = std::make_unique<int32_t[]>(data.size());
  CudaCheck() == cudaMemcpy(cpu_copy.get(), data.data(), data.size_bytes(), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < data.size(); i++) {
    printf("%d ", cpu_copy[i]);
  }
  printf("\r\n");
}

void DumpCudaMemory(const char* name, std::span<const float> data) {
  printf("%s  ", name);
  auto cpu_copy = std::make_unique<float[]>(data.size());
  CudaCheck() == cudaMemcpy(cpu_copy.get(), data.data(), data.size_bytes(), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < data.size(); i++) {
    printf("%f ", cpu_copy[i]);
  }
  printf("\r\n");
}
#endif

}  // namespace Generators
