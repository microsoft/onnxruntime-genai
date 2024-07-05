// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "model.h"

namespace Generators {

size_t SizeOf(ONNXTensorElementDataType type) {
  switch (type) {
    case Ort::TypeToTensorType<uint8_t>::type:
      return sizeof(uint8_t);
    case Ort::TypeToTensorType<int8_t>::type:
      return sizeof(int8_t);
    case Ort::TypeToTensorType<uint16_t>::type:
      return sizeof(uint16_t);
    case Ort::TypeToTensorType<int16_t>::type:
      return sizeof(int16_t);
    case Ort::TypeToTensorType<uint32_t>::type:
      return sizeof(uint32_t);
    case Ort::TypeToTensorType<int32_t>::type:
      return sizeof(int32_t);
    case Ort::TypeToTensorType<uint64_t>::type:
      return sizeof(int64_t);
    case Ort::TypeToTensorType<int64_t>::type:
      return sizeof(int64_t);
    case Ort::TypeToTensorType<bool>::type:
      return sizeof(bool);
    case Ort::TypeToTensorType<float>::type:
      return sizeof(float);
    case Ort::TypeToTensorType<double>::type:
      return sizeof(double);
    case Ort::TypeToTensorType<Ort::Float16_t>::type:
      return sizeof(Ort::Float16_t);
    case Ort::TypeToTensorType<Ort::BFloat16_t>::type:
      return sizeof(Ort::BFloat16_t);
    default:
      throw std::runtime_error("Unsupported ONNXTensorElementDataType in GetTypeSize");
  }
}

// IEEE 752-2008 binary16 format, 1 sign bit, 5 bit exponent, 10 bit fraction
float Float16ToFloat32(uint16_t v) {
  // Extract sign, exponent, and fraction from numpy.float16
  int const sign = (v & 0x8000) >> 15;
  int const exponent = (v & 0x7C00) >> 10;
  int const fraction = v & 0x03FF;

  // Handle special cases
  if (exponent == 0) {
    if (fraction == 0) {
      // Zero
      return sign != 0 ? -0.0f : 0.0f;
    }  // Subnormal number
    return std::ldexp((sign != 0 ? -1.0f : 1.0f) * static_cast<float>(fraction) / 1024.0f, -14);
  }
  if (exponent == 31) {
    if (fraction == 0) {
      // Infinity
      return sign != 0 ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
    }  // NaN
    return std::numeric_limits<float>::quiet_NaN();
  }

  // Normalized number
  return std::ldexp((sign != 0 ? -1.0f : 1.0f) * (1.0f + static_cast<float>(fraction) / 1024.0f), exponent - 15);
}

// C++17 compatible version of bit_cast for the code below
template <typename TTo, typename TFrom>
TTo bit_cast(TFrom x) {
  return *reinterpret_cast<TTo*>(&x);
}

// IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5,
// +-5.9604645E-8, 3.311 digits IEEE 752-2008 binary16 format, 1 sign bit, 5 bit exponent, 10 bit fraction
float FastFloat16ToFloat32(const uint16_t x) {
  const uint32_t e = (x & 0x7C00) >> 10;  // exponent
  const uint32_t m = (x & 0x03FF) << 13;  // mantissa

  const uint32_t v = bit_cast<uint32_t>((float)m) >> 23;  // log2 bit hack to count leading zeros in denormalized format
  return bit_cast<float>((x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) |
                         ((e == 0) & (m != 0)) *
                             ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000)));  // sign : normalized : denormalized
}

uint16_t FastFloat32ToFloat16(float v) {
  const uint32_t b =
      bit_cast<uint32_t>(v) + 0x00001000;  // round-to-nearest-even: add last bit after truncated mantissa

  const uint32_t e = (b & 0x7F800000) >> 23;  // exponent
  const uint32_t m = b & 0x007FFFFF;  // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator
                                      // flag - initial rounding
  return static_cast<uint16_t>((b & 0x80000000) >> 16 | (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
                               ((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) |
                               (e > 143) * 0x7FFF);  // sign : normalized : denormalized : saturate
}

void CopyToDevice(const Model& model, const OrtValue& source, OrtValue& ort_device) {
  auto type_and_shape = source.GetTensorTypeAndShapeInfo();
#if defined(USE_DML) || defined(USE_CUDA)
  const auto copy_size_in_bytes = 
    type_and_shape->GetElementCount() * SizeOf(type_and_shape->GetElementType());
  auto target_data = ort_device.GetTensorMutableRawData();
#endif

  if (model.device_type_ == DeviceType::DML) {
#if USE_DML
    //  Copy to DML device
    ComPtr<ID3D12Resource> target_resource;
    Ort::ThrowOnError(
        model.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(model.allocator_device_, target_data, &target_resource));

    auto source_span = std::span(source.GetTensorData<const uint8_t>(), copy_size_in_bytes);

    model.GetDmlUploadHeap()->BeginUploadToGpu(target_resource.Get(), 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                               source_span);
#else
    throw std::runtime_error("DML is not supported in this build");
#endif
  } else if (model.device_type_ == DeviceType::CUDA) {
#if USE_CUDA
    cudaMemcpyAsync(target_data, source.GetTensorRawData(), copy_size_in_bytes, cudaMemcpyHostToDevice,
        model.cuda_stream_);
#else
    throw std::runtime_error("CUDA is not supported in this build");
#endif

  } else
    throw std::runtime_error("Unsupported device type detected: " + 
      std::to_string(static_cast<int>(model.device_type_)));
}

std::shared_ptr<OrtValue> CopyToDevice(const OrtValue& source, const Model& model) {
  auto type_and_shape = source.GetTensorTypeAndShapeInfo();
  auto ort_device_value =
      OrtValue::CreateTensor(*model.GetAllocatorDevice(), type_and_shape->GetShape(), type_and_shape->GetElementType());

  CopyToDevice(model, source, *ort_device_value);
  return ort_device_value;
}

std::shared_ptr<OrtValue> DuplicateOrtValue(OrtValue& source) {
  // Create a duplicate of the ort_value over the same user supplied buffer
  auto type_and_shape = source.GetTensorTypeAndShapeInfo();
  const auto& mem_info = source.GetTensorMemoryInfo();
  const auto size_in_bytes = SizeOf(type_and_shape->GetElementType()) * type_and_shape->GetElementCount();
  return OrtValue::CreateTensor(mem_info, source.GetTensorMutableRawData(), size_in_bytes, type_and_shape->GetShape(),
                                type_and_shape->GetElementType());
}

}  // namespace Generators