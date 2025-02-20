// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"

namespace Generators {

std::unique_ptr<Images> LoadImages(std::span<const char* const> image_paths) {
  for (const char* image_path : image_paths) {
    if (!fs::path(image_path).exists()) {
      throw std::runtime_error("Image path does not exist: " + std::string(image_path));
    }
  }
  ort_extensions::OrtxObjectPtr<OrtxRawImages> images{};
  size_t num_images{};
  CheckResult(OrtxLoadImages(images.ToBeAssigned(), const_cast<const char**>(image_paths.data()), image_paths.size(), &num_images));

  return std::make_unique<Images>(std::move(images), num_images);
}

std::unique_ptr<Audios> LoadAudios(const std::span<const char* const>& audio_paths) {
  for (const char* audio_path : audio_paths) {
    if (!fs::path(audio_path).exists()) {
      throw std::runtime_error("Audio path does not exist: " + std::string(audio_path));
    }
  }
  ort_extensions::OrtxObjectPtr<OrtxRawAudios> audios;
  CheckResult(OrtxLoadAudios(audios.ToBeAssigned(), audio_paths.data(), audio_paths.size()));

  return std::make_unique<Audios>(std::move(audios), audio_paths.size());
}

template <typename T>
std::unique_ptr<OrtValue> ProcessTensor(OrtxTensor* tensor, Ort::Allocator& allocator) {
  const T* tensor_data{};
  const int64_t* tensor_shape{};
  size_t tensor_num_dims;
  CheckResult(OrtxGetTensorData(tensor, reinterpret_cast<const void**>(&tensor_data),
                                &tensor_shape, &tensor_num_dims));
  const int64_t tensor_num_elements = std::accumulate(tensor_shape,
                                                      tensor_shape + tensor_num_dims,
                                                      1LL, std::multiplies<int64_t>());
  auto tensor_value = OrtValue::CreateTensor<T>(allocator, std::span<int64_t>(const_cast<int64_t*>(tensor_shape), tensor_num_dims));
  std::copy(tensor_data, tensor_data + tensor_num_elements,
            tensor_value->template GetTensorMutableData<T>());
  return tensor_value;
}

template <>
std::unique_ptr<OrtValue> ProcessTensor<Ort::Float16_t>(OrtxTensor* tensor, Ort::Allocator& allocator) {
  const float* tensor_data{};
  const int64_t* tensor_shape{};
  size_t tensor_num_dims;
  CheckResult(OrtxGetTensorData(tensor, reinterpret_cast<const void**>(&tensor_data),
                                &tensor_shape, &tensor_num_dims));
  const int64_t tensor_num_elements = std::accumulate(tensor_shape,
                                                      tensor_shape + tensor_num_dims,
                                                      1LL, std::multiplies<int64_t>());
  auto tensor_value = OrtValue::CreateTensor<Ort::Float16_t>(allocator, std::span<int64_t>(const_cast<int64_t*>(tensor_shape), tensor_num_dims));
  auto tensor_value_fp32 = OrtValue::CreateTensor<float>(
      allocator.GetInfo(),
      std::span<float>(const_cast<float*>(tensor_data), tensor_num_elements),
      std::span<int64_t>(const_cast<int64_t*>(tensor_shape), tensor_num_dims));
  auto p_device = GetDeviceInterface(DeviceType::CPU);
  Cast(*tensor_value_fp32, tensor_value, *p_device, Ort::TypeToTensorType<Ort::Float16_t>);
  return tensor_value;
}

template <>
std::unique_ptr<OrtValue> ProcessTensor<int64_t, float>(OrtxTensor* tensor, Ort::Allocator& allocator) {
  const int64_t* tensor_data{};
  const int64_t* tensor_shape{};
  size_t tensor_num_dims;
  CheckResult(OrtxGetTensorData(tensor, reinterpret_cast<const void**>(&tensor_data),
                                &tensor_shape, &tensor_num_dims));
  const int64_t tensor_num_elements = std::accumulate(tensor_shape,
                                                      tensor_shape + tensor_num_dims,
                                                      1LL, std::multiplies<int64_t>());
  auto tensor_value = OrtValue::CreateTensor<float>(allocator, std::span<int64_t>(const_cast<int64_t*>(tensor_shape), tensor_num_dims));
  std::transform(tensor_data, tensor_data + tensor_num_elements,
                 tensor_value->GetTensorMutableData<float>(),
                 [](int64_t value) { return static_cast<float>(value); });
  return tensor_value;
}
template <>
std::unique_ptr<OrtValue> ProcessTensor<int64_t, Ort::Float16_t>(OrtxTensor* tensor, Ort::Allocator& allocator) {
  const int64_t* tensor_data{};
  const int64_t* tensor_shape{};
  size_t tensor_num_dims;
  CheckResult(OrtxGetTensorData(tensor, reinterpret_cast<const void**>(&tensor_data),
                                &tensor_shape, &tensor_num_dims));
  const int64_t tensor_num_elements = std::accumulate(tensor_shape,
                                                      tensor_shape + tensor_num_dims,
                                                      1LL, std::multiplies<int64_t>());
  auto tensor_value = OrtValue::CreateTensor<Ort::Float16_t>(allocator, std::span<int64_t>(const_cast<int64_t*>(tensor_shape), tensor_num_dims));
  auto tensor_value_fp32 = OrtValue::CreateTensor<float>(allocator, std::span<int64_t>(const_cast<int64_t*>(tensor_shape), tensor_num_dims));
  std::transform(tensor_data, tensor_data + tensor_num_elements,
                 tensor_value_fp32->GetTensorMutableData<float>(),
                 [](int64_t value) { return static_cast<float>(value); });
  auto p_device = GetDeviceInterface(DeviceType::CPU);
  Cast(*tensor_value_fp32, tensor_value, *p_device, Ort::TypeToTensorType<Ort::Float16_t>);
  return tensor_value;
}

template std::unique_ptr<OrtValue> ProcessTensor<float>(OrtxTensor* tensor, Ort::Allocator& allocator);
template std::unique_ptr<OrtValue> ProcessTensor<Ort::Float16_t>(OrtxTensor* tensor, Ort::Allocator& allocator);
template std::unique_ptr<OrtValue> ProcessTensor<int64_t>(OrtxTensor* tensor, Ort::Allocator& allocator);

template std::unique_ptr<OrtValue> ProcessTensor<int64_t, float>(OrtxTensor* tensor, Ort::Allocator& allocator);
template std::unique_ptr<OrtValue> ProcessTensor<int64_t, Ort::Float16_t>(OrtxTensor* tensor, Ort::Allocator& allocator);

}  // namespace Generators