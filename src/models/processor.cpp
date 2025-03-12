// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"

namespace Generators {

std::unique_ptr<Images> LoadImages(std::span<const char* const> image_paths) {
  if (image_paths.empty())
    throw std::runtime_error("No images provided");

  for (const char* image_path : image_paths) {
    if (!fs::path(image_path).exists()) {
      throw std::runtime_error("Image path does not exist: " + std::string(image_path));
    }
  }
  ort_extensions::OrtxObjectPtr<OrtxRawImages> images;
  size_t num_images{};
  CheckResult(OrtxLoadImages(images.ToBeAssigned(), const_cast<const char**>(image_paths.data()), image_paths.size(), &num_images));

  return std::make_unique<Images>(std::move(images), num_images);
}

std::unique_ptr<Images> LoadImagesFromBuffers(std::span<const void*> image_data,
                                              std::span<const size_t> image_data_sizes) {
  if (image_data.empty() || image_data_sizes.empty())
    throw std::runtime_error("No images provided");
  if (image_data.size() != image_data_sizes.size())
    throw std::runtime_error("Number of image data buffers does not match the number of image data sizes");

  std::vector<int64_t> sizes;
  for (size_t i = 0; i < image_data_sizes.size(); ++i)
    sizes.push_back(image_data_sizes[i]);

  ort_extensions::OrtxObjectPtr<OrtxRawImages> images;
  CheckResult(OrtxCreateRawImages(images.ToBeAssigned(), image_data.data(), sizes.data(), image_data.size()));

  return std::make_unique<Images>(std::move(images), image_data.size());
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

std::unique_ptr<Audios> LoadAudiosFromBuffers(std::span<const void*> audio_data,
                                              std::span<const size_t> audio_data_sizes) {
  if (audio_data.empty() || audio_data_sizes.empty())
    throw std::runtime_error("No audios provided");
  if (audio_data.size() != audio_data_sizes.size())
    throw std::runtime_error("Number of audio data buffers does not match the number of audio data sizes");

  std::vector<int64_t> sizes;
  for (size_t i = 0; i < audio_data_sizes.size(); ++i)
    sizes.push_back(audio_data_sizes[i]);

  ort_extensions::OrtxObjectPtr<OrtxRawAudios> audios;
  CheckResult(OrtxCreateRawAudios(audios.ToBeAssigned(), audio_data.data(), sizes.data(), audio_data.size()));

  return std::make_unique<Audios>(std::move(audios), audio_data.size());
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

template <>
std::unique_ptr<OrtValue> ProcessTensor<float, int64_t>(OrtxTensor* tensor, Ort::Allocator& allocator) {
  const float* tensor_data{};
  const int64_t* tensor_shape{};
  size_t tensor_num_dims;
  CheckResult(OrtxGetTensorData(tensor, reinterpret_cast<const void**>(&tensor_data),
                                &tensor_shape, &tensor_num_dims));
  const int64_t tensor_num_elements = std::accumulate(tensor_shape,
                                                      tensor_shape + tensor_num_dims,
                                                      1LL, std::multiplies<int64_t>());
  auto tensor_value = OrtValue::CreateTensor<int64_t>(allocator, std::span<int64_t>(const_cast<int64_t*>(tensor_shape), tensor_num_dims));
  std::transform(tensor_data, tensor_data + tensor_num_elements,
                 tensor_value->GetTensorMutableData<int64_t>(),
                 [](float value) { return static_cast<int64_t>(value + 0.5f); });
  return tensor_value;
}

template std::unique_ptr<OrtValue> ProcessTensor<float>(OrtxTensor* tensor, Ort::Allocator& allocator);
template std::unique_ptr<OrtValue> ProcessTensor<Ort::Float16_t>(OrtxTensor* tensor, Ort::Allocator& allocator);
template std::unique_ptr<OrtValue> ProcessTensor<int64_t>(OrtxTensor* tensor, Ort::Allocator& allocator);
template std::unique_ptr<OrtValue> ProcessTensor<bool>(OrtxTensor* tensor, Ort::Allocator& allocator);

template std::unique_ptr<OrtValue> ProcessTensor<int64_t, float>(OrtxTensor* tensor, Ort::Allocator& allocator);
template std::unique_ptr<OrtValue> ProcessTensor<int64_t, Ort::Float16_t>(OrtxTensor* tensor, Ort::Allocator& allocator);
template std::unique_ptr<OrtValue> ProcessTensor<float, int64_t>(OrtxTensor* tensor, Ort::Allocator& allocator);

}  // namespace Generators