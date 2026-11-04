// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "models/model.h"
#include "models/parallel_utils.h"
#include "models/preprocessing/processor.h"

#include <array>
#include <fstream>
#include <limits>

namespace Generators {

namespace {

template <typename Read>
Images::Info ReadImageInfo(size_t bytes, Read&& read) {
  Images::Info info{bytes};
  std::array<uint8_t, 24> header{};
  if (!read(0, header)) return info;
  constexpr std::array<uint8_t, 8> png{137, 80, 78, 71, 13, 10, 26, 10};
  if (std::equal(png.begin(), png.end(), header.begin()) &&
      header[12] == 'I' && header[13] == 'H' && header[14] == 'D' && header[15] == 'R') {
    const auto dimension = [&](size_t offset) {
      return (uint32_t{header[offset]} << 24) | (uint32_t{header[offset + 1]} << 16) |
             (uint32_t{header[offset + 2]} << 8) | uint32_t{header[offset + 3]};
    };
    info.width = dimension(16);
    info.height = dimension(20);
    return info;
  }
  if (header[0] != 0xff || header[1] != 0xd8) return info;
  size_t offset = 2;
  while (offset < bytes) {
    std::array<uint8_t, 2> marker{};
    if (!read(offset, marker) || marker[0] != 0xff) return info;
    ++offset;
    while (marker[1] == 0xff) {
      std::array<uint8_t, 1> next{};
      if (!read(++offset, next)) return info;
      marker[1] = next[0];
    }
    ++offset;
    if (marker[1] == 0xda || marker[1] == 0xd9) return info;
    if (marker[1] == 0x01 || (marker[1] >= 0xd0 && marker[1] <= 0xd7)) continue;
    std::array<uint8_t, 2> length{};
    if (!read(offset, length)) return info;
    const size_t count = (size_t{length[0]} << 8) | length[1];
    if (count < 2 || count > bytes - offset) return info;
    const bool frame = marker[1] >= 0xc0 && marker[1] <= 0xcf &&
                       marker[1] != 0xc4 && marker[1] != 0xc8 && marker[1] != 0xcc;
    if (frame) {
      std::array<uint8_t, 5> dimensions{};
      if (count < 7 || !read(offset + 2, dimensions)) return info;
      info.height = (uint32_t{dimensions[1]} << 8) | dimensions[2];
      info.width = (uint32_t{dimensions[3]} << 8) | dimensions[4];
      return info;
    }
    offset += count;
  }
  return info;
}

}  // namespace

std::unique_ptr<Images> LoadImages(std::span<const char* const> image_paths) {
  if (image_paths.empty())
    throw std::runtime_error("No images provided");

  std::vector<std::vector<uint8_t>> encoded;
  std::vector<const void*> data;
  std::vector<size_t> sizes;
  encoded.reserve(image_paths.size());
  for (const char* image_path : image_paths) {
    if (!fs::path(image_path).exists()) {
      throw std::runtime_error("Image path does not exist: " + std::string(image_path));
    }
    auto input = fs::path(image_path).open(std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::streamoff>(input.tellg());
    if (bytes < 0 || static_cast<uintmax_t>(bytes) > std::numeric_limits<size_t>::max() ||
        bytes > std::numeric_limits<std::streamsize>::max())
      throw std::runtime_error("Cannot read image: " + std::string(image_path));
    encoded.emplace_back(static_cast<size_t>(bytes));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(encoded.back().data()), static_cast<std::streamsize>(bytes));
    if (!input) throw std::runtime_error("Cannot read image: " + std::string(image_path));
    data.push_back(encoded.back().data());
    sizes.push_back(encoded.back().size());
  }
  // Inspect the same bytes Extensions owns, not a second file read that could observe new contents.
  return LoadImagesFromBuffers(data, sizes);
}

std::unique_ptr<Images> LoadImagesFromBuffers(std::span<const void*> image_data,
                                              std::span<const size_t> image_data_sizes) {
  if (image_data.empty() || image_data_sizes.empty())
    throw std::runtime_error("No images provided");
  if (image_data.size() != image_data_sizes.size())
    throw std::runtime_error("Number of image data buffers does not match the number of image data sizes");

  std::vector<int64_t> sizes;
  std::vector<Images::Info> info;
  info.reserve(image_data.size());
  for (size_t i = 0; i < image_data_sizes.size(); ++i) {
    if (!image_data[i] || image_data_sizes[i] == 0 ||
        image_data_sizes[i] > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
      throw std::runtime_error("Image buffer must be nonempty and fit in int64.");
    }
    sizes.push_back(image_data_sizes[i]);
    info.push_back(ReadImageInfo(image_data_sizes[i], [&](size_t offset, std::span<uint8_t> data) {
      if (offset > image_data_sizes[i] || data.size() > image_data_sizes[i] - offset) return false;
      std::copy_n(static_cast<const uint8_t*>(image_data[i]) + offset, data.size(), data.data());
      return true;
    }));
  }

  ort_extensions::OrtxObjectPtr<OrtxRawImages> images;
  CheckResult(OrtxCreateRawImages(images.ToBeAssigned(), image_data.data(), sizes.data(), image_data.size()));

  return std::make_unique<Images>(std::move(images), image_data.size(), std::move(info));
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
  for (size_t i = 0; i < audio_data_sizes.size(); ++i) {
    if (audio_data_sizes[i] == 0)
      throw std::runtime_error("Audio buffer " + std::to_string(i) + " is empty.");
    sizes.push_back(audio_data_sizes[i]);
  }

  ort_extensions::OrtxObjectPtr<OrtxRawAudios> audios;
  CheckResult(OrtxCreateRawAudios(audios.ToBeAssigned(), audio_data.data(), sizes.data(), audio_data.size()));

  return std::make_unique<Audios>(std::move(audios), audio_data.size());
}

template <typename T>
std::unique_ptr<OrtValue> ProcessTensor(ThreadPool* thread_pool, OrtxTensor* tensor,
                                        Ort::Allocator& allocator) {
  const T* tensor_data{};
  const int64_t* tensor_shape{};
  size_t tensor_num_dims;
  CheckResult(OrtxGetTensorData(tensor, reinterpret_cast<const void**>(&tensor_data),
                                &tensor_shape, &tensor_num_dims));
  const int64_t tensor_num_elements = std::accumulate(tensor_shape,
                                                      tensor_shape + tensor_num_dims,
                                                      1LL, std::multiplies<int64_t>());
  auto tensor_value = OrtValue::CreateTensor<T>(allocator, std::span<int64_t>(const_cast<int64_t*>(tensor_shape), tensor_num_dims));
  if (thread_pool) {
    ParallelCopy(thread_pool, tensor_data, tensor_value->template GetTensorMutableData<T>(),
                 static_cast<size_t>(tensor_num_elements));
  } else {
    std::copy_n(tensor_data, tensor_num_elements, tensor_value->template GetTensorMutableData<T>());
  }
  return tensor_value;
}

template <>
std::unique_ptr<OrtValue> ProcessTensor<Ort::Float16_t>(ThreadPool*, OrtxTensor* tensor, Ort::Allocator& allocator) {
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
std::unique_ptr<OrtValue> ProcessTensor<Ort::BFloat16_t>(ThreadPool*, OrtxTensor* tensor, Ort::Allocator& allocator) {
  const float* tensor_data{};
  const int64_t* tensor_shape{};
  size_t tensor_num_dims;
  CheckResult(OrtxGetTensorData(tensor, reinterpret_cast<const void**>(&tensor_data),
                                &tensor_shape, &tensor_num_dims));
  const int64_t tensor_num_elements = std::accumulate(tensor_shape,
                                                      tensor_shape + tensor_num_dims,
                                                      1LL, std::multiplies<int64_t>());
  auto tensor_value = OrtValue::CreateTensor<Ort::BFloat16_t>(allocator, std::span<int64_t>(const_cast<int64_t*>(tensor_shape), tensor_num_dims));
  auto tensor_value_fp32 = OrtValue::CreateTensor<float>(
      allocator.GetInfo(),
      std::span<float>(const_cast<float*>(tensor_data), tensor_num_elements),
      std::span<int64_t>(const_cast<int64_t*>(tensor_shape), tensor_num_dims));
  auto p_device = GetDeviceInterface(DeviceType::CPU);
  Cast(*tensor_value_fp32, tensor_value, *p_device, Ort::TypeToTensorType<Ort::BFloat16_t>);
  return tensor_value;
}

template <>
std::unique_ptr<OrtValue> ProcessTensor<int64_t, float>(ThreadPool* thread_pool, OrtxTensor* tensor, Ort::Allocator& allocator) {
  const int64_t* tensor_data{};
  const int64_t* tensor_shape{};
  size_t tensor_num_dims;
  CheckResult(OrtxGetTensorData(tensor, reinterpret_cast<const void**>(&tensor_data),
                                &tensor_shape, &tensor_num_dims));
  const int64_t tensor_num_elements = std::accumulate(tensor_shape,
                                                      tensor_shape + tensor_num_dims,
                                                      1LL, std::multiplies<int64_t>());
  auto tensor_value = OrtValue::CreateTensor<float>(allocator, std::span<int64_t>(const_cast<int64_t*>(tensor_shape), tensor_num_dims));
  ParallelTransform(thread_pool, tensor_data, tensor_value->GetTensorMutableData<float>(),
                    static_cast<size_t>(tensor_num_elements), 2.0,
                    [](int64_t value) { return static_cast<float>(value); });
  return tensor_value;
}

template <>
std::unique_ptr<OrtValue> ProcessTensor<int64_t, Ort::Float16_t>(ThreadPool* thread_pool, OrtxTensor* tensor, Ort::Allocator& allocator) {
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
  ParallelTransform(thread_pool, tensor_data, tensor_value_fp32->GetTensorMutableData<float>(),
                    static_cast<size_t>(tensor_num_elements), 2.0,
                    [](int64_t value) { return static_cast<float>(value); });
  auto p_device = GetDeviceInterface(DeviceType::CPU);
  Cast(*tensor_value_fp32, tensor_value, *p_device, Ort::TypeToTensorType<Ort::Float16_t>);
  return tensor_value;
}

template <>
std::unique_ptr<OrtValue> ProcessTensor<int64_t, Ort::BFloat16_t>(ThreadPool* thread_pool, OrtxTensor* tensor, Ort::Allocator& allocator) {
  const int64_t* tensor_data{};
  const int64_t* tensor_shape{};
  size_t tensor_num_dims;
  CheckResult(OrtxGetTensorData(tensor, reinterpret_cast<const void**>(&tensor_data),
                                &tensor_shape, &tensor_num_dims));
  const int64_t tensor_num_elements = std::accumulate(tensor_shape,
                                                      tensor_shape + tensor_num_dims,
                                                      1LL, std::multiplies<int64_t>());
  auto tensor_value = OrtValue::CreateTensor<Ort::BFloat16_t>(allocator, std::span<int64_t>(const_cast<int64_t*>(tensor_shape), tensor_num_dims));
  auto tensor_value_fp32 = OrtValue::CreateTensor<float>(allocator, std::span<int64_t>(const_cast<int64_t*>(tensor_shape), tensor_num_dims));
  ParallelTransform(thread_pool, tensor_data, tensor_value_fp32->GetTensorMutableData<float>(),
                    static_cast<size_t>(tensor_num_elements), 2.0,
                    [](int64_t value) { return static_cast<float>(value); });
  auto p_device = GetDeviceInterface(DeviceType::CPU);
  Cast(*tensor_value_fp32, tensor_value, *p_device, Ort::TypeToTensorType<Ort::BFloat16_t>);
  return tensor_value;
}

template <>
std::unique_ptr<OrtValue> ProcessTensor<float, int64_t>(ThreadPool* thread_pool, OrtxTensor* tensor, Ort::Allocator& allocator) {
  const float* tensor_data{};
  const int64_t* tensor_shape{};
  size_t tensor_num_dims;
  CheckResult(OrtxGetTensorData(tensor, reinterpret_cast<const void**>(&tensor_data),
                                &tensor_shape, &tensor_num_dims));
  const int64_t tensor_num_elements = std::accumulate(tensor_shape,
                                                      tensor_shape + tensor_num_dims,
                                                      1LL, std::multiplies<int64_t>());
  auto tensor_value = OrtValue::CreateTensor<int64_t>(allocator, std::span<int64_t>(const_cast<int64_t*>(tensor_shape), tensor_num_dims));
  ParallelTransform(thread_pool, tensor_data, tensor_value->GetTensorMutableData<int64_t>(),
                    static_cast<size_t>(tensor_num_elements), 2.0,
                    [](float value) { return static_cast<int64_t>(value + 0.5f); });
  return tensor_value;
}

template std::unique_ptr<OrtValue> ProcessTensor<float>(ThreadPool*, OrtxTensor*, Ort::Allocator&);
template std::unique_ptr<OrtValue> ProcessTensor<int32_t>(ThreadPool*, OrtxTensor*, Ort::Allocator&);
template std::unique_ptr<OrtValue> ProcessTensor<int64_t>(ThreadPool*, OrtxTensor*, Ort::Allocator&);
template std::unique_ptr<OrtValue> ProcessTensor<bool>(ThreadPool*, OrtxTensor*, Ort::Allocator&);

void EmplaceProcessedTensor(ThreadPool* thread_pool, NamedTensors& tensors, std::string_view name,
                            OrtxTensor* tensor, ONNXTensorElementDataType type,
                            Ort::Allocator& allocator) {
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    tensors.emplace(std::string(name), std::make_shared<Tensor>(ProcessTensor<float>(thread_pool, tensor, allocator)));
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
    tensors.emplace(std::string(name), std::make_shared<Tensor>(ProcessTensor<Ort::BFloat16_t>(thread_pool, tensor, allocator)));
  } else {
    tensors.emplace(std::string(name), std::make_shared<Tensor>(ProcessTensor<Ort::Float16_t>(thread_pool, tensor, allocator)));
  }
}

}  // namespace Generators