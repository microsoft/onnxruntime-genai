// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lora_adapter.h"

#include "../generators.h"
#include "../span.h"
#include "../flatbuffers.h"
#include "../flatbuffers/lora_format_version.h"
#include "../flatbuffers/flatbuffers_utils.h"
#include "model.h"
#include "onnxruntime_api.h"
#include "utils.h"

#include <fstream>

namespace Generators {
namespace {
uint64_t empty_input_buf[] = {0xdeadbeefbeefdead};
}  // namespace

namespace details {

std::shared_ptr<OrtValue> CreateEmptyInput(const OrtValue& original) {
  auto type_and_shape = original.GetTensorTypeAndShapeInfo();
  auto shape = type_and_shape->GetShape();

  // Modify shape
  const auto num_dims = shape.size();
  if (num_dims < 2) {
    throw std::runtime_error("Shape must have at least 2 dimensions");
  }

  // Zero out lora_r dim
  const size_t last_dim = shape[num_dims - 1];
  const size_t penal_dim = shape[num_dims - 2];
  if (last_dim < penal_dim) {
    shape[num_dims - 1] = 0;
  } else {
    shape[num_dims - 2] = 0;
  }

  const auto& mem_info = original.GetTensorMemoryInfo();
  return OrtValue::CreateTensor(mem_info, &empty_input_buf, 0, shape, type_and_shape->GetElementType());
}

void BinaryFormatHolder::Load(const std::string& file_name) {
  std::ifstream is(file_name, std::ios::binary | std::ios::ate);
  if (!is.good()) {
    throw std::runtime_error("Error opening flatbuffers file: " + file_name);
  }

  auto const file_size = static_cast<size_t>(is.tellg());
  is.seekg(0, std::ios::beg);

  buffer_.resize(file_size);
  is.read(reinterpret_cast<char*>(buffer_.data()), file_size);

  if (!is.good()) {
    throw std::runtime_error("Error reading flatbuffers file: " + file_name);
  }

  is.close();

  lora_parameters::utils::IsGenAiLoraFormatModelBytes(reinterpret_cast<const uint8_t*>(buffer_.data()), file_size);
  flatbuffers::Verifier verifier(buffer_.data(), file_size);
  lora_parameters::VerifyParametersBuffer(verifier);

  parameters_ = lora_parameters::GetParameters(buffer_.data());
  lora_parameters::IsLoraFormatVersionSupported(parameters_->version());
}

LoraParam::LoraParam(std::string name, const std::shared_ptr<Tensor>& parameter) : name_(std::move(name)) {
  // Create a duplicate of the ort_value over the same user supplied buffer
  // we want ort_value to be owned by a shared_ptr so it can be shared
  // We could still the unique_ptr from the original tensor, but that would not
  // be a good practice and the internal ORT OrtValue copy constructor is not public.
  ort_user_supplied_value_ = DuplicateOrtValue(*parameter->ort_tensor_);
}

LoraParam::LoraParam(std::string name, std::shared_ptr<OrtValue> ort_value)
    : name_(std::move(name)), ort_user_supplied_value_(std::move(ort_value)) {
}

void LoraAdapter::MakeDeviceCopyIfNeeded(const Model& model, LoraParam& param) {
  // Check if the target device is not CPU
  // XXX: Adjust for caching when implemented
  if (!param.ort_device_value_ && model.device_type_ != DeviceType::CPU) {
    // Check if the user has already supplied his buffers on the target device
    auto& source_value = param.ort_user_supplied_value_;
    const auto& mem_info = source_value->GetTensorMemoryInfo();
    auto src_device_type = mem_info.GetDeviceType();

    if ((model.device_type_ == DeviceType::CUDA &&
         src_device_type == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU)) {
      // Re-use what user has supplied on GPU
      param.ort_device_value_ = param.ort_user_supplied_value_;
    } else if (src_device_type != OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU) {
      // XXX: Can the user supply buffers already on DML?
      throw std::runtime_error("Lora parameter buffers are on unsupported device: " +
                               std::to_string(static_cast<int>(model.device_type_)));
    } else {
      param.ort_device_value_ = CopyToDevice(*source_value, model);
    }
  }
}

void LoraAdapter::LoadParametersFromFlatBuffer(const std::string& file_name) {
  format_holder_.Load(file_name);

  const auto* fbs_parameters = format_holder_.GetParameters();
  std::vector<LoraParam> parameters;
  parameters.reserve(fbs_parameters->parameters()->size());

  for (const auto* fbs_tensor : *fbs_parameters->parameters()) {
    auto [name, ort_value] = lora_parameters::utils::CreateOrtValueOverFlatBufferLoraParameter(*fbs_tensor);
    parameters.emplace_back(std::move(name), std::move(ort_value));
  }
  parameters_.swap(parameters);
}

}  // namespace details

LoraAdapterManagement::LoraAdapterManagement(const Model* model) : model_(model) {}

void LoraAdapterManagement::CreateAdapter(const std::string& adapter_name) {
  std::unique_lock lock(mutex_);
  auto hit = adapters_.find(adapter_name);
  if (hit != adapters_.end()) {
    throw std::runtime_error("Adapter: " + adapter_name + " already exist");
  }
  auto& adapter = adapters_[adapter_name];
  adapter.SetName(adapter_name);
}

void LoraAdapterManagement::LoadAdaptersFromConfig(const fs::path& model_path, const Config& config) {

  for (const auto& [adapter_name, file_name] : config.lora_adapters.adapters) {
    auto hit = adapters_.find(adapter_name);
    if (hit != adapters_.end()) {
      throw std::runtime_error("Adapter: " + adapter_name + " already exist");
    }
    auto& adapter = adapters_[adapter_name];
    adapter.SetName(adapter_name);
    auto full_path = model_path / file_name;
    adapter.LoadParametersFromFlatBuffer(full_path.string());
  }
}

void LoraAdapterManagement::AddParameter(const std::string& adapter_name, std::string param_name,
                                         std::shared_ptr<OrtValue> ort_value) {
  std::shared_lock lock(mutex_);
  auto hit = adapters_.find(adapter_name);
  if (hit == adapters_.end()) {
    throw std::runtime_error("Adapter: " + adapter_name + " does not exist");
  }

  auto& adapter = hit->second;

  adapter.AddParameter(std::move(param_name), std::move(ort_value));
}

}  // namespace Generators