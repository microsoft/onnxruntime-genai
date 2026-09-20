// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "generator/generators.h"
#include "cpu_embedding.h"

namespace Generators {

CpuEmbedding::CpuEmbedding(Model& model, OrtEnv& env) : config_{model.config_->model.embedding} {
  auto options = OrtSessionOptions::Create();
  Config::SessionOptions cpu_options = config_.session_options.value_or(Config::SessionOptions{});
  if (!cpu_options.provider_options.empty() || !cpu_options.providers.empty()) {
    throw std::runtime_error("Engine model.embedding must use CPU (no provider_options).");
  }
  // Tiny decode lookups do not benefit from a thread pool. Explicit config can override this.
  if (!cpu_options.intra_op_num_threads) cpu_options.intra_op_num_threads = 1;
  model.CreateSessionOptionsFromConfig(cpu_options, *options, false, true, false, false);
  session_ = model.CreateSession(env, config_.filename, options.get());
  SessionInfo info;
  info.Add(*session_);
  if (session_->GetInputNames().size() != 1 || session_->GetOutputNames().size() != 1 ||
      !info.HasInput(config_.inputs.input_ids) || !info.HasOutput(config_.outputs.embeddings) ||
      info.GetInputDataType(config_.inputs.input_ids) != Ort::TypeToTensorType<int64_t> ||
      info.GetInputShape(config_.inputs.input_ids) != std::vector<int64_t>{-1}) {
    throw std::runtime_error("Engine CPU embedding requires one dynamic int64[num_tokens] input and one output.");
  }
  const auto shape = info.GetOutputShape(config_.outputs.embeddings);
  type_ = info.GetOutputDataType(config_.outputs.embeddings);
  if (shape.size() != 2 || shape[0] >= 0 || shape[1] <= 0 ||
      (type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 && type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 &&
       type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
    throw std::runtime_error("Engine CPU embedding output must be floating point [num_tokens, hidden_size].");
  }
  hidden_size_ = shape[1];
  if (hidden_size_ != model.config_->model.decoder.hidden_size) {
    throw std::runtime_error("CPU embedding width must match model.decoder.hidden_size.");
  }
  run_options_ = OrtRunOptions::Create();
  if (config_.run_options) {
    for (const auto& [key, value] : *config_.run_options) {
      run_options_->AddConfigEntry(key.c_str(), value.c_str());
    }
  }
}

void CpuEmbedding::ValidateConsumer(const SessionInfo& info, const std::string& name) const {
  if (!info.HasInput(name) || info.GetInputDataType(name) != type_ ||
      info.GetInputShape(name) != std::vector<int64_t>{-1, hidden_size_}) {
    throw std::runtime_error("CPU embedding output does not match consumer input '" + name + "'.");
  }
}

void CpuEmbedding::Run(std::span<const int64_t> ids, Tensor& output) const {
  const std::array<int64_t, 1> id_shape{static_cast<int64_t>(ids.size())};
  const std::vector<int64_t> shape{static_cast<int64_t>(ids.size()), hidden_size_};
  if (output.GetType() != type_ || output.GetShape() != shape) {
    throw std::runtime_error("CPU embedding destination has an incompatible shape or type.");
  }
  const auto& cpu_memory = GetDeviceInterface(DeviceType::CPU)->GetAllocator().GetInfo();
  auto input = OrtValue::CreateTensor(cpu_memory, const_cast<int64_t*>(ids.data()), ids.size_bytes(),
                                      id_shape, Ort::TypeToTensorType<int64_t>);
  auto bytes = output.GetByteSpan();
  auto host = bytes.CpuSpan();
  auto output_memory = output.p_device_->GetType() == DeviceType::CUDA
                           ? OrtMemoryInfo::Create("CudaPinned", OrtDeviceAllocator,
                                                   output.p_device_->GetMemoryInfo()->GetDeviceId(), OrtMemTypeCPUOutput)
                           : OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  auto result = OrtValue::CreateTensor(*output_memory, host.data(), host.size_bytes(), shape, type_);
  const char* input_name = config_.inputs.input_ids.c_str();
  const char* output_name = config_.outputs.embeddings.c_str();
  OrtValue* input_value = input.get();
  OrtValue* output_value = result.get();
  session_->Run(run_options_.get(), &input_name, &input_value, 1, &output_name, &output_value, 1);
  // The pinned mirror has the destination buffer's lifetime, so this asynchronous copy can
  // remain ordered with the consuming CUDA session without entering graph capture.
  if (output.p_device_->GetType() == DeviceType::CUDA) {
    bytes.CopyCpuToDevice();
  } else if (output.p_device_->GetType() != DeviceType::CPU) {
    bytes.CopyFromCpu(host);
  }
}

}  // namespace Generators
