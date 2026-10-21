// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "models/io/vision_stage_io.h"

#include <algorithm>
#include <cstring>

namespace Generators {

void ApplyVisionRunOption(OrtRunOptions& options, const char* key, const char* value, size_t stage_index) {
  if (strcmp(key, "terminate_session") == 0) {
    if (strcmp(value, "0") == 0) {
      options.UnsetTerminate();
    } else if (strcmp(value, "1") == 0) {
      options.SetTerminate();
    } else {
      throw std::runtime_error(std::string("terminate_session key value unexpected: ") + value);
    }
  } else if (strcmp(key, "enable_profiling") == 0) {
#if ORT_API_VERSION >= 25
    if (strcmp(value, "0") == 0) {
      options.DisableProfiling();
    } else {
      constexpr const char* default_profile_prefix = "onnxruntime_run_profile";
      const char* prefix = strcmp(value, "1") == 0 ? default_profile_prefix : value;
      if (stage_index > 1) {
        throw std::runtime_error("Vision profiling requires an encoder or projector stage");
      }
      const auto stage_prefix = fs::path(std::string(prefix) + (stage_index == 0 ? "_encoder" : "_projector"));
      options.EnableProfiling(stage_prefix.c_str());
    }
#else
    (void)stage_index;
    throw std::runtime_error("enable_profiling requires ONNX Runtime 1.25 or later");
#endif
  } else {
    options.AddConfigEntry(key, value);
  }
}

std::unique_ptr<OrtRunOptions> CreateVisionStageRunOptions(const Config::Model::Vision& vision, size_t stage_index) {
  auto options = OrtRunOptions::Create();
  const auto& configured = vision.pipeline[stage_index].run_options.has_value()
                               ? vision.pipeline[stage_index].run_options
                               : vision.run_options;
  if (configured) {
    for (const auto& [key, value] : *configured) {
      ApplyVisionRunOption(*options, key.c_str(), value.c_str(), stage_index);
    }
  }
  return options;
}

std::unique_ptr<OrtValue> RunVisionEncoder(
    OrtSession& session, OrtRunOptions* run_options, const char* const* input_names,
    const OrtValue* const* inputs, size_t input_count, const char* output_name,
    DeviceInterface& encoder_device) {
  const auto output_names = session.GetOutputNames();
  const auto output_it = std::find(output_names.begin(), output_names.end(), output_name);
  if (output_it == output_names.end()) {
    throw std::runtime_error(std::string("Vision encoder output not found: ") + output_name);
  }

  const auto type_info = session.GetOutputTypeInfo(static_cast<size_t>(output_it - output_names.begin()));
  const auto& tensor_info = type_info->GetTensorTypeAndShapeInfo();
  const auto shape = tensor_info.GetShape();
  if (std::any_of(shape.begin(), shape.end(), [](int64_t dim) { return dim <= 0; })) {
    auto outputs = session.Run(run_options, input_names, inputs, input_count, &output_name, 1);
    return std::move(outputs.front());
  }

  auto output = OrtValue::CreateTensor(encoder_device.GetAllocator(), shape, tensor_info.GetElementType());
  OrtValue* bound_output = output.get();
  session.Run(run_options, input_names, inputs, input_count, &output_name, &bound_output, 1);
  if (bound_output != output.get()) {
    std::unique_ptr<OrtValue> unexpected_output(bound_output);
    throw std::runtime_error("Vision encoder replaced its preallocated output");
  }
  return output;
}

std::unique_ptr<OrtValue> StageVisionEncoderOutput(OrtValue& encoded, DeviceInterface& encoder_device,
                                                   DeviceInterface& projector_device) {
  if (encoded.GetTensorMemoryInfo().GetDeviceType() != OrtMemoryInfoDeviceType_CPU &&
      encoder_device.GetType() == DeviceType::CPU) {
    throw std::runtime_error("Vision encoder returned device memory without a matching device interface");
  }
  auto& source_device = DeviceForTensor(encoded, encoder_device);
  if (source_device.GetType() != DeviceType::CPU) {
    const auto& memory_info = encoded.GetTensorMemoryInfo();
    auto expected_info = source_device.GetMemoryInfo();
    if (memory_info.GetAllocatorName() != expected_info->GetAllocatorName() ||
        memory_info.GetDeviceId() != expected_info->GetDeviceId()) {
      throw std::runtime_error("Vision encoder output memory does not match the configured encoder device");
    }
  }
  if (SessionCanAccess(projector_device, source_device)) return nullptr;

  auto info = encoded.GetTensorTypeAndShapeInfo();
  auto& cpu = *GetDeviceInterface(DeviceType::CPU);
  auto staged = OrtValue::CreateTensor(cpu.GetAllocator(), info->GetShape(), info->GetElementType());
  if (info->GetElementCount() != 0) {
    ByteWrapTensor(cpu, *staged).CopyFrom(ByteWrapTensor(source_device, encoded));
  }
  return staged;
}

}  // namespace Generators
