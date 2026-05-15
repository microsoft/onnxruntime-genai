// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <optional>

#include "interface.h"
#include "session_options.h"
#include "../models/session_options.h"

namespace Generators::QNNExecutionProvider {

static bool IsQNNGPUSharedAllocatorAvailable() {
  // GPU shared memory allocator availability depends on both:
  //   a. The graphics drivers installed on the system
  //   b. The QNN EP package loaded by onnxruntime-genai
  // If either dependency is out-of-date, the allocator will not be available.
  // To check for support, we try to execute a single-node model through QNN
  // with shared memory allocations and cache the result.
  static std::optional<bool> cached_result = std::nullopt;
  if (!cached_result) {
    // Trivial ONNX model to do a scalar increment (X + 1 = Y)
    const uint8_t trivial_add_model[] = {
        0x08, 0x07, 0x12, 0x0f, 0x61, 0x64, 0x64, 0x5f, 0x6f, 0x6e, 0x65, 0x5f,
        0x65, 0x78, 0x61, 0x6d, 0x70, 0x6c, 0x65, 0x3a, 0x5a, 0x0a, 0x18, 0x0a,
        0x01, 0x58, 0x0a, 0x03, 0x6f, 0x6e, 0x65, 0x12, 0x01, 0x59, 0x1a, 0x06,
        0x41, 0x64, 0x64, 0x4f, 0x6e, 0x65, 0x22, 0x03, 0x41, 0x64, 0x64, 0x12,
        0x0b, 0x41, 0x64, 0x64, 0x4f, 0x6e, 0x65, 0x47, 0x72, 0x61, 0x70, 0x68,
        0x2a, 0x0f, 0x08, 0x01, 0x10, 0x01, 0x22, 0x04, 0x00, 0x00, 0x80, 0x3f,
        0x42, 0x03, 0x6f, 0x6e, 0x65, 0x5a, 0x0f, 0x0a, 0x01, 0x58, 0x12, 0x0a,
        0x0a, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x01, 0x62, 0x0f,
        0x0a, 0x01, 0x59, 0x12, 0x0a, 0x0a, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0a,
        0x02, 0x08, 0x01, 0x42, 0x04, 0x0a, 0x00, 0x10, 0x0d};

    auto session_options = OrtSessionOptions::Create();
    Config::ProviderOptions provider_options = Config::ProviderOptions{
        "QNN",
        {},
        Config::DeviceFilteringOptions{OrtHardwareDeviceType_GPU}
    };

    if (!AppendExecutionProviderV2(*session_options, provider_options,
                                   DeviceType::QNN, "QNNExecutionProvider")) {
      AppendExecutionProviderV1(*session_options, provider_options);
    }

    session_options->SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);

    const auto session = OrtSession::Create(GetOrtEnv(), trivial_add_model, sizeof(trivial_add_model), session_options.get());

    try {
      const auto memory_info = OrtMemoryInfo::Create("QnnHtpShared",
                                                     OrtAllocatorType::OrtDeviceAllocator,
                                                     0,
                                                     OrtMemType::OrtMemTypeDefault);
      const auto allocator = Ort::Allocator::Create(*session, *memory_info);
      if (!allocator) {
        cached_result = false;
        return false;
      }

      auto allocator_free = [&allocator](void* p) { allocator->Free(p); };

      float* input_data_ptr = static_cast<float*>(allocator->Alloc(sizeof(float)));
      if (input_data_ptr == nullptr) {
        cached_result = false;
        return false;
      }
      std::unique_ptr<float, decltype(allocator_free)> input_data(input_data_ptr, allocator_free);

      float* output_data_ptr = static_cast<float*>(allocator->Alloc(sizeof(float)));
      if (output_data_ptr == nullptr) {
        cached_result = false;
        return false;
      }
      std::unique_ptr<float, decltype(allocator_free)> output_data(output_data_ptr, allocator_free);

      *input_data = 1.0f;
      std::array<int64_t, 1> shape = {1};

      std::span<float> input_data_span(input_data.get(), 1);
      std::span<float> output_data_span(output_data.get(), 1);
      std::span<const int64_t> shape_span(shape.data(), shape.size());

      auto x = OrtValue::CreateTensor(*memory_info, input_data_span, shape_span);
      auto y = OrtValue::CreateTensor(*memory_info, output_data_span, shape_span);

      auto binding = OrtIoBinding::Create(*session);
      binding->BindInput("X", *x);
      binding->BindOutput("Y", *y);
      session->Run(OrtRunOptions::Create().get(), *binding);

      cached_result = true;
    } catch (const Ort::Exception&) {
      cached_result = false;
    }
  }

  return cached_result.value();
}

DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& config,
                                         bool /*disable_graph_capture*/) {
  DeviceInterface* device = nullptr;
  session_options.AddConfigEntry("ep.share_ep_contexts", "1");
  if (Generators::IsQNNGPUBackend(config)) {
    if (IsQNNGPUSharedAllocatorAvailable()) {
      device = GetDeviceInterface(DeviceType::QNN);
    } else {
      Log("warning",
          "QNN GPU shared memory allocator is not available!"
          " Falling back to CPU allocations for the KV cache. This will reduce performance."
          " To avoid this, try updating the QNN EP package and the graphics drivers on your system.");
    }
  } else {
    // Shared allocator only exposed for NPU if enabled via provider option
    if (const auto opt_it = std::find_if(
            provider_options.options.begin(), provider_options.options.end(),
            [](const auto& pair) { return pair.first == "enable_htp_shared_memory_allocator"; });
        opt_it != provider_options.options.end() && opt_it->second == "1") {
      device = GetDeviceInterface(DeviceType::QNN);
    }
  }
  // is_primary_session_options is set to false because the device is set based on
  // the presence of the "enable_htp_shared_memory_allocator" option,
  // not based on whether this is the primary session options or not.
  if (!AppendExecutionProviderV2(session_options, provider_options,
                                 DeviceType::QNN, "QNNExecutionProvider")) {
    AppendExecutionProviderV1(session_options, provider_options);
  }

  return device;
}

}  // namespace Generators::QNNExecutionProvider
