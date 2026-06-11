// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <string>

#include "interface.h"
#include "session_options.h"
#include "../models/model.h"
#include "../models/session_options.h"

namespace Generators::QNNExecutionProvider {

static bool IsAllocatorAvailable(const Config& config) {
  // This serves primarily to check for Dx12 shared memory allocator support, which depends on both:
  //   a. The graphics drivers installed on the system
  //   b. The QNN EP package loaded by onnxruntime-genai
  // If either dependency is out-of-date, the allocator will not be available.
  auto session_options = OrtSessionOptions::Create();
  Config::ProviderOptions provider_options = GetQNNInterface()->GetProviderOptionsForAllocatorSession(config);

  if (!AppendExecutionProviderV2(*session_options, provider_options,
                                 DeviceType::QNN, "QNNExecutionProvider")) {
    AppendExecutionProviderV1(*session_options, provider_options);
  }

  session_options->SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);

  const auto session = OrtSession::Create(GetOrtEnv(),
                                          Generators::GetTrivialModel().data(),
                                          Generators::GetTrivialModel().size(),
                                          session_options.get());

  try {
    const auto memory_info = GetQNNInterface()->GetMemoryInfo();
    const auto allocator = Ort::Allocator::Create(*session, *memory_info);
    return allocator != nullptr;
  } catch (const Ort::Exception&) {
    return false;
  }
}

DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& config,
                                         bool /*disable_graph_capture*/) {
  DeviceInterface* device = nullptr;
  session_options.AddConfigEntry("ep.share_ep_contexts", "1");

  bool allocator_requested = false;
  if (const auto opt_it = std::find_if(
          provider_options.options.begin(), provider_options.options.end(),
          [](const auto& pair) { return pair.first == "enable_htp_shared_memory_allocator" || pair.first == "enable_dx12_shared_memory_allocator"; });
      opt_it != provider_options.options.end() && opt_it->second == "1") {
    allocator_requested = true;
  }

  static const bool allocator_available = IsAllocatorAvailable(config);

  if (allocator_available) {
    device = GetDeviceInterface(DeviceType::QNN);
  } else if (Generators::IsQNNGPUBackend(config) && allocator_requested && g_log.enabled) {
    // Only warn if allocator requested but not available
    // Warn for GPU and not HTP, as the gfx drivers apply only to GPU
    Log("warning",
        "Shared memory allocator for QNN GPU is not available!"
        " Falling back to CPU allocations for the KV cache. This will reduce performance."
        " To avoid this, try updating the QNN EP package and the graphics drivers on your system.");
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
