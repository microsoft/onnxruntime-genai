// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <stdexcept>

#include "generator/generators.h"

namespace Generators::CUDAExecutionProvider {

// Converts generic CUDA allocator options to provider-bridge options. Options that require the
// registered V2 plugin are rejected before the provider is appended.
inline Config::ProviderOptions NormalizeProviderBridgeOptions(
    const Config::ProviderOptions& provider_options) {
  Config::ProviderOptions normalized = provider_options;
  normalized.options.clear();

  const bool has_gpu_mem_limit = std::any_of(
      provider_options.options.begin(), provider_options.options.end(),
      [](const auto& option) { return option.first == "gpu_mem_limit"; });
  for (const auto& option : provider_options.options) {
    if (option.first == "max_mem") {
      if (has_gpu_mem_limit) {
        throw std::invalid_argument(
            "CUDA provider options max_mem and gpu_mem_limit cannot both be set.");
      }
      normalized.options.emplace_back("gpu_mem_limit", option.second);
    } else if (option.first == "initial_chunk_size_bytes" ||
               option.first == "max_dead_bytes_per_chunk" ||
               option.first == "initial_growth_chunk_size_bytes" ||
               option.first.starts_with("arena.")) {
      throw std::runtime_error(
          "Fine-grained CUDA arena options require a registered CUDA V2 plugin execution "
          "provider.");
    } else {
      normalized.options.emplace_back(option);
    }
  }
  return normalized;
}

// Writes the CUDA compute stream pointer (as a stringified integer) into a
// session config entry keyed by |config_key|. This is used by both the CUDA
// and NvTensorRtRtx providers so that the EP can share the same stream.
void AddCudaStreamConfig(OrtSessionOptions& session_options, DeviceInterface* device,
                         const std::string& config_key = "user_compute_stream");

// Registers the CUDA execution provider on |session_options|. Tries the V2
// plugin path first; falls back to the provider-bridge (CUDA V2 options) path.
DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& config,
                                         bool disable_graph_capture = false);

}  // namespace Generators::CUDAExecutionProvider
