// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "../generators.h"

namespace Generators::CUDAExecutionProvider {

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
