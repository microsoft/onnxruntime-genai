// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "../generators.h"

namespace Generators::NvTensorRtRtxExecutionProvider {

// Configures NvTensorRtRtx profile shapes, sets the CUDA stream, and registers
// the execution provider on |session_options| (V2 plugin path, then V1 fallback).
DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& config,
                                         bool disable_graph_capture = false);

}  // namespace Generators::NvTensorRtRtxExecutionProvider
