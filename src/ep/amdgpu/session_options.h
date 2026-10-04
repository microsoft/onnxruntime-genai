// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "generator/generators.h"

namespace Generators::AMDGPUExecutionProvider {

DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& config,
                                         bool disable_graph_capture = false);

// Teardown counterpart to the registration done in AppendExecutionProvider. Releases the OrtEnv-shared
// allocators and unregisters the umbrella EP library, but only when genai itself registered it — a host
// that pre-registered the library keeps both for its own sessions. Called from Model::~Model via
// CloseAMDGPUInterface; ownership is tracked privately in session_options.cpp.
void ReleaseOwnedUmbrellaEp();

}  // namespace Generators::AMDGPUExecutionProvider
