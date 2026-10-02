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

// Teardown counterpart to the umbrella-EP registration done in AppendExecutionProvider. Unregisters
// the umbrella EP library from the process-global OrtEnv, but ONLY when genai itself registered it
// (i.e. no host had pre-registered). When a host — e.g. the ModelBench harness — registered the
// library at startup, this is a no-op: the host's own (non-OGA) sessions share that single
// process-global registration and must not lose the EP when a genai model is destroyed. Called from
// Model::~Model via CloseAMDGPUInterface. Ownership is tracked privately inside session_options.cpp.
void UnregisterUmbrellaEpIfOwned();

}  // namespace Generators::AMDGPUExecutionProvider
