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

// Teardown counterpart to the registration in AppendExecutionProvider: releases the OrtEnv-shared
// allocators and unregisters the umbrella EP library, only when genai owns the registration. Called from
// Model::~Model (via CloseAMDGPUInterface) and from ~OrtGlobals at shutdown. env + flag are passed by
// reference, not fetched via GetOrtEnv()/GetOrtGlobals(), because ~OrtGlobals holds g_ort_globals_mutex
// and re-entering those accessors would deadlock. Clears owns_registration only on a successful
// unregister (a failure leaves it true to retry).
void ReleaseOwnedUmbrellaEp(OrtEnv& env, bool& owns_registration);

}  // namespace Generators::AMDGPUExecutionProvider
