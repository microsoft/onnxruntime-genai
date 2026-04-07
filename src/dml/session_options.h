// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "../generators.h"

namespace Generators::DMLExecutionProvider {

// Initialises the DML interface (if not already done), optionally enables graph
// capture, and registers the DirectML execution provider on |session_options|.
DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& config,
                                         bool disable_graph_capture = false);

}  // namespace Generators::DMLExecutionProvider
