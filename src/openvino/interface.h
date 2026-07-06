// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "../config.h"
namespace Generators {

// Creates a fresh OpenVINO DeviceInterface instance. Ownership is taken by OrtGlobals.
// `env` is the OrtGlobals env this interface belongs to (created before the interface and
// destroyed after it, per the reverse-order teardown), passed for signature consistency across EPs.
std::unique_ptr<DeviceInterface> CreateOpenVINOInterface(OrtEnv& env);

struct Model;
bool IsOpenVINOStatefulModel(const Model& model);

void OpenVINO_AppendProviderOptions(OrtSessionOptions& session_options,
                                    const Generators::Config& config,
                                    const Generators::Config::ProviderOptions& provider_options);

}  // namespace Generators