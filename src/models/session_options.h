// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "../generators.h"
#include "model.h"

namespace Generators {

namespace CUDAExecutionProvider {

void AppendExecutionProvider(
    OrtSessionOptions& session_options,
    const Config::ProviderOptions& provider_options,
    bool is_primary_session_options,
    DeviceInterface*& p_device,
    std::unique_ptr<OrtArenaCfg>& arena_cfg);

}

namespace NvTensorRtRtxExecutionProvider {

/**
 * @brief Creates profile shapes for NvTensorRtRtx execution provider optimization.
 *
 * This function generates profiles for TensorRT execution provider optimization.
 * If multi-profile is enabled, it creates separate profiles for context and generation phases.
 * If multi-profile is disabled, it creates a single profile with simple shapes.
 *
 */
void ConfigureProfile(const Config& config, OrtSessionOptions& session_options,
                      bool is_multi_profile_enabled);

}  // namespace NvTensorRtRtxExecutionProvider

}  // namespace Generators
