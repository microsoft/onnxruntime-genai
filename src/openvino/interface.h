// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "../config.h"
namespace Generators {

DeviceInterface* GetOpenVINOInterface();

struct Model;
bool IsOpenVINOStatefulModel(const Model& model);

void OpenVINO_AppendProviderOptions(OrtSessionOptions &session_options,
                                    const Generators::Config &config,
                                    const Generators::Config::ProviderOptions& provider_options);

}  // namespace Generators