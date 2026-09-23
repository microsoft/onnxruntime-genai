// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "config.h"

namespace Generators {

struct DeviceInterface;

void ApplyRuntimeProfileForSelectedDevice(Config& config, DeviceInterface& device);

}  // namespace Generators