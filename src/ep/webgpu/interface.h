// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>

namespace Generators {

// Creates a fresh WebGPU DeviceInterface instance. Ownership is taken by OrtGlobals.
std::unique_ptr<DeviceInterface> CreateWebGPUInterface();

}  // namespace Generators