// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>

namespace Generators {

struct DeviceInterface;
struct StateUpdateReplayDesc;

// Creates a fresh WebGPU DeviceInterface instance. Ownership is taken by OrtGlobals.
std::unique_ptr<DeviceInterface> CreateWebGPUInterface();

namespace WebGPU {
// Internal opt-in helper; DeviceInterface::ReplayStateUpdates still uses CPU staging.
void RunGatedDeltaNetStateReplay(const StateUpdateReplayDesc& descriptor);
}  // namespace WebGPU

}  // namespace Generators