// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace Generators {

// Name the EP library is registered under, and that its OrtEpDevice is discovered by.
constexpr const char* kAMDGPUExecutionProviderName = "AMDGPUExecutionProvider";

DeviceInterface* GetAMDGPUInterface();

// Full per-model teardown (mirrors CloseDmlInterface): destroys the interface singletons and
// unregisters the umbrella EP library so the plugin releases its process-global device, allocators,
// and outstanding allocation handles. Called from Model::~Model. The next model re-registers a fresh
// EP (and fresh device) via EnsureUmbrellaEpRegistered. Required so a GPU hang cannot leave stale
// device-backed allocation handles alive into the next model (dangling-handle access violation).
void CloseAMDGPUInterface();

}  // namespace Generators
