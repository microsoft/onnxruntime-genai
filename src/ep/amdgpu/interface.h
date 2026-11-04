// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace Generators {

// Name the EP library is registered under, and that its OrtEpDevice is discovered by.
constexpr const char* kAMDGPUExecutionProviderName = "AMDGPUExecutionProvider";

DeviceInterface* GetAMDGPUInterface();

// Acquire a reference to the shared AMDGPU interface singleton, once per Model whose device is AMDGPU.
// Pairs with the release in CloseAMDGPUInterface so the singleton outlives every model using it (e.g.
// a target + draft decoder pair); the full teardown runs only on the last release.
void AcquireAMDGPUInterface();

// True while at least one AMDGPU Model is live. Lets the per-session allocator reset skip when a model
// is already alive, so it never destroys an allocator that model's buffers still reference.
bool HasLiveAMDGPUModel();

// Null the AMDGPU interface singleton's allocator-derived state (allocator, memory info, pinned
// allocator, device id) in place, without destroying the singleton. Called per session so the next
// device init (InitOrt / InitDeviceAllocators) rebinds a fresh allocator instead of reusing stale
// pointers from the prior session within the same model.
void ResetAMDGPUInterfaceAllocatorState();

// Full per-model teardown (mirrors CloseDmlInterface): destroys the interface singletons and
// unregisters the umbrella EP library so the plugin releases its process-global device, allocators,
// and outstanding allocation handles. Called from Model::~Model. The next model re-registers a fresh
// EP (and fresh device) via EnsureUmbrellaEpRegistered. Required so a GPU hang cannot leave stale
// device-backed allocation handles alive into the next model (dangling-handle access violation).
void CloseAMDGPUInterface();

}  // namespace Generators
