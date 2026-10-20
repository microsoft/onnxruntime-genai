// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace Generators {

// Name the EP library is registered under, and that its OrtEpDevice is discovered by.
constexpr const char* kAMDGPUExecutionProviderName = "AMDGPUExecutionProvider";

DeviceInterface* GetAMDGPUInterface();

// Null the AMDGPU interface singleton's allocator-derived state (allocator, memory info, pinned
// allocator, device id) in place, without destroying the singleton. Called per model so the next
// device init rebinds a fresh allocator.
void ResetAMDGPUInterfaceAllocatorState();

}  // namespace Generators
