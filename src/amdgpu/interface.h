// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace Generators {

// Name the EP library is registered under, and that its OrtEpDevice is discovered by.
constexpr const char* kAMDGPUExecutionProviderName = "AMDGPUExecutionProvider";

DeviceInterface* GetAMDGPUInterface();

}  // namespace Generators
