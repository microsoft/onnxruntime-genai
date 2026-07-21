// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace Generators {

DeviceInterface* GetAMDGPUInterface();

// Inputs-only interface backed by the host-accessible allocator. Decode inputs allocated
// here are CPU-writable and GPU-readable, so per-step updates happen in place with no
// roundtrip. KV and scoring keep the default interface. Null if no host-accessible allocator.
DeviceInterface* GetAMDGPUPinnedInputsInterface();

}  // namespace Generators
