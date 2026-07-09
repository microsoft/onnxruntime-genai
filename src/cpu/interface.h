// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Generators {

// Creates a fresh CPU DeviceInterface instance. Ownership is taken by OrtGlobals.
std::unique_ptr<DeviceInterface> CreateCpuInterface();

}