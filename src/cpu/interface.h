// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Generators {

// Creates a fresh CPU DeviceInterface instance. Ownership is taken by OrtGlobals.
// `env` is the OrtGlobals env this interface belongs to (created before the interface and
// destroyed after it, per the reverse-order teardown), passed for signature consistency across EPs.
std::unique_ptr<DeviceInterface> CreateCpuInterface(OrtEnv& env);

}