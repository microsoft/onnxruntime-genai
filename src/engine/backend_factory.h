// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "engine.h"

namespace Generators {

struct EngineBackendBundle {
  EngineDependencies dependencies;
  BackendCapabilities capabilities;
};

// Resources/profile selection belongs to Model initialization. This factory only assembles
// collaborators for that already-selected model and leases its execution context.
EngineBackendBundle CreateEngineBackend(std::shared_ptr<Model> model);

// Compatibility for injected collaborators. Explicit descriptors are checked against the actual
// execution interfaces; old test/custom bundles can omit one and retain their existing behavior.
BackendCapabilities ResolveBackendCapabilities(
    const Model& model, const CacheManager& cache_manager, const Scheduler& scheduler,
    const ModelExecutor& executor, std::optional<BackendCapabilities> declared = std::nullopt);

}  // namespace Generators
