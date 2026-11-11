// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "shared_kv_cache.h"

namespace Generators {

// Shared past/present cache variant where the active EP compacts sliding-window layers
// internally (for example CPU/CUDA GQA with sliding_window_cache=1).
struct EpManagedSlidingKeyValueCache : SharedKeyValueCache {
  using SharedKeyValueCache::SharedKeyValueCache;
};

}  // namespace Generators
