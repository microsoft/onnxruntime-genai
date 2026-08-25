// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "regular_kv_cache.h"
#include "shared_kv_cache.h"

namespace Generators {

// Regular dynamic cache selected when the active device reports KV-cache quantization.
struct QuantizedRegularKeyValueCache : RegularKeyValueCache {
  using RegularKeyValueCache::RegularKeyValueCache;
};

// Shared-buffer cache selected when quantization and past/present sharing are both active.
struct QuantizedSharedKeyValueCache : SharedKeyValueCache {
  using SharedKeyValueCache::SharedKeyValueCache;
};

}  // namespace Generators

