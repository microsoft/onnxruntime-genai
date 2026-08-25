// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "dynamic_kv_cache.h"

namespace Generators {

// Dynamic regular exposed past/present cache with separate past and present buffers.
struct RegularKeyValueCache : DynamicKeyValueCache {
  using DynamicKeyValueCache::DynamicKeyValueCache;
};

}  // namespace Generators

