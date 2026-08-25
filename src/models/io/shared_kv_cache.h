// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "static_kv_cache.h"

namespace Generators {

struct SharedKeyValueCache : DefaultKeyValueCacheBase {
  using DefaultKeyValueCacheBase::DefaultKeyValueCacheBase;

  void Update(DeviceSpan<int32_t> beam_indices, int total_length) override;
  void RewindTo(size_t index) override;
};

}  // namespace Generators
