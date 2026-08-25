// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_kv_cache.h"

#include "windowed_kv_cache.h"

namespace Generators {

void SharedKeyValueCache::Update(DeviceSpan<int32_t> /*beam_indices*/, int total_length) {
  current_length_ = total_length;
}

void SharedKeyValueCache::RewindTo(size_t index) {
  CheckWindowedKvCacheRewind(windowed_cache_size_, current_length_, index);
}

}  // namespace Generators

