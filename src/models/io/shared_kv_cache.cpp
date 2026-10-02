// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_kv_cache.h"

#include "windowed_kv_cache.h"

namespace Generators {

void SharedKeyValueCache::Update(DeviceSpan<int32_t> /*beam_indices*/, int total_length) {
  current_length_ = total_length;
}

void SharedKeyValueCache::RewindTo(size_t index) {
  if (static_cast<int64_t>(index) > current_length_)
    throw std::runtime_error("Cannot rewind the KV cache to " + std::to_string(index) +
                             " past its current length of " + std::to_string(current_length_) + ".");
  if (index != 0)
    CheckWindowedKvCacheRewind(windowed_cache_size_, current_length_, index);
  current_length_ = static_cast<int>(index);
}

bool SharedKeyValueCache::CanRewindTo(size_t index) const {
  // A full rewind needs no evicted history: the replay starts from a zero-length past.
  return static_cast<int64_t>(index) <= current_length_ &&
         (index == 0 || CanRewindWindowedKvCache(windowed_cache_size_, current_length_, index));
}

}  // namespace Generators
