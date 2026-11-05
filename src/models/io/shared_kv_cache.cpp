// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_kv_cache.h"

#include "windowed_kv_cache.h"

namespace Generators {

void SharedKeyValueCache::Update(DeviceSpan<int32_t> /*beam_indices*/, int total_length) {
  if (shared_cache_growth_) {
    shape_[2] = shared_cache_growth_->EnsureCapacity(total_length, current_length_ > 0, presents_, input_index_, output_index_);
    for (auto& shape : layer_shapes_) shape[2] = shape_[2];
  }
  current_length_ = total_length;
}

void SharedKeyValueCache::RewindTo(size_t index) {
  CheckWindowedKvCacheRewind(windowed_cache_size_, current_length_, index);
  current_length_ = static_cast<int>(index);
}

}  // namespace Generators
