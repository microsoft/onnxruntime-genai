// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_kv_cache.h"

#include "windowed_kv_cache.h"
#include "models/state-checkpoint.h"

namespace Generators {

void SharedKeyValueCache::Update(DeviceSpan<int32_t> /*beam_indices*/, int total_length) {
  current_length_ = total_length;
}

void SharedKeyValueCache::RewindTo(size_t index) {
  CheckWindowedKvCacheRewind(windowed_cache_size_, current_length_, index);
  current_length_ = static_cast<int>(index);
}

bool SharedKeyValueCache::BindCheckpoint(StateCheckpointDescription& description) {
  if (windowed_cache_size_ != 0) return false;
  for (size_t i = 0; i < presents_.size(); ++i)
    description.BindTensor(input_name_strings_[i], [this, i] { return presents_[i].get(); },
                           {StateCheckpointLayout::Kind::Sequence, 2});
  description.BindValue(current_length_);
  return true;
}

}  // namespace Generators
