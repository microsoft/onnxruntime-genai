// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "static_kv_cache.h"

namespace Generators {

namespace KeyValueCacheDetail {
std::unique_ptr<OrtValue> CreateLogicalKeyValueCacheOutputView(OrtValue& backing_tensor,
                                                               int logical_sequence_length);
}

struct SharedKeyValueCache : DefaultKeyValueCacheBase {
  using DefaultKeyValueCacheBase::DefaultKeyValueCacheBase;

  void Update(DeviceSpan<int32_t> beam_indices, int total_length) override;
  void RewindTo(size_t index) override;

 private:
  std::vector<std::unique_ptr<OrtValue>> output_views_;
  int output_view_length_{-1};
};

}  // namespace Generators
