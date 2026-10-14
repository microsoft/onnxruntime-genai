// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "static_kv_cache.h"

namespace Generators {

namespace KeyValueCacheDetail {
std::unique_ptr<OrtValue> CreateLogicalKeyValueCacheOutputView(OrtValue& backing_tensor,
                                                               int64_t logical_sequence_length);
int64_t GetLogicalKeyValueCacheOutputLength(OrtValue& backing_tensor,
                                            int total_length,
                                            bool use_graph_capture);
}

struct SharedKeyValueCache : DefaultKeyValueCacheBase {
  using DefaultKeyValueCacheBase::DefaultKeyValueCacheBase;

  void Update(DeviceSpan<int32_t> beam_indices, int total_length) override;
  void RewindTo(size_t index) override;

 private:
  std::vector<std::unique_ptr<OrtValue>> output_views_;
  std::vector<int64_t> output_view_lengths_;
};

}  // namespace Generators
