// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "prefix_snapshot_cache.h"

namespace Generators {

struct PagedWindowPrefixCheckpoint : PrefixSnapshotCache::Entry {
  size_t token_count{};
  size_t ring_blocks{};
};

}  // namespace Generators
