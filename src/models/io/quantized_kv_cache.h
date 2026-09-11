// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../model.h"

namespace Generators {

int64_t ComputeQuantizedKvCacheHeadSize(int head_size, int kv_cache_quantization_bits, ONNXTensorElementDataType type);

}  // namespace Generators
