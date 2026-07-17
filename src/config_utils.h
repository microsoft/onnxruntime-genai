// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "config.h"

#include <string_view>

namespace Generators {

// Returns the KV cache quantization bit-width (0 = disabled, 4, or 8) configured for the
// given provider via the "kvCacheQuantizationBits" provider option. The caller must pass the
// provider name explicitly so the correct provider's options are inspected.
int GetKvCacheQuantizationBits(const Config::SessionOptions& session_options,
                               std::string_view provider_name);

}  // namespace Generators
