// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../models/decoder_only.h"
#include "scheduled_requests.h"

namespace Generators {

struct ModelIO {
  Decoder() = default;

  std::vector<const char*> input_names, output_names;
  std::vector<OrtValue*> inputs, outputs;
};

}  // namespace Generators
