// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../scheduled_requests.h"
#include "../../models/model.h"

namespace Generators {

struct ModelIO : State {
  ModelIO(const GeneratorParams& params, const Model& model) : State(params, model) {}

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {}) override {
    return {};
  }
};

struct Decoder {
  Decoder() = default;

  virtual void Decode(ScheduledRequests& scheduled_requests) = 0;
};

}  // namespace Generators
