// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../scheduled_requests.h"

namespace Generators {

struct Decoder {
  Decoder() = default;

  virtual void Decode(ScheduledRequests& scheduled_requests) = 0;
};

}  // namespace Generators
