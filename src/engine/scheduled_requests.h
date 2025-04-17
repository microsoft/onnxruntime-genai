// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "request.h"

namespace Generators {

struct ScheduledRequests {
  ScheduledRequests(std::vector<std::shared_ptr<Request>> requests);

  std::unique_ptr<OrtRunOptions> RunOptions();

  std::shared_ptr<GeneratorParams> Params() const;

  std::vector<std::shared_ptr<Request>> Requests();

  size_t NumRequests() const;

  void GenerateNextTokens(std::vector<DeviceSpan<float>>& logits);

  explicit operator bool() const;

 private:
  std::vector<std::shared_ptr<Request>> requests_;
};

}  // namespace Generators
