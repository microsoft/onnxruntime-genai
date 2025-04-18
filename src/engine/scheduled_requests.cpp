// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "request.h"

#include "engine.h"
#include "../search.h"

namespace Generators {

ScheduledRequests::ScheduledRequests(std::vector<std::shared_ptr<Request>> requests,
                                     std::shared_ptr<Model> model)
    : requests_{requests}, model_{model} {
}

std::unique_ptr<OrtRunOptions> ScheduledRequests::RunOptions() {
  return OrtRunOptions::Create();
}

std::shared_ptr<GeneratorParams> ScheduledRequests::Params() {
  return std::make_shared<GeneratorParams>(*model_);
}

}  // namespace Generators
