// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "request.h"

namespace Generators {

struct ScheduledRequests {
  ScheduledRequests(std::vector<std::shared_ptr<Request>> requests,
                    std::shared_ptr<Model> model);

  std::unique_ptr<OrtRunOptions> RunOptions();

  std::shared_ptr<GeneratorParams> Params();

  auto begin() const {
    return requests_.begin();
  }

  auto end() const {
    return requests_.end();
  }

  size_t size() const {
    return requests_.size();
  }

  explicit operator bool() const { return !requests_.empty(); };

  auto operator[](size_t idx) const {
    assert(idx < requests_.size());
    return requests_[idx];
  }

 private:
  std::vector<std::shared_ptr<Request>> requests_;
  std::shared_ptr<Model> model_;
};

}  // namespace Generators
