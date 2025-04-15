// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "request.h"
#include "model_executor.h"
#include "scheduler.h"

namespace Generators {

struct Engine : std::enable_shared_from_this<Engine>,
                LeakChecked<Engine>,
                ExternalRefCounted<Engine> {
  Engine(std::shared_ptr<Model> model);

  void AddRequest(std::shared_ptr<Request> request);

  void RemoveRequest(std::shared_ptr<Request> request);

  void Step();

  bool HasPendingRequests() const;

 private:
  std::shared_ptr<Model> model_;
  std::unique_ptr<Scheduler> scheduler_;
  std::unique_ptr<ModelExecutor> model_executor_;
};

}  // namespace Generators
