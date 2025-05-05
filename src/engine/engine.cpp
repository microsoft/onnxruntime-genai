// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "engine.h"

namespace Generators {

Engine::Engine(std::shared_ptr<Model> model)
    : model_{model},
      cache_manager_{CreateCacheManager(model)},
      scheduler_{std::make_unique<Scheduler>(model, cache_manager_)},
      model_executor_{std::make_unique<ModelExecutor>(model, cache_manager_)} {}

void Engine::AddRequest(std::shared_ptr<Request> request) {
  request->Assign(shared_from_this());
  scheduler_->AddRequest(request);
}

void Engine::RemoveRequest(std::shared_ptr<Request> request) {
  scheduler_->RemoveRequest(request);
}

void Engine::Step() {
  if (auto scheduled_requests = scheduler_->Schedule()) {
    model_executor_->Decode(scheduled_requests);
    scheduled_requests.GenerateNextTokens();
  }
}

bool Engine::HasPendingRequests() const {
  return scheduler_->HasPendingRequests();
}

}  // namespace Generators
