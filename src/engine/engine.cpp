// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "engine.h"

namespace Generators {

Engine::Engine(std::shared_ptr<Model> model)
    : model_{model},
      cache_manager_{CacheManager::Create(model)},
      scheduler_{Scheduler::Create(model, cache_manager_)},
      model_executor_{std::make_unique<ModelExecutor>(model, cache_manager_)} {}

void Engine::AddRequest(std::shared_ptr<Request> request) {
  request->Assign(shared_from_this());
  scheduler_->AddRequest(request);
}

void Engine::RemoveRequest(std::shared_ptr<Request> request) {
  scheduler_->RemoveRequest(request);
}

std::shared_ptr<Request> Engine::Step() {
  if (!HasPendingRequests()) {
    return nullptr;
  }

  if (!ready_requests_.empty()) {
    auto request = ready_requests_.front();
    ready_requests_.pop();
    return request;
  }

  if (auto scheduled_requests = scheduler_->Schedule()) {
    model_executor_->Decode(scheduled_requests);
    scheduled_requests.GenerateNextTokens();

    for (auto& request : scheduled_requests) {
      if (request->HasUnseenTokens()) {
        ready_requests_.push(request);
      }
    }
  }

  if (ready_requests_.empty()) {
    throw std::runtime_error("Expected at least one request to be ready, but none were found.");
  }

  auto request = ready_requests_.front();
  ready_requests_.pop();
  return request;
}

bool Engine::HasPendingRequests() const {
  return !ready_requests_.empty() || scheduler_->HasPendingRequests();
}

}  // namespace Generators
