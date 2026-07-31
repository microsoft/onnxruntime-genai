// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "engine.h"

namespace Generators {

Engine::Engine(std::shared_ptr<Model> model)
    : Engine(model, CreateDependencies(model)) {}

Engine::Engine(std::shared_ptr<Model> model, EngineDependencies dependencies)
    : model_{std::move(model)},
      cache_manager_{std::move(dependencies.cache_manager)},
      scheduler_{std::move(dependencies.scheduler)},
      model_executor_{std::move(dependencies.model_executor)} {
  // Fail fast on a missing collaborator rather than crashing later on first use.
  if (!cache_manager_) {
    throw std::runtime_error("Engine requires a non-null cache manager.");
  }
  if (!scheduler_) {
    throw std::runtime_error("Engine requires a non-null scheduler.");
  }
  if (!model_executor_) {
    throw std::runtime_error("Engine requires a non-null model executor.");
  }
}

EngineDependencies Engine::CreateDependencies(std::shared_ptr<Model> model) {
  std::shared_ptr<CacheManager> cache_manager = CacheManager::Create(model);
  auto scheduler = Scheduler::Create(model, cache_manager);
  auto model_executor = std::make_unique<ModelExecutor>(model, cache_manager);
  return EngineDependencies{std::move(cache_manager), std::move(scheduler), std::move(model_executor)};
}

void Engine::AddRequest(std::shared_ptr<Request> request) {
  request->Assign(shared_from_this());
  scheduler_->AddRequest(request);
}

void Engine::RemoveRequest(std::shared_ptr<Request> request) {
  scheduler_->RemoveRequest(request);
}

std::shared_ptr<Request> Engine::Step() {
  // An EOS-only batch may complete without adding anything to ready_requests_.
  while (HasPendingRequests()) {
    if (!ready_requests_.empty()) {
      auto request = ready_requests_.front();
      ready_requests_.pop();
      return request;
    }

    auto scheduled_requests = scheduler_->Schedule();
    model_executor_->Decode(scheduled_requests);
    scheduled_requests.GenerateNextTokens();

    for (auto& request : scheduled_requests) {
      if (request->HasUnseenTokens() || request->IsDone()) {
        ready_requests_.push(request);
      }
    }
  }

  return nullptr;
}

bool Engine::HasPendingRequests() const {
  return !ready_requests_.empty() || scheduler_->HasPendingRequests();
}

}  // namespace Generators
