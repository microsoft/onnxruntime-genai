// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "engine.h"

namespace Generators {

Engine::Engine(std::shared_ptr<Model> model)
    : model_{model},
      cache_manager_{CacheManager::Create(model)},
      scheduler_{Scheduler::Create(model, cache_manager_)},
      model_executor_{std::make_unique<ModelExecutor>(model, cache_manager_)} {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  telemetry_id_ = AllocateEngineTelemetryId();
  dynamic_batching_ = cache_manager_->SupportsDynamicBatching();
#endif
}

void Engine::AddRequest(std::shared_ptr<Request> request) {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  request->Assign(shared_from_this(), model_->telemetry_session_id_, telemetry_id_,
                  dynamic_batching_);
#else
  request->Assign(shared_from_this(), 0, 0, false);
#endif
  scheduler_->AddRequest(request);
}

void Engine::RemoveRequest(std::shared_ptr<Request> request) {
  scheduler_->RemoveRequest(request);
  request->CompleteRemoval();
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
