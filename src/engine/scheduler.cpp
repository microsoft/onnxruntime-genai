// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "engine.h"
#include "admission.h"

namespace Generators {

Scheduler::Scheduler(std::shared_ptr<Model> model)
    : model_{model} {
  constexpr size_t default_static_batch_size = 4;
  size_t max_batch_size = default_static_batch_size;
  const auto& engine_config = model->config_->engine;
  if (engine_config.dynamic_batching)
    max_batch_size = std::max(max_batch_size, engine_config.dynamic_batching->max_batch_size);
  if (engine_config.static_batching)
    max_batch_size = std::max(max_batch_size, engine_config.static_batching->max_batch_size);

  batched_sampling_plan_.Reserve(max_batch_size);
  batched_sampler_ = model->p_device_scoring_->CreateBatchedSampler(
      max_batch_size, model->config_->model.vocab_size);
}

ScheduledRequests Scheduler::CreateScheduledRequests(const StepPlan& plan) {
  std::vector<std::shared_ptr<Request>> requests;
  requests.reserve(plan.requests.size());
  for (const auto& entry : plan.requests) {
    requests.push_back(entry.request);
  }
  return ScheduledRequests{std::move(requests), model_, GetBatchedSampler(),
                           GetBatchedSamplingPlan()};
}

StaticBatchScheduler::StaticBatchScheduler(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache_manager)
    : Scheduler{model}, model_{model}, cache_manager_{cache_manager} {}

void StaticBatchScheduler::AddRequest(std::shared_ptr<Request> request) {
  if (auto* sampler = GetBatchedSampler())
    request->SamplingState(*sampler);
  requests_pool_.push_back(request);
}

void StaticBatchScheduler::RemoveRequest(std::shared_ptr<Request> request) {
  // For statically batched requests, memory is managed as a single block for the entire batch,
  // so individual requests cannot be deallocated until the whole batch is completed.
  // Therefore, deallocation is only performed for dynamically batched requests below.
  // we simply mark the request to be removed and it will be deallocated when the
  // entire batch is completed.
  to_be_removed_requests_.insert(request);
}

ScheduledRequests StaticBatchScheduler::Schedule() {
  std::vector<std::shared_ptr<Request>> requests_to_schedule;
  for (auto& request : requests_pool_) {
    if (request->status_ == RequestStatus::Assigned) {
      requests_to_schedule.push_back(request);
    }
  }

  constexpr size_t static_batch_size = 4;
  for (size_t batch_size = std::min(static_batch_size, requests_to_schedule.size());
       batch_size != 0; batch_size /= 2) {
    std::vector<std::shared_ptr<Request>> batch_requests(requests_to_schedule.begin(),
                                                         requests_to_schedule.begin() + batch_size);
    if (cache_manager_->CanAllocate(batch_requests)) {
      // Before allocating, we need to ensure that the existing requests in the cache manager
      // are complete and that if they were previously removed from the engine, they are no longer
      // in the requests pool.
      for (auto& request : cache_manager_->AllocatedRequests()) {
        if (request->status_ != RequestStatus::Completed && to_be_removed_requests_.count(request)) {
          throw std::runtime_error("Encountered a request that was removed from the engine but was not completed.");
        }
        requests_pool_.erase(std::remove(requests_pool_.begin(), requests_pool_.end(), request), requests_pool_.end());
      }

      cache_manager_->Allocate(batch_requests);
      for (auto& request : batch_requests) {
        request->Schedule();
      }
      requests_to_schedule.erase(requests_to_schedule.begin(), requests_to_schedule.begin() + batch_size);
      break;
    }
  }

  ScheduledRequests scheduled_requests(cache_manager_->AllocatedRequests(), model_, GetBatchedSampler(),
                                       GetBatchedSamplingPlan());

  if (!scheduled_requests) {
    throw std::runtime_error("Unable to schedule requests: no requests available or all requests are completed.");
  }

  return scheduled_requests;
}

bool StaticBatchScheduler::HasPendingRequests() const {
  for (auto& request : requests_pool_) {
    if (request->status_ != RequestStatus::Completed) {
      return true;
    }
  }
  return false;
}

DynamicBatchScheduler::DynamicBatchScheduler(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache_manager)
    : Scheduler{model}, model_{model}, cache_manager_{cache_manager} {}

void DynamicBatchScheduler::AddRequest(std::shared_ptr<Request> request) {
  if (auto* sampler = GetBatchedSampler())
    request->SamplingState(*sampler);
  requests_pool_.push_back(request);
}

void DynamicBatchScheduler::RemoveRequest(std::shared_ptr<Request> request) {
  std::vector<std::shared_ptr<Request>> requests_to_remove{request};
  cache_manager_->Deallocate(requests_to_remove);

  requests_pool_.erase(std::remove(requests_pool_.begin(), requests_pool_.end(), request), requests_pool_.end());
}

ScheduledRequests DynamicBatchScheduler::Schedule() {
  StepPlan plan;
  const auto result = PlanStep(plan);
  if (!result.executable) {
    if (result.outcome.kind == StepOutcomeKind::UnserviceableRequest) {
      throw std::runtime_error("A request cannot be serviced by the configured paged cache.");
    }
    throw std::runtime_error("Unable to schedule requests: no requests available or all requests are completed.");
  }

  std::vector<std::shared_ptr<Request>> newly_admitted;
  for (const auto& entry : plan.requests) {
    if (entry.newly_admitted) {
      newly_admitted.push_back(entry.request);
    }
  }
  if (!newly_admitted.empty()) {
    cache_manager_->Allocate(newly_admitted);
    for (const auto& request : newly_admitted) {
      request->Schedule();
    }
  }

  std::vector<std::shared_ptr<Request>> requests;
  requests.reserve(plan.requests.size());
  for (const auto& entry : plan.requests) {
    requests.push_back(entry.request);
  }
  return ScheduledRequests{std::move(requests), model_, GetBatchedSampler(),
                           GetBatchedSamplingPlan()};
}

void DynamicBatchScheduler::ReapCompletedRequests() {
  auto allocated_requests = cache_manager_->AllocatedRequests();
  std::vector<std::shared_ptr<Request>> completed_requests;
  std::copy_if(allocated_requests.begin(), allocated_requests.end(),
               std::back_inserter(completed_requests),
               [](const std::shared_ptr<Request>& request) {
                 return request->status_ == RequestStatus::Completed;
               });
  if (!completed_requests.empty()) {
    cache_manager_->Deallocate(completed_requests);
    requests_pool_.erase(
        std::remove_if(requests_pool_.begin(), requests_pool_.end(),
                       [](const std::shared_ptr<Request>& request) {
                         return request->status_ == RequestStatus::Completed;
                       }),
        requests_pool_.end());
  }
}

StepPlanningResult DynamicBatchScheduler::PlanStep(StepPlan& plan) {
  // Completed requests release their blocks before admission, making that capacity available to
  // requests waiting in Assigned state during this same planning pass.
  ReapCompletedRequests();

  plan.requests.clear();
  plan.token_count = 0;
  plan.proposed_block_table_columns = 0;
  plan.graph_capture_eligible = false;

  const auto add_request = [&plan](const std::shared_ptr<Request>& request,
                                   bool newly_admitted) {
    const auto snapshot = request->Snapshot();
    const RequestStatus expected_status =
        newly_admitted ? RequestStatus::Assigned : RequestStatus::InProgress;
    if (snapshot.status != expected_status) {
      throw std::runtime_error("Request status is invalid for dynamic step planning.");
    }
    const auto unprocessed_token_count =
        snapshot.current_sequence_length - snapshot.processed_sequence_length;
    if (unprocessed_token_count <= 0) {
      throw std::runtime_error("Cannot plan a request with no unprocessed tokens.");
    }

    RequestStepPlan entry;
    entry.request = request;
    entry.request_id = request.get();
    entry.sequence_length_before = snapshot.current_sequence_length;
    entry.unprocessed_token_count =
        static_cast<size_t>(unprocessed_token_count);
    entry.target_cache_slots = RequiredSlots(
        static_cast<size_t>(snapshot.processed_sequence_length),
        entry.unprocessed_token_count);
    entry.is_prefill = snapshot.is_prefill;
    entry.newly_admitted = newly_admitted;
    plan.requests.push_back(std::move(entry));
  };

  const auto allocated_requests = cache_manager_->AllocatedRequests();
  // Existing cache residents come first and retain block-table order. New requests follow them as
  // admission candidates; the cache planner may compact this list to the subset that fits.
  for (const auto& request : allocated_requests) {
    add_request(request, false);
  }
  const size_t committed_request_count = plan.requests.size();

  for (const auto& request : requests_pool_) {
    if (request->status_ == RequestStatus::Assigned) {
      add_request(request, true);
    }
  }

  auto result = cache_manager_->PlanStepResources(plan, committed_request_count);
  if (!result.executable) {
    return result;
  }

  // VarlenDecoderIO concatenates every request's pending tokens into one flat input. These offsets
  // describe that packed layout and identify the last logits row for each request, which is the row
  // used to sample its next token.
  size_t packed_token_offset = 0;
  plan.graph_capture_eligible = true;
  for (auto& entry : plan.requests) {
    entry.packed_token_offset = packed_token_offset;
    entry.logits_row_index =
        packed_token_offset + entry.unprocessed_token_count - 1;
    packed_token_offset += entry.unprocessed_token_count;
    plan.token_count += entry.unprocessed_token_count;
    plan.graph_capture_eligible &=
        !entry.is_prefill && entry.unprocessed_token_count == 1;
  }
  return result;
}

bool DynamicBatchScheduler::HasPendingRequests() const {
  for (auto& request : requests_pool_) {
    if (request->status_ != RequestStatus::Completed) {
      return true;
    }
  }
  return false;
}

std::unique_ptr<Scheduler> Scheduler::Create(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache_manager) {
  if (cache_manager->SupportsDynamicBatching()) {
    return std::make_unique<DynamicBatchScheduler>(model, cache_manager);
  }

  return std::make_unique<StaticBatchScheduler>(model, cache_manager);
}

}  // namespace Generators
