// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "engine.h"
#include "admission.h"
#include "decode_first_scheduler_policy.h"
#include "request_index.h"
#include "sequence_positions.h"

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

  size_t max_verification_rows = max_batch_size;
  if (engine_config.dynamic_batching) {
    max_verification_rows = std::max(
        max_verification_rows,
        engine_config.dynamic_batching->max_scheduled_tokens);
  }
  batched_sampling_plan_.Reserve(max_batch_size, max_verification_rows);
  batched_sampler_ = model->p_device_scoring_->CreateBatchedSampler(
      max_batch_size, model->config_->model.vocab_size);
}

ScheduledRequests Scheduler::CreateScheduledRequests(const StepPlan& plan) {
  return ScheduledRequests{plan, model_, GetBatchedSampler(),
                           GetBatchedSamplingPlan()};
}

StaticBatchScheduler::StaticBatchScheduler(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache_manager)
    : Scheduler{model}, model_{model}, cache_manager_{cache_manager} {}

void StaticBatchScheduler::AddRequest(std::shared_ptr<Request> request) {
  // The static batch decoder rebuilds its contiguous cache from the whole sequence every step, so it
  // cannot resume a half written prompt. Only the paged cache can hold one.
  if (request->SearchOptions().chunk_size.value_or(0) != 0) {
    throw std::runtime_error(
        "search.chunk_size requires dynamic batching; the static batch scheduler cannot chunk a prefill.");
  }
  if (auto* sampler = GetBatchedSampler())
    request->SamplingState(*sampler);
  requests_pool_.push_back(request);
}

void StaticBatchScheduler::RemoveRequest(std::shared_ptr<Request> request) {
  // For statically batched requests, memory is managed as a single block for the entire batch,
  // so individual requests cannot be deallocated until the whole batch is completed.
  // Therefore, deallocation is only performed for dynamically batched requests below.
  if (!cache_manager_->IsResident(request)) {
    requests_pool_.erase(
        std::remove(requests_pool_.begin(), requests_pool_.end(), request),
        requests_pool_.end());
  }
}

ScheduledRequests StaticBatchScheduler::Schedule() {
  const auto allocated_requests = cache_manager_->AllocatedRequests();

  for (const auto& request : allocated_requests) {
    if (IsQueued(request->status_)) {
      // Prepare before the queued-to-active transition so allocation failure leaves the request at
      // its last externally visible boundary.
      request->PrepareForStep(kMaxGeneratedTokenIndicesPerStep);
      request->Schedule();
    } else if (IsExecuting(request->status_)) {
      request->PrepareForStep(kMaxGeneratedTokenIndicesPerStep);
    }
  }

  std::vector<std::shared_ptr<Request>> requests_to_schedule;
  for (auto& request : requests_pool_) {
    if (IsQueued(request->status_) && !cache_manager_->IsResident(request)) {
      requests_to_schedule.push_back(request);
    }
  }

  constexpr size_t static_batch_size = 4;
  for (size_t batch_size = std::min(static_batch_size, requests_to_schedule.size());
       batch_size != 0; batch_size /= 2) {
    std::vector<std::shared_ptr<Request>> batch_requests(requests_to_schedule.begin(),
                                                         requests_to_schedule.begin() + batch_size);
    if (cache_manager_->CanAllocate(batch_requests)) {
      // Static cache allocation publishes the new batch immediately. Reserve output bookkeeping
      // first so a bad_alloc cannot leave cache ownership committed without a runnable request.
      for (auto& request : batch_requests) {
        request->PrepareForStep(kMaxGeneratedTokenIndicesPerStep);
      }

      // Before allocating, we need to ensure that the existing requests in the cache manager
      // are terminal and no longer need to remain in the scheduler pool.
      for (auto& request : allocated_requests) {
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
    if (IsExecutable(request->status_)) {
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
  throw std::logic_error(
      "Dynamic batching requires transactional step planning.");
}

StepPlanningResult DynamicBatchScheduler::PlanStep(StepPlan& plan) {
  plan.requests.clear();
  plan.scheduled_request_limit = 0;
  plan.token_count = 0;
  plan.proposed_block_table_columns = 0;
  plan.fixed_state = {};
  plan.graph_capture_eligible = false;

  struct Candidate {
    RequestStepPlan entry;
    DecodeFirstBudgetCandidate budget;
    size_t processed_sequence_length{};
  };
  const auto allocated_requests = cache_manager_->AllocatedRequests();
  std::vector<Candidate> candidates;
  candidates.reserve(allocated_requests.size() + requests_pool_.size());
  const size_t cache_query_token_cap = cache_manager_->MaxQueryTokensPerRequest();
  const size_t max_draft_token_count = cache_manager_->MaxDraftTokensPerStep();

  const auto add_candidate = [&candidates, cache_query_token_cap, max_draft_token_count](
                                 const std::shared_ptr<Request>& request,
                                 bool newly_admitted) {
    const auto snapshot = request->Snapshot();
    const bool valid_status =
        newly_admitted ? IsQueued(snapshot.status) : IsExecutable(snapshot.status);
    if (!valid_status) {
      throw StepPlanningConsistencyError(
          "Request status is invalid for dynamic step planning.");
    }
    const auto remaining_token_count =
        snapshot.current_sequence_length - snapshot.processed_sequence_length;
    if (remaining_token_count <= 0) {
      throw StepPlanningConsistencyError(
          "Cannot plan a request with no unprocessed tokens.");
    }

    Candidate candidate;
    candidate.entry.request = request;
    candidate.entry.request_id = request.get();
    candidate.entry.sequence_length_before = snapshot.current_sequence_length;
    // Drafts extend a decode step, which by definition ends at the sequence tail. A prefill chunk
    // has committed tokens of its own left to push through, so it can never verify one.
    candidate.entry.draft_token_count =
        snapshot.is_prefill
            ? 0
            : std::min(request->PendingDraftTokenCount(), max_draft_token_count);
    candidate.entry.unprocessed_token_count = 1 + candidate.entry.draft_token_count;
    candidate.entry.target_cache_slots = RequiredSlots(
        static_cast<size_t>(snapshot.processed_sequence_length),
        candidate.entry.unprocessed_token_count);
    candidate.entry.whole_sequence_cache_slots =
        SlotsForWholeSequence(snapshot.current_sequence_length);
    candidate.entry.is_prefill = snapshot.is_prefill;
    candidate.entry.newly_admitted = newly_admitted;
    auto prefill_token_cap = request->SearchOptions().chunk_size;
    if (cache_query_token_cap != 0 &&
        (prefill_token_cap.value_or(0) == 0 || *prefill_token_cap > cache_query_token_cap)) {
      prefill_token_cap = cache_query_token_cap;
    }
    candidate.budget = DecodeFirstBudgetCandidate{
        snapshot.is_prefill,
        static_cast<size_t>(remaining_token_count),
        prefill_token_cap,
        candidate.entry.draft_token_count,
    };
    candidate.processed_sequence_length =
        static_cast<size_t>(snapshot.processed_sequence_length);
    candidates.push_back(std::move(candidate));
  };

  for (const auto& request : allocated_requests) {
    if (IsTurnComplete(request->status_)) {
      continue;
    }
    add_candidate(request, false);
  }

  for (const auto& request : requests_pool_) {
    if (IsQueued(request->status_) && !cache_manager_->IsResident(request)) {
      add_candidate(request, true);
    }
  }

  std::vector<DecodeFirstBudgetCandidate> budget_candidates;
  budget_candidates.reserve(candidates.size());
  for (const auto& candidate : candidates)
    budget_candidates.push_back(candidate.budget);
  const auto order = DecodeFirstCandidateOrder(budget_candidates);
  plan.requests.reserve(candidates.size());
  std::vector<DecodeFirstBudgetCandidate> ordered_budget_candidates;
  std::vector<size_t> ordered_processed_lengths;
  ordered_budget_candidates.reserve(candidates.size());
  ordered_processed_lengths.reserve(candidates.size());
  for (size_t candidate_index : order) {
    plan.requests.push_back(candidates[candidate_index].entry);
    ordered_budget_candidates.push_back(candidates[candidate_index].budget);
    ordered_processed_lengths.push_back(
        candidates[candidate_index].processed_sequence_length);
  }

  const auto& dynamic_batching = *model_->config_->engine.dynamic_batching;
  plan.scheduled_request_limit = DecodeFirstProvisionalRequestLimit(
      dynamic_batching.max_scheduled_tokens,
      dynamic_batching.max_batch_size);

  // Budget the candidates most likely to be selected before cache feasibility evaluates their
  // growth. Later candidates remain one-token fallbacks so the cache can skip a blocked request
  // and admit a later fitting one without considering drafts the global budget cannot execute.
  const size_t budgeted_candidate_count =
      std::min(plan.requests.size(), plan.scheduled_request_limit);
  std::vector<size_t> token_counts;
  try {
    token_counts = AllocateDecodeFirstTokenBudget(
        std::span<const DecodeFirstBudgetCandidate>{ordered_budget_candidates}
            .first(budgeted_candidate_count),
        dynamic_batching.max_scheduled_tokens);
  } catch (const std::invalid_argument& error) {
    throw StepPlanningConsistencyError(error.what());
  }
  for (size_t index = 0; index < plan.requests.size(); ++index) {
    auto& entry = plan.requests[index];
    entry.unprocessed_token_count =
        index < token_counts.size() ? token_counts[index] : 1;
    entry.draft_token_count =
        entry.is_prefill ? 0 : entry.unprocessed_token_count - 1;
    entry.target_cache_slots = RequiredSlots(
        ordered_processed_lengths[index],
        entry.unprocessed_token_count);
  }

  RequestIndex candidate_request_ids{candidates.size()};
  for (size_t index = 0; index < candidates.size(); ++index) {
    if (!candidate_request_ids.Insert(
            candidates[index].entry.request_id, index)) {
      throw StepPlanningConsistencyError(
          "Scheduler candidates contain an invalid or duplicate request.");
    }
  }
  auto result = cache_manager_->PlanStepResources(plan);
  if (!result.executable) {
    return result;
  }

  if (plan.requests.size() > plan.scheduled_request_limit) {
    throw StepPlanningConsistencyError(
        "Cache planning selected more requests than the scheduled row limit.");
  }
  RequestIndex selected_request_ids{plan.requests.size()};
  size_t selected_token_count = 0;
  for (size_t index = 0; index < plan.requests.size(); ++index) {
    const auto& entry = plan.requests[index];
    const auto candidate_index =
        candidate_request_ids.Find(entry.request_id);
    if (!entry.request || entry.request_id != entry.request.get() ||
        !candidate_index ||
        candidates[*candidate_index].entry.request != entry.request ||
        !selected_request_ids.Insert(entry.request_id, index)) {
      throw StepPlanningConsistencyError(
          "Cache planning returned an invalid or duplicate request.");
    }
    if (entry.unprocessed_token_count == 0 ||
        selected_token_count > dynamic_batching.max_scheduled_tokens ||
        entry.unprocessed_token_count >
            dynamic_batching.max_scheduled_tokens - selected_token_count) {
      throw StepPlanningConsistencyError(
          "Cache planning exceeded the scheduled token budget.");
    }
    selected_token_count += entry.unprocessed_token_count;
  }

  for (size_t index = 0; index < plan.requests.size(); ++index) {
    plan.requests[index].scheduling_order = index;
  }
  cache_manager_->FinalizeStepResources(plan);
  cache_manager_->OrderStepForExecution(plan);

  // VarlenDecoderIO concatenates every request's pending tokens into one flat input. These offsets
  // describe that packed layout and identify the last logits row for each request, which is the row
  // used to sample its next token.
  size_t packed_token_offset = 0;
  plan.graph_capture_eligible = true;
  for (size_t i = 0; i < plan.requests.size(); ++i) {
    auto& entry = plan.requests[i];
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
    if (IsExecutable(request->status_)) {
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
