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

std::unique_ptr<BatchedSamplerState>
Scheduler::PrepareSamplingStateForRewind(
    const Request& request, uint64_t draw_count) const {
  const auto* state = request.SamplingStateForRewind();
  if (!state) {
    return nullptr;
  }
  if (!batched_sampler_ || !batched_sampler_->OwnsState(*state)) {
    throw std::runtime_error(
        "Request sampling state does not belong to its Engine scheduler.");
  }
  return batched_sampler_->CreateRewoundState(*state, draw_count);
}

std::unique_ptr<BatchedSamplerState> Scheduler::CreateSamplingState(
    const Request& request) const {
  if (auto* sampler = GetBatchedSampler()) {
    return sampler->CreateState(request.SearchOptions().random_seed);
  }
  return nullptr;
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
  requests_pool_.reserve(requests_pool_.size() + 1);
  auto sampling_state = CreateSamplingState(*request);
  request->CommitSamplingState(std::move(sampling_state));
  requests_pool_.push_back(std::move(request));
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

void StaticBatchScheduler::DetachRequestForTeardown(
    const std::shared_ptr<Request>& request) noexcept {
  cache_manager_->DetachRequestForTeardown(request);
  requests_pool_.erase(
      std::remove(requests_pool_.begin(), requests_pool_.end(), request),
      requests_pool_.end());
}

ScheduledRequests StaticBatchScheduler::Schedule() {
  const auto allocated_requests = cache_manager_->AllocatedRequests();

  for (const auto& request : allocated_requests) {
    if (IsQueued(request->status_)) {
      request->Schedule();
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
  requests_pool_.reserve(requests_pool_.size() + 1);
  auto sampling_state = CreateSamplingState(*request);
  request->CommitSamplingState(std::move(sampling_state));
  requests_pool_.push_back(std::move(request));
}

void DynamicBatchScheduler::RemoveRequest(std::shared_ptr<Request> request) {
  std::vector<std::shared_ptr<Request>> requests_to_remove{request};
  cache_manager_->Deallocate(requests_to_remove);

  requests_pool_.erase(std::remove(requests_pool_.begin(), requests_pool_.end(), request), requests_pool_.end());
}

void DynamicBatchScheduler::DetachRequestForTeardown(
    const std::shared_ptr<Request>& request) noexcept {
  cache_manager_->DetachRequestForTeardown(request);
  requests_pool_.erase(
      std::remove(requests_pool_.begin(), requests_pool_.end(), request),
      requests_pool_.end());
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
    size_t remaining_token_count{};
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
    candidate.remaining_token_count = static_cast<size_t>(remaining_token_count);
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
  for (size_t candidate_index : order) {
    auto entry = candidates[candidate_index].entry;
    entry.draft_token_count = 0;
    entry.unprocessed_token_count = 1;
    entry.target_cache_slots = RequiredSlots(
        candidates[candidate_index].processed_sequence_length, 1);
    plan.requests.push_back(std::move(entry));
  }

  const auto& dynamic_batching = *model_->config_->engine.dynamic_batching;
  plan.scheduled_request_limit = DecodeFirstProvisionalRequestLimit(
      dynamic_batching.max_scheduled_tokens,
      dynamic_batching.max_batch_size);

  auto result = cache_manager_->PlanStepResources(plan);
  if (!result.executable) {
    return result;
  }

  RequestIndex candidate_index{candidates.size()};
  for (size_t index = 0; index < candidates.size(); ++index) {
    if (!candidate_index.Insert(
            candidates[index].entry.request_id, index)) {
      throw StepPlanningConsistencyError(
          "Scheduler candidates contain an invalid or duplicate request.");
    }
  }
  std::vector<DecodeFirstBudgetCandidate> selected_candidates;
  std::vector<size_t> selected_processed_lengths;
  selected_candidates.reserve(plan.requests.size());
  selected_processed_lengths.reserve(plan.requests.size());
  for (const auto& entry : plan.requests) {
    const auto candidate_index_value =
        candidate_index.Find(entry.request_id);
    if (!candidate_index_value)
      throw StepPlanningConsistencyError(
          "Cache planning selected an unknown request.");
    const auto& candidate = candidates[*candidate_index_value];
    selected_candidates.push_back(candidate.budget);
    selected_processed_lengths.push_back(
        candidate.processed_sequence_length);
  }
  std::vector<size_t> token_counts;
  try {
    token_counts = AllocateDecodeFirstTokenBudget(
        selected_candidates, dynamic_batching.max_scheduled_tokens);
  } catch (const std::invalid_argument& error) {
    throw StepPlanningConsistencyError(error.what());
  }

  for (size_t index = 0; index < plan.requests.size(); ++index) {
    auto& entry = plan.requests[index];
    entry.scheduling_order = index;
    entry.unprocessed_token_count = token_counts[index];
    entry.draft_token_count = entry.is_prefill ? 0 : token_counts[index] - 1;
    entry.target_cache_slots = RequiredSlots(
        selected_processed_lengths[index],
        entry.unprocessed_token_count);
  }

  // Only drafts push a request's reserved slots past the probe that was just validated: a prefill
  // chunk reserves its whole sequence either way, and a draft-free decode still takes one slot. So
  // a plan without drafts needs no second planning pass, and must not pay for one every step.
  const bool plans_drafts = std::any_of(
      plan.requests.begin(), plan.requests.end(),
      [](const RequestStepPlan& entry) { return entry.draft_token_count != 0; });
  if (plans_drafts) {
    const size_t selected_request_count = plan.requests.size();
    auto budgeted_requests = plan.requests;
    while (true) {
      const auto resource_result = cache_manager_->PlanStepResources(plan);
      if (resource_result.executable &&
          plan.requests.size() == selected_request_count) {
        for (const auto& entry : plan.requests) {
          token_counts[entry.scheduling_order] = entry.unprocessed_token_count;
        }
        break;
      }

      RequestIndex planned_requests{plan.requests.size()};
      for (size_t index = 0; index < plan.requests.size(); ++index) {
        if (!planned_requests.Insert(plan.requests[index].request_id, index)) {
          throw StepPlanningConsistencyError(
              "Final cache planning selected a duplicate request.");
        }
      }

      const auto reduce_draft_count = [&](RequestStepPlan& entry) {
        --entry.draft_token_count;
        --entry.unprocessed_token_count;
        token_counts[entry.scheduling_order] = entry.unprocessed_token_count;
        entry.target_cache_slots = RequiredSlots(
            selected_processed_lengths[entry.scheduling_order],
            entry.unprocessed_token_count);
      };
      bool reduced_draft_count = false;
      for (auto& entry : budgeted_requests) {
        if (!planned_requests.Find(entry.request_id) &&
            entry.draft_token_count != 0) {
          reduce_draft_count(entry);
          reduced_draft_count = true;
        }
      }
      if (!reduced_draft_count) {
        const auto optional_work = std::find_if(
            budgeted_requests.rbegin(), budgeted_requests.rend(),
            [](const RequestStepPlan& entry) {
              return entry.draft_token_count != 0;
            });
        if (optional_work == budgeted_requests.rend()) {
          throw StepPlanningConsistencyError(
              "Final cache planning rejected required request work.");
        }
        reduce_draft_count(*optional_work);
      }
      plan.requests = budgeted_requests;
    }
  }
  cache_manager_->OrderStepForExecution(plan);

  // VarlenDecoderIO concatenates every request's pending tokens into one flat input. These offsets
  // describe that packed layout and identify the last logits row for each request, which is the row
  // used to sample its next token.
  size_t packed_token_offset = 0;
  plan.graph_capture_eligible = true;
  for (size_t i = 0; i < plan.requests.size(); ++i) {
    auto& entry = plan.requests[i];
    const size_t scheduling_index = entry.scheduling_order;
    entry.unprocessed_token_count = token_counts[scheduling_index];
    entry.target_cache_slots = RequiredSlots(
        selected_processed_lengths[scheduling_index],
        entry.unprocessed_token_count);
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
