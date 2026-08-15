// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "engine.h"
#include "admission.h"
#include "decode_first_scheduler_policy.h"
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

  batched_sampling_plan_.Reserve(max_batch_size);
  batched_sampler_ = model->p_device_scoring_->CreateBatchedSampler(
      max_batch_size, model->config_->model.vocab_size);
}

ScheduledRequests Scheduler::CreateScheduledRequests(const StepPlan& plan) {
  return ScheduledRequests{plan, model_, GetBatchedSampler(),
                           GetBatchedSamplingPlan()};
}

StaticBatchScheduler::StaticBatchScheduler(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache_manager)
    : Scheduler{model}, model_{model}, cache_manager_{cache_manager} {
  // Static batching allocates and releases the whole batch as a unit, so it cannot hand one
  // request's cache back to the pool. Say so instead of quietly ignoring the setting.
  const auto& engine_config = model->config_->engine;
  if (engine_config.dynamic_batching &&
      engine_config.dynamic_batching->enable_recompute_preemption) {
    throw std::runtime_error(
        "engine.dynamic_batching.enable_recompute_preemption requires dynamic batching; "
        "the static batch scheduler cannot suspend a single request.");
  }
}

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
    : Scheduler{model}, model_{model}, cache_manager_{cache_manager} {
  const auto& dynamic_batching = *model_->config_->engine.dynamic_batching;
  preemption_settings_ = RecomputePreemptionSettings{
      dynamic_batching.enable_recompute_preemption,
      dynamic_batching.max_preemptions_per_step,
      dynamic_batching.max_preemptions_per_request,
      dynamic_batching.min_decode_steps_before_preemption,
  };
  if (preemption_settings_.enabled &&
      !cache_manager_->SupportsRecomputePreemption()) {
    throw std::runtime_error(
        "engine.dynamic_batching.enable_recompute_preemption requires a cache manager that can "
        "release one request's blocks independently of the rest of the batch.");
  }
  if (preemption_settings_.enabled &&
      preemption_settings_.max_victims_per_step == 0) {
    throw std::invalid_argument(
        "engine.dynamic_batching.max_preemptions_per_step must be positive when recompute "
        "preemption is enabled.");
  }
}

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

StepPlanningResult DynamicBatchScheduler::PlanStep(
    StepPlan& plan, const StepPlanningLimits& limits) {
  // Completed requests release their blocks before admission, making that capacity available to
  // requests waiting in Assigned state during this same planning pass.
  ReapCompletedRequests();

  // A capacity retry is another attempt at the same engine step, so the requests this step already
  // suspended stay out of admission. Clearing the set there would let the retry hand the reclaimed
  // capacity straight back to the victim and leave the request it was taken for still blocked.
  if (!limits.retry_of_current_step)
    suspended_this_step_.clear();

  plan.new_admission_limit = 0;
  auto result = PlanStepOnce(plan, limits);
  if (!preemption_settings_.enabled) {
    return result;
  }

  // Two situations justify giving up a resident's committed cache:
  //   * a resident that cannot grow while nothing else can run, where the alternative is telling
  //     the caller the engine is out of capacity and making no progress at all; and
  //   * a waiting request that only free blocks are holding back, which is the head-of-line case
  //     recompute preemption exists to fix.
  // A starved resident is considered before any waiting request, so it would take the reclaimed
  // blocks first. When one exists and the step can still run other work, nothing is preempted:
  // reclaiming capacity that cannot reach the waiting request would discard committed work for
  // nothing. Row limits and residency limits are left alone for the same reason.
  const BlockCapacityShortfall shortfall =
      result.blocked_resident.Any()
          ? (result.executable ? BlockCapacityShortfall{} : result.blocked_resident)
          : result.blocked_admission;
  if (!shortfall.Any()) {
    return result;
  }

  // Preemption happens here, at a step boundary: the transaction has not been checkpointed yet, so
  // suspending a resident is a complete, self-consistent state change that a later rollback never
  // has to undo. It runs at most once per pass and only when the reclaimed blocks would cover the
  // shortfall, which is what keeps capacity from being traded back and forth.
  ++preemption_metrics_.block_starved_passes;
  const size_t new_admissions_before = CountNewAdmissions(plan);
  if (PreemptForBlockShortfall(shortfall) == 0) {
    return result;
  }
  // Spend the reclaimed capacity on the one blocked request it was measured for. Admitting more
  // waiting work would push the victim further back in the queue for blocks it released itself.
  plan.new_admission_limit = new_admissions_before + 1;
  return PlanStepOnce(plan, limits);
}

size_t DynamicBatchScheduler::CountNewAdmissions(const StepPlan& plan) {
  return static_cast<size_t>(
      std::count_if(plan.requests.begin(), plan.requests.end(),
                    [](const RequestStepPlan& entry) { return entry.newly_admitted; }));
}

size_t DynamicBatchScheduler::PreemptForBlockShortfall(
    const BlockCapacityShortfall& shortfall) {
  const auto residents = cache_manager_->AllocatedRequests();
  const auto cache_snapshot = cache_manager_->Snapshot();

  struct ResidentOwnership {
    size_t blocks{};
    size_t used_slots{};
  };
  const auto ownership_of = [&cache_snapshot](const void* request_id) {
    // Sum every table the cache holds for the request: a layout that splits one sequence across
    // more than one table still reports a single reclaimable total here.
    ResidentOwnership ownership;
    for (const auto& owner : cache_snapshot.requests) {
      if (owner.request_id == request_id) {
        ownership.blocks += owner.block_ids.size();
        ownership.used_slots += owner.used_slots;
      }
    }
    return ownership;
  };

  std::vector<RecomputePreemptionCandidate> candidates;
  candidates.reserve(residents.size());
  for (const auto& request : residents) {
    // A blocked resident cannot pay for its own growth by giving up the blocks it is growing from.
    if (request.get() == shortfall.request_id)
      continue;

    const auto ownership = ownership_of(request.get());
    // Only a request whose committed cache matches its committed sequence can be rewound to a
    // clean boundary. A mismatch means the cache and the request already disagree, so leave it
    // resident and let the invariant checks report it rather than rewinding on bad accounting.
    const bool ownership_agrees =
        ownership.used_slots == static_cast<size_t>(request->ProcessedSequenceLength());
    candidates.push_back(RecomputePreemptionCandidate{
        request.get(),
        ownership.blocks,
        request->PreemptionCount(),
        request->DecodeStepsSinceAdmission(),
        request->IsPreemptible() && ownership_agrees,
    });
  }

  const auto decision =
      SelectRecomputeVictims(candidates, shortfall.blocks, preemption_settings_);
  if (decision.Empty()) {
    ++preemption_metrics_.declined_preemptions;
    return 0;
  }

  for (const size_t index : decision.victims) {
    const auto victim = std::find_if(
        residents.begin(), residents.end(),
        [id = candidates[index].request_id](const std::shared_ptr<Request>& request) {
          return request.get() == id;
        });
    if (victim == residents.end())
      throw std::logic_error("Preemption selected a request the cache does not hold.");

    // Reclaim the cache first: if the pool rejects the release, the request has not been touched.
    // The request transition afterwards cannot fail, because eligibility already established both
    // its preconditions and that its ownership matches its committed sequence.
    const size_t committed_slots =
        static_cast<size_t>((*victim)->ProcessedSequenceLength());
    const auto reclaimed = cache_manager_->ReclaimRequestCache(*victim);
    if (reclaimed.released_slots != committed_slots) {
      throw std::runtime_error(
          "Reclaimed key-value slots do not match the request's committed sequence length.");
    }
    (*victim)->SuspendForRecompute();

    suspended_this_step_.push_back(candidates[index].request_id);
    ++preemption_metrics_.preemptions;
    preemption_metrics_.reclaimed_blocks += reclaimed.released_blocks;
    preemption_metrics_.recomputed_tokens += committed_slots;
  }

  ++preemption_metrics_.preemption_passes;
  return decision.victims.size();
}

StepPlanningResult DynamicBatchScheduler::PlanStepOnce(
    StepPlan& plan, const StepPlanningLimits& limits) {
  plan.requests.clear();
  plan.scheduled_request_limit = 0;
  plan.token_count = 0;
  plan.proposed_block_table_columns = 0;
  plan.graph_capture_eligible = false;

  struct Candidate {
    RequestStepPlan entry;
    DecodeFirstBudgetCandidate budget;
    size_t processed_sequence_length{};
  };
  std::vector<Candidate> candidates;

  const auto add_candidate = [&candidates](const std::shared_ptr<Request>& request,
                                           bool newly_admitted) {
    const auto snapshot = request->Snapshot();
    // A waiting candidate is either one that has never been admitted or one that was suspended to
    // free capacity; both own no cache blocks and are admitted the same way.
    const bool status_valid =
        newly_admitted ? (snapshot.status == RequestStatus::Assigned ||
                          snapshot.status == RequestStatus::Suspended)
                       : snapshot.status == RequestStatus::InProgress;
    if (!status_valid) {
      throw std::runtime_error("Request status is invalid for dynamic step planning.");
    }
    const auto remaining_token_count =
        snapshot.current_sequence_length - snapshot.processed_sequence_length;
    if (remaining_token_count <= 0) {
      throw std::runtime_error("Cannot plan a request with no unprocessed tokens.");
    }

    Candidate candidate;
    candidate.entry.request = request;
    candidate.entry.request_id = request.get();
    candidate.entry.sequence_length_before = snapshot.current_sequence_length;
    candidate.entry.unprocessed_token_count = 1;
    candidate.entry.target_cache_slots = RequiredSlots(
        static_cast<size_t>(snapshot.processed_sequence_length), 1);
    candidate.entry.whole_sequence_cache_slots =
        SlotsForWholeSequence(snapshot.current_sequence_length);
    candidate.entry.is_prefill = snapshot.is_prefill;
    candidate.entry.newly_admitted = newly_admitted;
    candidate.budget = DecodeFirstBudgetCandidate{
        snapshot.is_prefill,
        static_cast<size_t>(remaining_token_count),
        request->SearchOptions().chunk_size,
    };
    candidate.processed_sequence_length =
        static_cast<size_t>(snapshot.processed_sequence_length);
    candidates.push_back(std::move(candidate));
  };

  const auto allocated_requests = cache_manager_->AllocatedRequests();
  for (const auto& request : allocated_requests) {
    add_candidate(request, false);
  }

  // Waiting requests are considered suspended-first, then in arrival order within each group. A
  // request that already held capacity and gave it up is ahead of one that has never been admitted,
  // which is what stops a victim from falling behind the requests admitted with its own blocks.
  const auto add_waiting = [&](RequestStatus status) {
    for (const auto& request : requests_pool_) {
      if (request->status_ != status)
        continue;
      if (std::find(suspended_this_step_.begin(), suspended_this_step_.end(),
                    request.get()) != suspended_this_step_.end()) {
        continue;
      }
      add_candidate(request, true);
    }
  };
  add_waiting(RequestStatus::Suspended);
  add_waiting(RequestStatus::Assigned);

  std::vector<DecodeFirstBudgetCandidate> budget_candidates;
  budget_candidates.reserve(candidates.size());
  for (const auto& candidate : candidates)
    budget_candidates.push_back(candidate.budget);
  const auto order = DecodeFirstCandidateOrder(budget_candidates);
  const auto& dynamic_batching = *model_->config_->engine.dynamic_batching;
  const size_t max_scheduled_tokens = std::min(
      dynamic_batching.max_scheduled_tokens,
      limits.max_scheduled_tokens.value_or(
          dynamic_batching.max_scheduled_tokens));
  const size_t max_prefill_requests = std::min(
      dynamic_batching.max_batch_size,
      limits.max_prefill_requests.value_or(
          dynamic_batching.max_batch_size));
  if (max_scheduled_tokens == 0) {
    throw std::invalid_argument("The step token limit must be positive.");
  }
  const bool has_decode_candidate = std::any_of(
      candidates.begin(), candidates.end(),
      [](const Candidate& candidate) {
        return !candidate.budget.is_prefill;
      });
  if (max_prefill_requests == 0 && !has_decode_candidate) {
    throw std::invalid_argument(
        "A zero prefill limit requires runnable decode work.");
  }

  plan.requests.reserve(candidates.size());
  for (size_t candidate_index : order) {
    plan.requests.push_back(candidates[candidate_index].entry);
  }

  plan.scheduled_request_limit = DecodeFirstProvisionalRequestLimit(
      max_scheduled_tokens,
      dynamic_batching.max_batch_size);

  auto result = cache_manager_->PlanStepResources(plan);
  if (!result.executable) {
    return result;
  }

  size_t selected_prefill_requests = 0;
  const auto first_deferred = std::remove_if(
      plan.requests.begin(), plan.requests.end(),
      [&selected_prefill_requests,
       max_prefill_requests](const RequestStepPlan& entry) {
        if (!entry.is_prefill) {
          return false;
        }
        if (selected_prefill_requests < max_prefill_requests) {
          ++selected_prefill_requests;
          return false;
        }
        return true;
      });
  if (first_deferred != plan.requests.end()) {
    plan.requests.erase(first_deferred, plan.requests.end());
    result.capacity_deferred = true;
  }

  std::vector<DecodeFirstBudgetCandidate> selected_candidates;
  selected_candidates.reserve(plan.requests.size());
  for (const auto& entry : plan.requests) {
    const auto candidate = std::find_if(
        candidates.begin(), candidates.end(),
        [&entry](const Candidate& value) {
          return value.entry.request_id == entry.request_id;
        });
    if (candidate == candidates.end())
      throw std::logic_error("Cache planning selected an unknown request.");
    selected_candidates.push_back(candidate->budget);
  }
  const auto token_counts = AllocateDecodeFirstTokenBudget(
      selected_candidates, max_scheduled_tokens);

  // VarlenDecoderIO concatenates every request's pending tokens into one flat input. These offsets
  // describe that packed layout and identify the last logits row for each request, which is the row
  // used to sample its next token.
  size_t packed_token_offset = 0;
  plan.graph_capture_eligible = true;
  for (size_t i = 0; i < plan.requests.size(); ++i) {
    auto& entry = plan.requests[i];
    const auto candidate = std::find_if(
        candidates.begin(), candidates.end(),
        [&entry](const Candidate& value) {
          return value.entry.request_id == entry.request_id;
        });
    if (candidate == candidates.end())
      throw std::logic_error("Cache planning selected an unknown request.");
    entry.unprocessed_token_count = token_counts[i];
    entry.target_cache_slots = RequiredSlots(
        candidate->processed_sequence_length,
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
