// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "request.h"

#include "engine.h"
#include "../search.h"
#include <exception>

namespace Generators {

namespace {

// Collapses Request::GenerateNextTokens' dispatch into the single (k, p, temperature) triple that
// each branch ends up handing to the sampler on CUDA, where SelectTop() is SampleTopKTopP(1, 0, 1),
// SampleTopK(k, t) is SampleTopKTopP(k, 1, t) and SampleTopP(p, t) is SampleTopKTopP(-1, p, t).
// Returns nothing for options the per-request path rejects, so that it keeps raising the error.
std::optional<BatchedSamplingParams> ResolveSampleArgs(const Config::Search& search) {
  if (!search.do_sample || search.top_k == 1 || search.temperature == 0)
    return BatchedSamplingParams{1, 0.0f, 1.0f};

  if (search.num_beams != 1 || search.top_p < 0.0f || search.top_p > 1.0f || search.top_k < 0)
    return std::nullopt;

  if (search.top_p > 0.0f && search.top_p < 1.0f && search.top_k > 1)
    return BatchedSamplingParams{search.top_k, search.top_p, search.temperature};
  if (search.top_k > 1)
    return BatchedSamplingParams{search.top_k, 1.0f, search.temperature};
  return BatchedSamplingParams{-1, search.top_p, search.temperature};
}

}  // namespace

ScheduledRequests::ScheduledRequests(std::vector<std::shared_ptr<Request>> requests,
                                     std::shared_ptr<Model> model,
                                     BatchedSampler* batched_sampler,
                                     BatchedSamplingPlan* sampling_plan)
    : requests_{requests}, model_{model}, batched_sampler_{batched_sampler}, sampling_plan_{sampling_plan} {
}

std::unique_ptr<OrtRunOptions> ScheduledRequests::RunOptions() {
  return OrtRunOptions::Create();
}

ExecutionContext& ScheduledRequests::CreateExecutionContext() {
  execution_context_ = std::make_unique<ExecutionContext>();
  return *execution_context_;
}

std::shared_ptr<GeneratorParams> ScheduledRequests::Params() {
  if (!params_) {
    params_ = std::make_shared<GeneratorParams>(*model_);
  }
  return params_;
}

void ScheduledRequests::AddDecoderState(std::unique_ptr<DecoderIO> decoder_state) {
  decoder_state_ = std::move(decoder_state);
}

void ScheduledRequests::GenerateNextTokens() {
  if (!decoder_state_) {
    throw std::runtime_error("Cannot generate next tokens without the decoder state.");
  }

  try {
    auto logits = ProcessLogits();

    if (TryGenerateNextTokensBatched(logits))
      return;

    // Every request owns an independent single-sequence search, so token selection runs once per
    // request. Completing each one inline would block the host on the device once per request and
    // serialize the whole batch; launching all of them first means only the first completion below
    // actually waits for the device.
    for (size_t request_idx = 0; request_idx < requests_.size(); ++request_idx) {
      if (requests_[request_idx]->status_ != RequestStatus::Completed) {
        requests_[request_idx]->GenerateNextTokens(logits[request_idx]);
      }
    }

    for (size_t request_idx = 0; request_idx < requests_.size(); ++request_idx) {
      if (requests_[request_idx]->status_ != RequestStatus::Completed) {
        requests_[request_idx]->CompleteGeneration();
      }
    }
  } catch (...) {
    const auto error = std::current_exception();
    try {
      model_->p_device_scoring_->Synchronize();
    } catch (...) {
    }
    std::rethrow_exception(error);
  }
}

std::vector<DeviceSpan<float>> ScheduledRequests::ProcessLogits() {
  if (!decoder_state_) {
    throw std::runtime_error("Cannot process logits without the decoder state.");
  }

  std::vector<DeviceSpan<float>> logits = decoder_state_->ProcessLogits();
  if (logits.size() != requests_.size()) {
    throw std::runtime_error("Logits size does not match the number of requests.");
  }

  return logits;
}

// Samples all active requests through the scheduler-owned sampler. It owns the reusable workspace
// and groups rows by resolved sampling parameters, while each Request owns its persistent RNG state.
bool ScheduledRequests::TryGenerateNextTokensBatched(std::vector<DeviceSpan<float>>& logits) {
  if (!PrepareBatchedSamplingPlan(false))
    return false;

  for (size_t request_idx = 0; request_idx < requests_.size(); ++request_idx) {
    if (requests_[request_idx]->status_ != RequestStatus::Completed)
      sampling_plan_->logits.push_back(logits[request_idx]);
  }

  if (sampling_plan_->requests.empty())
    return true;

  for (size_t request_idx = 0; request_idx < sampling_plan_->requests.size(); ++request_idx) {
    sampling_plan_->requests[request_idx]->PrepareGeneration(sampling_plan_->logits[request_idx]);
  }

  auto next_tokens = batched_sampler_->Sample(sampling_plan_->logits, sampling_plan_->params,
                                              sampling_plan_->states,
                                              model_->config_->model.vocab_size);

  for (size_t request_idx = 0; request_idx < sampling_plan_->requests.size(); ++request_idx) {
    if (!sampling_plan_->requests[request_idx]->BindNextTokensSlot(next_tokens.subspan(request_idx, 1)))
      throw std::runtime_error("The scoring device supports batched sampling but the request search does not.");
    sampling_plan_->requests[request_idx]->OnNextTokensSampled();
  }

  next_tokens.CopyDeviceToCpu();

  for (auto* request : sampling_plan_->requests) {
    request->CompleteGeneration();
  }

  return true;
}

bool ScheduledRequests::PrepareBatchedSamplingPlan(
    bool require_transaction_support) {
  if (!batched_sampler_ || !sampling_plan_ ||
      (require_transaction_support && !batched_sampler_->SupportsTransactions())) {
    return false;
  }

  sampling_plan_->Clear();
  for (const auto& request : requests_) {
    if (request->status_ == RequestStatus::Completed)
      continue;

    const auto args = ResolveSampleArgs(request->SearchOptions());
    if (!args || !request->SupportsBatchedSampling()) {
      sampling_plan_->Clear();
      return false;
    }
    sampling_plan_->requests.push_back(request.get());
    sampling_plan_->params.push_back(*args);
    sampling_plan_->states.push_back(&request->SamplingState(*batched_sampler_));
  }
  return true;
}

void ScheduledRequests::BeginTransaction() {
  if (transaction_checkpoint_count_ != 0 || sampler_checkpoint_active_)
    throw std::logic_error("Scheduled request transaction is already active.");

  transaction_uses_batched_sampler_ = PrepareBatchedSamplingPlan(true);
  try {
    for (const auto& request : requests_) {
      request->SaveStateForTransaction(!transaction_uses_batched_sampler_);
      ++transaction_checkpoint_count_;
    }
    if (transaction_uses_batched_sampler_) {
      batched_sampler_->SaveStateForTransaction(sampling_plan_->states);
      sampler_checkpoint_active_ = true;
    }
  } catch (...) {
    const auto error = std::current_exception();
    try {
      RestoreStateForTransaction();
    } catch (...) {
    }
    std::rethrow_exception(error);
  }
}

void ScheduledRequests::GenerateNextTokensForTransaction(
    const StepPlan& plan,
    std::vector<RequestStepResult>& results) {
  if (plan.requests.size() != requests_.size() ||
      transaction_checkpoint_count_ != requests_.size()) {
    throw std::logic_error("Scheduled request transaction does not match the step plan.");
  }

  auto logits = ProcessLogits();
  results.clear();
  if (transaction_uses_batched_sampler_) {
    if (sampling_plan_->requests.size() != requests_.size())
      throw std::logic_error("Batched sampling plan does not match the scheduled requests.");

    sampling_plan_->logits.clear();
    for (size_t i = 0; i < requests_.size(); ++i) {
      sampling_plan_->logits.push_back(logits[i]);
      requests_[i]->PrepareGenerationForTransaction(logits[i]);
    }

    auto next_tokens = batched_sampler_->Sample(
        sampling_plan_->logits, sampling_plan_->params,
        sampling_plan_->states, model_->config_->model.vocab_size);
    for (size_t i = 0; i < requests_.size(); ++i) {
      if (!requests_[i]->BindNextTokensSlot(next_tokens.subspan(i, 1)))
        throw std::runtime_error("The request search rejected the batched sampler output.");
      requests_[i]->OnNextTokensSampled();
    }
    next_tokens.CopyDeviceToCpu();
    for (size_t i = 0; i < requests_.size(); ++i) {
      results.push_back(
          requests_[i]->StageGenerationForTransaction(plan.requests[i]));
    }
    return;
  }

  for (size_t i = 0; i < requests_.size(); ++i) {
    results.push_back(requests_[i]->ApplyLogitsForTransaction(logits[i]));
  }
}

void ScheduledRequests::RestoreStateForTransaction() {
  std::exception_ptr error;
  try {
    model_->p_device_scoring_->Synchronize();
  } catch (...) {
    error = std::current_exception();
  }

  if (sampler_checkpoint_active_) {
    try {
      batched_sampler_->RestoreStateForTransaction();
    } catch (...) {
      error = std::current_exception();
    }
    sampler_checkpoint_active_ = false;
  }

  std::vector<Request*> pending_restore_completion;
  pending_restore_completion.reserve(transaction_checkpoint_count_);
  while (transaction_checkpoint_count_ > 0) {
    auto* request = requests_[--transaction_checkpoint_count_].get();
    try {
      request->RestoreStateForTransaction(true);
      pending_restore_completion.push_back(request);
    } catch (...) {
      if (!error)
        error = std::current_exception();
    }
  }
  transaction_uses_batched_sampler_ = false;

  try {
    model_->p_device_scoring_->Synchronize();
  } catch (...) {
    if (!error)
      error = std::current_exception();
  }
  for (auto* request : pending_restore_completion) {
    try {
      request->CompleteStateRestoreForTransaction();
    } catch (...) {
      if (!error)
        error = std::current_exception();
    }
  }
  if (error)
    std::rethrow_exception(error);
}

void ScheduledRequests::CommitStateForTransaction() {
  if (transaction_checkpoint_count_ != requests_.size())
    throw std::logic_error("Scheduled request transaction is not active.");

  for (const auto& request : requests_) {
    request->CommitStateForTransaction();
  }
  transaction_checkpoint_count_ = 0;
  if (sampler_checkpoint_active_) {
    batched_sampler_->CommitStateForTransaction();
    sampler_checkpoint_active_ = false;
  }
  transaction_uses_batched_sampler_ = false;
}

}  // namespace Generators
