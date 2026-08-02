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
                                     BatchedSamplingPlan* sampling_plan,
                                     bool allow_chunked_prefill)
    : requests_{requests}, model_{model}, batched_sampler_{batched_sampler}, sampling_plan_{sampling_plan} {
  // Fixes what each request contributes to this step before anything reads UnprocessedTokens():
  // the cache sizing, the decoder inputs and the logits row selection all have to agree on it.
  for (auto& request : requests_) {
    request->ScheduleTokens(allow_chunked_prefill);
  }
}

std::unique_ptr<OrtRunOptions> ScheduledRequests::RunOptions() {
  return OrtRunOptions::Create();
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
    std::vector<DeviceSpan<float>> logits = decoder_state_->ProcessLogits();
    if (logits.size() != requests_.size()) {
      throw std::runtime_error("Logits size does not match the number of requests.");
    }

    // A request whose prompt is still being chunked ends this step in the middle of its own prompt,
    // so its last logits row predicts a token the prompt already supplies. It selects nothing and
    // only moves its cursor, which is what makes the next step resume where this one stopped.
    // The flags are snapshotted first because AdvanceChunk() moves the cursor that defines them.
    chunk_complete_.assign(requests_.size(), false);
    for (size_t request_idx = 0; request_idx < requests_.size(); ++request_idx) {
      auto& request = requests_[request_idx];
      if (request->status_ == RequestStatus::Completed)
        continue;
      chunk_complete_[request_idx] = request->IsChunkComplete();
      if (!chunk_complete_[request_idx])
        request->AdvanceChunk();
    }

    if (TryGenerateNextTokensBatched(logits))
      return;

    // Every request owns an independent single-sequence search, so token selection runs once per
    // request. Completing each one inline would block the host on the device once per request and
    // serialize the whole batch; launching all of them first means only the first completion below
    // actually waits for the device.
    for (size_t request_idx = 0; request_idx < requests_.size(); ++request_idx) {
      if (requests_[request_idx]->status_ != RequestStatus::Completed && chunk_complete_[request_idx]) {
        requests_[request_idx]->GenerateNextTokens(logits[request_idx]);
      }
    }

    for (size_t request_idx = 0; request_idx < requests_.size(); ++request_idx) {
      if (requests_[request_idx]->status_ != RequestStatus::Completed && chunk_complete_[request_idx]) {
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

// Samples all active requests through the scheduler-owned sampler. It owns the reusable workspace
// and groups rows by resolved sampling parameters, while each Request owns its persistent RNG state.
bool ScheduledRequests::TryGenerateNextTokensBatched(std::vector<DeviceSpan<float>>& logits) {
  if (!batched_sampler_ || !sampling_plan_)
    return false;

  sampling_plan_->Clear();

  for (size_t request_idx = 0; request_idx < requests_.size(); ++request_idx) {
    if (requests_[request_idx]->status_ == RequestStatus::Completed)
      continue;
    // Requests still mid-prompt have already advanced their cursor and select nothing this step.
    if (!chunk_complete_[request_idx])
      continue;

    const auto args = ResolveSampleArgs(requests_[request_idx]->SearchOptions());
    if (!args || !requests_[request_idx]->SupportsBatchedSampling())
      return false;
    sampling_plan_->requests.push_back(requests_[request_idx].get());
    sampling_plan_->logits.push_back(logits[request_idx]);
    sampling_plan_->params.push_back(*args);
    sampling_plan_->states.push_back(&requests_[request_idx]->SamplingState(*batched_sampler_));
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

}  // namespace Generators
