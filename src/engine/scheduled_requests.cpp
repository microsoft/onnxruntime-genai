// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "request.h"

#include "engine.h"
#include "../search.h"

namespace Generators {

namespace {

struct SampleArgs {
  int k{};
  float p{};
  float temperature{};
};

// Collapses Request::GenerateNextTokens' dispatch into the single (k, p, temperature) triple that
// each branch ends up handing to the sampler on CUDA, where SelectTop() is SampleTopKTopP(1, 0, 1),
// SampleTopK(k, t) is SampleTopKTopP(k, 1, t) and SampleTopP(p, t) is SampleTopKTopP(-1, p, t).
// Returns nothing for options the per-request path rejects, so that it keeps raising the error.
std::optional<SampleArgs> ResolveSampleArgs(const Config::Search& search) {
  if (!search.do_sample || search.top_k == 1 || search.temperature == 0)
    return SampleArgs{1, 0.0f, 1.0f};

  if (search.num_beams != 1 || search.top_p < 0.0f || search.top_p > 1.0f || search.top_k < 0)
    return std::nullopt;

  if (search.top_p > 0.0f && search.top_p < 1.0f && search.top_k > 1)
    return SampleArgs{search.top_k, search.top_p, search.temperature};
  if (search.top_k > 1)
    return SampleArgs{search.top_k, 1.0f, search.temperature};
  return SampleArgs{-1, search.top_p, search.temperature};
}

}  // namespace

ScheduledRequests::ScheduledRequests(std::vector<std::shared_ptr<Request>> requests,
                                     std::shared_ptr<Model> model)
    : requests_{requests}, model_{model} {
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

  std::vector<DeviceSpan<float>> logits = decoder_state_->ProcessLogits();
  if (logits.size() != requests_.size()) {
    throw std::runtime_error("Logits size does not match the number of requests.");
  }

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
}

// Samples every scheduled request in one call instead of once per request. The per-request path
// costs a sampler launch sequence and a full-vocabulary workspace per sequence, which dominates the
// decode step once the batch grows. Returns false when the batch does not qualify, leaving the
// requests untouched so the caller can fall back.
bool ScheduledRequests::TryGenerateNextTokensBatched(std::vector<DeviceSpan<float>>& logits) {
  const size_t batch_size = requests_.size();
  // A single request has nothing to share, and the engine's scratch buffer has to hold the batch.
  if (batch_size < 2 || shared_next_tokens_.size() < batch_size)
    return false;

  const auto args = ResolveSampleArgs(requests_[0]->SearchOptions());
  if (!args)
    return false;
  // Argmax ignores the random state, so batching cannot change its result. Sampling shares one
  // batch-wide generator, which a request that pinned a seed is entitled not to expect.
  const bool is_argmax = args->k == 1 && args->p == 0.0f;

  const size_t vocab_size = static_cast<size_t>(model_->config_->model.vocab_size);
  auto* first_row = logits[0].Span().data();

  for (size_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    // A completed request is skipped by the per-request loops, which would leave a hole in the
    // logits rows that the batched sampler cannot express.
    if (requests_[request_idx]->status_ == RequestStatus::Completed)
      return false;

    const auto& search = requests_[request_idx]->SearchOptions();
    const auto request_args = ResolveSampleArgs(search);
    if (!request_args || request_args->k != args->k || request_args->p != args->p ||
        request_args->temperature != args->temperature)
      return false;

    if (!is_argmax && search.random_seed != -1)
      return false;

    // The sampler reads one [batch_size, vocab_size] tensor, so the rows have to sit back to back
    // in request order within a single allocation. This also rules out a step that mixes prefill
    // and decode, where the rows are picked out of a larger tensor.
    if (logits[request_idx].size() != vocab_size ||
        !logits[request_idx].SameBufferAs(logits[0]) ||
        logits[request_idx].Span().data() != first_row + request_idx * vocab_size)
      return false;
  }

  // Only a CUDA greedy search accepts a shared slot. A partial bind is harmless: a bound search
  // still samples into its own slot on the per-request path and copies the shared buffer back
  // itself, so falling back below stays correct.
  for (size_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    if (!requests_[request_idx]->BindNextTokensSlot(shared_next_tokens_.subspan(request_idx, 1)))
      return false;
  }

  // Past this point the requests are mutated, so there is no going back to the per-request path:
  // the logits processors below are not idempotent.
  for (size_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    requests_[request_idx]->PrepareGeneration(logits[request_idx]);
  }

  auto batched_logits = logits[0].subspan(0, batch_size * vocab_size);
  if (!model_->p_device_scoring_->SampleTopKTopP(batched_logits, shared_next_tokens_.subspan(0, batch_size),
                                                 static_cast<int>(vocab_size), static_cast<int>(batch_size),
                                                 args->k, args->p, args->temperature))
    throw std::runtime_error("The scoring device accepted a shared next token buffer but cannot sample a batch.");

  for (size_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    requests_[request_idx]->OnNextTokensSampled();
  }

  // The one host/device synchronization of the step. It has to follow the tails above so that it
  // observes the tokens they pad for sequences that just hit an end-of-sequence token.
  shared_next_tokens_.CopyDeviceToCpu();

  for (size_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    requests_[request_idx]->CompleteGeneration();
  }

  return true;
}

}  // namespace Generators
